"""Generate ConceptLens experiments and turn them into an Observable bundle, in one command.

Steps (each is skipped when its output is already complete, so an interrupted run resumes):
  1. images    - find directions on a GAN checkpoint, then edit every code with every direction, written in the
                 original layout: <experiments>/<domain>-<method>-<layer>-<application>/
                 codes/{i}.jpg, walked/{i}-{d}.jpg, directions.pt, setting.json
  2. features  - CLIP ViT-H-14 (dfn5b) features, as the repo's process_experiment.py computes them (batched here,
                 see extract_features.py; --extractor repo runs the repo's script)
  3. export    - export_for_observable.py -> an attachable bundle
  4. parity    - (--parity) records the original server's answers on this data and checks the JavaScript port

Methods. Each returns unit directions in W (the 512-d space every style layer reads):
  sefa      SeFA (Shen & Zhou, CVPR 2021): the top eigenvectors of A^T A, where A stacks the chosen layers' style
            affine weights with unit-norm rows, as in the official code.
  ganspace  GANSpace-style PCA of the chosen layers' style vectors (s = A w + b) over 100k samples; the top
            components are mapped back to W by least squares, as the repo does.
  va        Vector arithmetic: differences between distinct pairs of 20 k-means centroids of W.
  svm       InterFaceGAN-style (Shen et al., CVPR 2020): CLIP ranks 10k generated images by a text concept, and
            a linear SVM separates the top of the ranking from the bottom in W.
  clip      CLIP-guided (new): for each text concept, the unit direction whose edit moves CLIP image
            embeddings along the text direction (neutral prompt -> concept prompt), optimised with the fixed
            step and layers used for rendering (StyleGAN-NADA's directional loss, as one global direction).

Edits: w + alpha * d with alpha = --edit_dist (5, as in the paper). --application local (default) adds it only to the
--layer layers, where SeFA and GANSpace looked for it; global adds it to every layer, as the paper's data did.

Fixes relative to the repo's data_generation/ scripts (see README for the full list):
  - Originals are rendered once from the same z as the walks and shared by every method (the scripts passed W codes
    to generator.forward, which expects z; and GPU rendering is not bit-exact, so per-method copies differed).
  - SeFA and GANSpace use the full weight matrix / all samples. The scripts took eigenvectors of random subsets
    (511 of the 2,048 weight rows; 32-1,024 of the samples) with arbitrary signs and then clustered or
    concatenated them, which gave noisy near-duplicates of a few axes.
  - SVM: CLIP ViT-B-32 is built with QuickGELU (the OpenAI weights need it; the repo got GELU), 10k samples instead
    of 1k, top vs bottom of the ranking instead of top 30% vs the rest, prompts scored against a neutral prompt,
    and the edited codes are no longer part of its training data.
  - GANSpace works on StyleGAN2, generated images are converted to 8-bit properly before CLIP sees them, walked
    images are written as they are made, reference PyTorch ops replace the custom CUDA ops, everything is seeded.

usage:
    python generate.py                                  # FFHQ 256, early layers, 5 methods x 48 directions, 200 codes
    python generate.py --domain s2_wild512 --concepts concepts-wild.txt
    python generate.py --application global --methods sefa,ganspace,va,svm   # the paper's global edits
"""
import argparse
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # the Concept-Lens-ObservableHQ folder: export script, parity/, bundle/
METHODS = ("sefa", "ganspace", "va", "svm", "clip")  # folder names; the first four match the original scripts
TEXT_METHODS = ("svm", "clip")  # one direction per text concept
PSI, CUTOFF = 0.7, 8  # truncation used everywhere in the original scripts
VERSION = 2  # direction-finding and rendering code; experiments made by older code are regenerated
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
TTY = sys.stdout.isatty()
sys.stdout.reconfigure(errors="replace")  # a Windows console or log may not encode every character


def default_repo():
    """The concept-lens checkout: the enclosing repository when this folder lives inside it (the remaster branch,
    <repo>/remaster/generation), otherwise the clone setup.ps1 makes in generation/concept-lens."""
    enclosing = os.path.dirname(PARENT)
    if all(os.path.isdir(os.path.join(enclosing, d)) for d in ("data_generation", "server")):
        return enclosing
    return os.path.join(HERE, "concept-lens")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", default=default_repo(), help="concept-lens checkout")
    p.add_argument("--domain", default="s2_ffhq256", help="a StyleGAN domain from the repo's _helpers.py, or swin_* (StyleSwin)")
    p.add_argument("--layer", default="early", help="layers the methods look at: early, middle, late or a list like [0,1]")
    p.add_argument("--application", default="local", choices=["local", "global"],
                   help="local: edit only the --layer layers; global: edit every layer, as the paper")
    p.add_argument("--methods", default=",".join(METHODS), help="comma-separated, in display order")
    p.add_argument("--n_codes", type=int, default=200)
    p.add_argument("--n_directions", type=int, default=48, help="per method")
    p.add_argument("--edit_dist", type=float, default=5, help="alpha in G(w + alpha d); the paper uses 5 for all methods")
    p.add_argument("--seed", type=int, default=121, help="codes use this seed, direction finding seed + 1")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_code_samples", type=int, default=100000, help="W samples for GANSpace and vector arithmetic")
    p.add_argument("--n_svm_samples", type=int, default=10000, help="images CLIP ranks for the SVM method")
    p.add_argument("--svm_fraction", type=float, default=0.2, help="SVM positives = this top share of the ranking, negatives = the same bottom share")
    p.add_argument("--clip_steps", type=int, default=150, help="optimisation steps per concept for the clip method")
    p.add_argument("--clip_batch", type=int, default=8, help="codes per optimisation step for the clip method")
    p.add_argument("--clip_lr", type=float, default=0.03)
    p.add_argument("--concepts", default=os.path.join(HERE, "concepts-ffhq.txt"))
    p.add_argument("--experiments", default=os.path.join(HERE, "experiments"), help="folder for experiment folders")
    p.add_argument("--bundle", default=None, help="export folder (default ../bundle/<domain>-<layer>-<application>)")
    p.add_argument("--images", type=int, default=128, help="thumbnail size in the bundle; 0 leaves images out")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="feature precision in the bundle")
    p.add_argument("--extractor", default="batched", choices=["batched", "repo"], help="batched: extract_features.py; repo: process_experiment.py")
    p.add_argument("--edist", action="store_true", help="also write the edit-distance strips (edist/), unused by the viewer")
    p.add_argument("--cuda-ops", action="store_true", help="compile StyleGAN's custom CUDA ops (needs MSVC and the CUDA toolkit)")
    p.add_argument("--redo", default="", help="steps to rerun even if done: images,features,export")
    p.add_argument("--skip", default="", help="steps to leave out: features,export")
    p.add_argument("--parity", action="store_true", help="record the original server's responses and run parity.mjs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--check", action="store_true", help="load the generator, render one image to check.jpg and stop")
    return p.parse_args()


# ── setup ────────────────────────────────────────────────────────────────────

def import_repo(repo):
    if not os.path.isdir(os.path.join(repo, "data_generation")):
        sys.exit(f"No concept-lens checkout at {repo}. Run setup.ps1 first, or pass --repo.")
    # The original scripts run from data_generation/ and also import data_generation.*; the NVIDIA
    # pickles need torch_utils and dnnlib importable at the top level.
    sys.path[:0] = [repo, os.path.join(repo, "data_generation")]
    if HERE not in sys.path:
        sys.path.append(HERE)  # adapters for other generator families (styleswin.py)


def use_reference_ops():
    """Make bias_act/upfirdn2d/filtered_lrelu take their pure-PyTorch path instead of compiling a plugin."""
    for package in ("data_generation.torch_utils.ops", "torch_utils.ops"):
        for name in ("bias_act", "upfirdn2d", "filtered_lrelu"):
            try:
                module = importlib.import_module(f"{package}.{name}")
            except Exception:
                continue
            module._init = lambda: False


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def style_weights_and_biases(generator):
    """StyleGAN2 version of the StyleGAN3-only helper GANSpace needs: the affine layers of the conv layers,
    in the same order as get_style_weights() (so layer indices mean the same thing for SeFA and GANSpace)."""
    weights, biases = {}, {}
    for name, value in generator.named_parameters():
        m = re.match(r"synthesis\..*conv\d\.affine\.(weight|bias)$", name)
        if m:
            (weights if m.group(1) == "weight" else biases)[name] = value.detach().cpu()
    return weights, biases


def load_generator(domain, device):
    """Returns the generator and a function from a --layer value to style-layer indices. Every generator
    offers the repo's interface: z_dim, w_dim, map_ws, forward, forward_walking, forward_ws, get_style_weights."""
    if domain.startswith("swin_"):
        import styleswin
        generator = styleswin.load(domain, device)
        return generator, generator.layer_indices

    from _helpers import parse_generator_fp, parse_layer_configuration, prepare_model

    path, n_layers = parse_generator_fp(domain)
    if not os.path.exists(path):
        sys.exit(f"Missing checkpoint {path}. Run setup.ps1 -Domain {domain} first.")
    generator = prepare_model(domain, path, device).eval().requires_grad_(False)
    if not hasattr(generator, "get_style_weights_and_biases"):
        generator.get_style_weights_and_biases = lambda: style_weights_and_biases(generator)
    return generator, lambda layer: parse_layer_configuration(layer, n_layers)


def read_concepts(path):
    """One concept per line. '# template: ...{}...' and '# neutral: ...' lines set the prompts; other # lines are comments."""
    concepts, prompts = [], {"template": "a photo of {}.", "neutral": "a photo."}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*(template|neutral)\s*:\s*(.+)$", line)
            if m:
                prompts[m.group(1)] = m.group(2).strip()
            elif line and not line.startswith("#"):
                concepts.append(line)
    if "{}" not in prompts["template"]:
        sys.exit(f"{path}: the template line needs {{}} where the concept goes")
    return concepts, prompts


def load_clip(name, device):
    """OpenAI CLIP weights with the QuickGELU activation they were trained with."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained="openai")
    return model.to(device).eval().requires_grad_(False), preprocess, open_clip.get_tokenizer(name)


def encode_text(model, tokenizer, texts, device):
    with torch.no_grad():
        t = model.encode_text(tokenizer(texts).to(device)).float()
    return t / t.norm(dim=-1, keepdim=True)


# ── directions ───────────────────────────────────────────────────────────────

def sample_w(generator, n, device, seed, full=False, batch=20000):
    """W codes as LatentCodeGenerator.sample_latent_codes (psi 0.7, cutoff 8), from their own random stream."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for lo in range(0, n, batch):
        z = torch.randn(min(batch, n - lo), generator.z_dim, generator=g).to(device)
        with torch.no_grad():
            ws = generator.map_ws(z, None, PSI, CUTOFF)
        out.append(ws if full else ws[:, 0, :])
    return torch.cat(out)


def orient(d):
    """Eigenvectors have no sign; fix one (largest-magnitude entry positive) so reruns give the same edits."""
    idx = d.abs().argmax(dim=1)
    return d * torch.sign(d[torch.arange(len(d)), idx])[:, None]


def chosen_weights(generator, layers):
    weights = list(generator.get_style_weights().values())
    missing = [i for i in layers if i >= len(weights)]
    if missing:
        sys.exit(f"layers {missing} have no style weights (the generator has {len(weights)} style layers)")
    return torch.cat([weights[i].detach().double().cpu() for i in layers])  # [sum of channels, w_dim]


def sefa_directions(generator, layers, n):
    """Official SeFA: eigenvectors of A^T A with A's rows (one per modulated channel) normalised, largest first."""
    A = chosen_weights(generator, layers)
    A = A / A.norm(dim=1, keepdim=True)
    values, vectors = torch.linalg.eigh(A.T @ A)
    return orient(vectors[:, torch.argsort(values, descending=True)[:n]].T.float())


def ganspace_directions(generator, layers, n, device, n_samples, seed):
    """PCA of the chosen layers' styles s = A w + b over n_samples codes, then back to W: d = argmin |A d - u|.
    The covariance of s is A cov(w) A^T, so the PCA is exact without materialising every style vector."""
    A = chosen_weights(generator, layers)
    w = sample_w(generator, n_samples, device, seed).double()
    wc = w - w.mean(0)
    cov_w = (wc.T @ wc / (len(wc) - 1)).cpu()
    values, vectors = torch.linalg.eigh(A @ cov_w @ A.T)
    u = vectors[:, torch.argsort(values, descending=True)[:n]]  # [channels, n]
    d = torch.linalg.lstsq(A, u, driver="gelsd").solution.T  # [n, w_dim]
    return orient((d / d.norm(dim=1, keepdim=True)).float())


def va_directions(generator, n, device, n_samples, seed):
    """Differences between distinct (unordered) pairs of 20 k-means centroids of W."""
    from sklearn.cluster import KMeans

    w = sample_w(generator, n_samples, device, seed)
    centroids = torch.tensor(KMeans(n_clusters=20, random_state=seed).fit(w.cpu().numpy()).cluster_centers_)
    pairs = [(a, b) for a in range(20) for b in range(a + 1, 20)]
    if n > len(pairs):
        sys.exit(f"vector arithmetic has {len(pairs)} distinct centroid pairs; asked for {n}")
    rng = np.random.default_rng(seed)
    directions = []
    for k in rng.choice(len(pairs), size=n, replace=False):
        a, b = pairs[k] if rng.random() < 0.5 else pairs[k][::-1]
        directions.append(centroids[a] - centroids[b])
    d = torch.stack(directions).float()
    return d / d.norm(dim=1, keepdim=True)


def render_for_clip(generator, ws, preprocess, model, device, batch_size):
    """CLIP image embeddings (normalised) of G(ws), through 8-bit images as a viewer would see them."""
    from utils.image_utils import convert_images_to_uint8

    out = []
    for lo in range(0, len(ws), batch_size):
        with torch.no_grad():
            images = generator.forward_ws(ws[lo:lo + batch_size], noise_mode="const")
            pixels = convert_images_to_uint8(images.float().cpu().numpy(), nchw_to_nhwc=True)
            batch = torch.stack([preprocess(Image.fromarray(p)) for p in pixels]).to(device)
            f = model.encode_image(batch).float()
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def svm_directions(generator, concepts, prompts, device, n_samples, fraction, seed, batch_size):
    """For each concept, CLIP scores n_samples generated images by (concept prompt - neutral prompt) similarity;
    a linear SVM on W separates the top `fraction` of the ranking from the bottom `fraction`."""
    from sklearn.svm import LinearSVC

    ws = sample_w(generator, n_samples, device, seed, full=True)
    model, preprocess, tokenizer = load_clip("ViT-B-32-quickgelu", device)
    image = render_for_clip(generator, ws, preprocess, model, device, batch_size)
    neutral = encode_text(model, tokenizer, [prompts["neutral"]], device)
    texts = encode_text(model, tokenizer, [prompts["template"].format(c) for c in concepts], device)
    scores = (image @ texts.T - image @ neutral.T).cpu().numpy()  # [samples, concepts]
    w0 = ws[:, 0, :].cpu().numpy()
    k = max(1, int(round(fraction * n_samples)))
    directions = []
    for j in range(len(concepts)):
        order = np.argsort(scores[:, j])
        rows = np.concatenate([order[-k:], order[:k]])
        labels = np.r_[np.ones(k), np.zeros(k)]
        normal = LinearSVC(random_state=seed, max_iter=20000).fit(w0[rows], labels).coef_.ravel()
        directions.append(torch.tensor(normal / np.linalg.norm(normal), dtype=torch.float32))
    del model
    return torch.stack(directions)


class ClipLoss:
    """StyleGAN-NADA's directional CLIP loss over two OpenAI CLIP backbones with random crops, so the
    direction has to change the image the way the text changes, for more than one model and view."""

    def __init__(self, device, prompts, views=2):
        import torch.nn.functional as F

        self.F, self.device, self.views = F, device, views
        self.models = [load_clip(name, device) for name in ("ViT-B-32-quickgelu", "ViT-B-16-quickgelu")]
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.prompts = prompts

    def text_directions(self, concept):
        out = []
        for model, _, tokenizer in self.models:
            t = encode_text(model, tokenizer, [self.prompts["template"].format(concept), self.prompts["neutral"]], self.device)
            d = t[0] - t[1]
            out.append(d / d.norm())
        return out

    def crops(self, images, g):
        """The same random square crops (80-100% of the side) for the original and edited batch; no flips,
        since some concepts are about left and right."""
        n, _, h, _ = images.shape
        out = []
        for _ in range(self.views):
            side = int(h * (0.8 + 0.2 * torch.rand(1, generator=g).item()))
            top, left = (int(torch.randint(0, h - side + 1, (1,), generator=g)) for _ in range(2))
            out.append(images[:, :, top:top + side, left:left + side])
        return out

    def embed(self, model, crops):
        x = torch.cat([self.F.interpolate(c, size=224, mode="bicubic", align_corners=False, antialias=True) for c in crops])
        x = ((x.clamp(-1, 1) + 1) / 2 - self.mean) / self.std
        e = model.encode_image(x).float()
        return e / e.norm(dim=-1, keepdim=True)

    def __call__(self, original, edited, targets, g):
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))
        loss, along = 0.0, 0.0
        for (model, _, _), t in zip(self.models, targets):
            gen_a, gen_b = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
            with torch.no_grad():
                e0 = self.embed(model, self.crops(original, gen_a))
            e1 = self.embed(model, self.crops(edited, gen_b))
            delta = e1 - e0
            loss = loss + (1 - self.F.cosine_similarity(delta, t[None], dim=-1)).mean()
            along = along + (delta @ t).mean().detach()
        return loss / len(self.models), along / len(self.models)


def clip_directions(generator, concepts, prompts, walk_layers, alpha, device, steps, batch, lr, seed):
    """New method. For each concept, a unit direction d in W, applied exactly as it will be rendered (w + alpha d on
    walk_layers), optimised with Adam so the change in CLIP image embeddings points along the text direction."""
    criterion = ClipLoss(device, prompts)
    directions, log = [], []
    for j, concept in enumerate(concepts):
        g = torch.Generator().manual_seed(seed * 1000 + j)
        v = torch.randn(generator.w_dim, generator=g).to(device)
        v = (v / v.norm()).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        targets = criterion.text_directions(concept)
        t0 = time.time()
        for step in range(steps):
            z = torch.randn(batch, generator.z_dim, generator=g).to(device)
            with torch.no_grad():
                ws = generator.map_ws(z, None, PSI, CUTOFF)
                original = generator.forward_ws(ws, noise_mode="const")
            d = v / v.norm()
            shift = torch.zeros(ws.shape[1], 1, device=device)
            shift[walk_layers if walk_layers is not None else slice(None)] = 1
            edited = generator.forward_ws(ws + alpha * shift[None] * d[None, None], noise_mode="const")
            loss, along = criterion(original, edited, targets, g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        d = (v / v.norm()).detach().float().cpu()
        directions.append(d)
        log.append({"concept": concept, "loss": round(float(loss), 4), "clip_shift": round(float(along), 4)})
        print(f"  clip {j + 1}/{len(concepts)} {concept!r}: directional loss {float(loss):.3f}, "
              f"shift along text {float(along):.3f} ({time.time() - t0:.0f}s)", flush=True)
    del criterion
    return torch.stack(directions), log


def step_in_sd(generator, directions, alpha, device, seed):
    """How far the edit goes, in standard deviations of W codes along each direction (truncated layers)."""
    w = sample_w(generator, 20000, device, seed).float()
    d = directions.to(device)
    sd = ((w - w.mean(0)) @ d.T).std(0)
    s = (alpha / sd).cpu()
    return {"median": round(float(s.median()), 2), "min": round(float(s.min()), 2), "max": round(float(s.max()), 2)}


# ── images ───────────────────────────────────────────────────────────────────

def save_batch(images, paths):
    from utils.image_utils import convert_images_to_uint8

    for pixels, path in zip(convert_images_to_uint8(images.float().cpu().numpy(), nchw_to_nhwc=True), paths):
        Image.fromarray(pixels).save(path)  # PIL's default JPEG quality, as the original scripts


def progress(done, total, lo, batch_size, label):
    if TTY:
        print(f"\r  {label}: {done}/{total} codes", end="", flush=True)
    elif done == total or (lo // batch_size) % 4 == 3:  # a few lines, for log files
        print(f"  {label}: {done}/{total} codes", flush=True)


def render_originals(generator, z_codes, folder, batch_size):
    os.makedirs(folder, exist_ok=True)
    for lo in range(0, len(z_codes), batch_size):
        z = z_codes[lo:lo + batch_size]
        with torch.no_grad():
            # The fix: originals from the same z as the walks (forward maps z; the scripts passed w).
            original, _ = generator.forward(z, None, PSI, CUTOFF, noise_mode="const")
        save_batch(original, [os.path.join(folder, f"{i}.jpg") for i in range(lo, lo + len(z))])
        progress(lo + len(z), len(z_codes), lo, batch_size, "originals")
    if TTY:
        print()


def render_walks(generator, z_codes, directions, edit_dist, walk_layers, root, batch_size):
    import torchvision

    walked_dir = os.path.join(root, "walked")
    os.makedirs(walked_dir, exist_ok=True)
    preview = []
    for lo in range(0, len(z_codes), batch_size):
        z = z_codes[lo:lo + batch_size]
        ids = range(lo, lo + len(z))
        for d, direction in enumerate(directions):
            with torch.no_grad():
                walked, _ = generator.forward_walking(z, None, direction, edit_dist, walk_layers, PSI, CUTOFF, noise_mode="const")
            save_batch(walked, [os.path.join(walked_dir, f"{i}-{d}.jpg") for i in ids])
            if lo == 0:
                preview.append(walked[:4].float().cpu())
        progress(lo + len(z), len(z_codes), lo, batch_size, "images")
    if TTY:
        print()
    # samples.jpg: the first 4 codes (rows) under every direction (columns), as the scripts wrote it.
    grid = torch.stack(preview).transpose(0, 1).flatten(0, 1)
    torchvision.utils.save_image(grid, os.path.join(root, "samples.jpg"), nrow=len(directions), normalize=True, value_range=(-1, 1))


def complete(root, setting):
    """True when this experiment was already generated with exactly these settings."""
    try:
        with open(os.path.join(root, "setting.json")) as f:
            done = json.load(f).get("generation")
    except (OSError, ValueError):
        return False
    return (done == setting and os.path.exists(os.path.join(root, "directions.pt"))
            and len(os.listdir(os.path.join(root, "codes"))) == setting["n_codes"]
            and len(os.listdir(os.path.join(root, "walked"))) == setting["n_codes"] * setting["n_directions"])


def generate(a, experiments, root_of):
    generator = layers_of = None
    concepts, prompts = read_concepts(a.concepts) if any(m in TEXT_METHODS for m in experiments) else ([], {})
    shared = os.path.join(a.experiments, f"_codes-{a.domain}-seed{a.seed}-n{a.n_codes}")  # originals, rendered once
    for method, name in experiments.items():
        root = root_of(name)
        n = min(a.n_directions, len(concepts)) if method in TEXT_METHODS else a.n_directions
        setting = {"version": VERSION, "domain": a.domain, "method": method, "layer": a.layer, "application": a.application,
                   "n_codes": a.n_codes, "n_directions": n, "edit_dist": a.edit_dist, "seed": a.seed,
                   "psi": PSI, "truncation_cutoff": CUTOFF}
        if method in TEXT_METHODS:
            setting.update(concepts=concepts[:n], prompts=prompts)
        if method == "svm":
            setting.update(n_svm_samples=a.n_svm_samples, svm_fraction=a.svm_fraction, clip="ViT-B-32-quickgelu openai")
        elif method == "clip":
            setting.update(clip_steps=a.clip_steps, clip_batch=a.clip_batch, clip_lr=a.clip_lr,
                           clip="ViT-B-32-quickgelu + ViT-B-16-quickgelu openai")
        elif method in ("ganspace", "va"):
            setting.update(n_code_samples=a.n_code_samples)
        if "images" not in a.redo and complete(root, setting):
            print(f"{name}: images already generated")
            continue
        if method in TEXT_METHODS and n < a.n_directions:
            print(f"  note: {a.concepts} has {len(concepts)} concepts, so {method} gets {n} directions")
        if generator is None:
            generator, layers_of = load_generator(a.domain, a.device)
            layers = layers_of(a.layer)
            walk_layers = layers if a.application == "local" else None
            torch.manual_seed(a.seed)  # same codes for every method (and as the first version of this script)
            z_codes = torch.randn(size=(a.n_codes, generator.z_dim)).to(a.device)
            if not (os.path.isdir(shared) and len(os.listdir(shared)) == a.n_codes) or "images" in a.redo:
                print(f"{a.domain}: rendering {a.n_codes} originals")
                shutil.rmtree(shared, ignore_errors=True)
                render_originals(generator, z_codes, shared + ".tmp", a.batch_size)
                os.replace(shared + ".tmp", shared)
        t0 = time.time()
        print(f"{name}: finding {n} directions (layers {layers}, {a.application} edits)")
        search_seed = a.seed + 1  # never the codes' stream, so no edited code is in any method's samples
        seed_everything(search_seed)
        info = {}
        if method == "sefa":
            directions = sefa_directions(generator, layers, n)
        elif method == "ganspace":
            directions = ganspace_directions(generator, layers, n, a.device, a.n_code_samples, search_seed)
        elif method == "va":
            directions = va_directions(generator, n, a.device, a.n_code_samples, search_seed)
        elif method == "svm":
            directions = svm_directions(generator, concepts[:n], prompts, a.device, a.n_svm_samples, a.svm_fraction,
                                        search_seed, a.batch_size)
        else:
            directions, info["optimisation"] = clip_directions(generator, concepts[:n], prompts, walk_layers, a.edit_dist,
                                                               a.device, a.clip_steps, a.clip_batch, a.clip_lr, search_seed)
        directions = directions.to(a.device).float()
        assert directions.shape == (n, generator.w_dim), directions.shape
        info["step_in_sd"] = step_in_sd(generator, directions, a.edit_dist, a.device, a.seed + 2)
        cos = (directions @ directions.T).abs().fill_diagonal_(0)
        info["max_abs_cosine_between_directions"] = round(float(cos.max()), 3)
        print(f"  step = {info['step_in_sd']['median']} SD of W along the median direction; "
              f"max |cos| between directions {info['max_abs_cosine_between_directions']}")

        # Start clean so stale files never mix with new ones (feature extraction counts files).
        for sub in ("codes", "walked", "edist", "tensors"):
            shutil.rmtree(os.path.join(root, sub), ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        if os.path.exists(os.path.join(root, "setting.json")):
            os.remove(os.path.join(root, "setting.json"))
        torch.save(directions.cpu(), os.path.join(root, "directions.pt"))
        shutil.copytree(shared, os.path.join(root, "codes"))  # byte-identical originals for every method
        render_walks(generator, z_codes, directions, a.edit_dist, walk_layers, root, a.batch_size)
        if a.edist:
            from common import ImageGenerator
            from torch.utils.data import DataLoader
            from utils.latent_code_dataset import LatentCodeDataset

            loader = DataLoader(LatentCodeDataset(z_codes), batch_size=a.batch_size, shuffle=False)
            ImageGenerator(generator, z_codes.device).apply_directions_edist(
                loader, directions, [0, 2, 5, 8, 11], os.path.join(root, "edist"))
        with open(os.path.join(root, "setting.json"), "w") as f:  # keys the server's request_setting reads, plus provenance
            json.dump({"experiment_name": name, "n_code": a.n_codes, "n_direction": n, "n_method": 1,
                       "feature_space": "clip", "generation": setting, "diagnostics": info}, f, indent=2)
        print(f"  done in {time.time() - t0:.0f}s")
    if generator is not None:
        del generator
        torch.cuda.empty_cache()


# ── the remaining steps ──────────────────────────────────────────────────────

def run(cmd, **kw):
    """Run a step; when output goes to a log instead of a console, turn progress-bar redraws into lines."""
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    if TTY:
        subprocess.run(cmd, check=True, env=env, **kw)
        return
    env.setdefault("TQDM_MININTERVAL", "20")  # one progress update every 20 s
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw)
    pending = b""
    while True:
        chunk = proc.stdout.read1(65536)
        if not chunk:
            break
        pieces = re.split(rb"[\r\n]", pending + chunk)
        pending = pieces.pop()
        for piece in pieces:
            if piece.strip():
                print(piece.decode("utf-8", "replace").rstrip(), flush=True)
    if pending.strip():
        print(pending.decode("utf-8", "replace").rstrip(), flush=True)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def extract(a, names, root_of):
    """Features for every experiment. The originals' features are computed once and copied, so every method
    carries identical original features (the server takes them from whichever method comes first)."""
    first_original = None
    for name in names:
        root = root_of(name)
        done = os.path.join(root, "tensors", "features.pt")
        if os.path.exists(done) and os.path.getmtime(done) > os.path.getmtime(os.path.join(root, "setting.json")) \
                and "features" not in a.redo:
            print(f"{name}: features already extracted")
        else:
            print(f"{name}: extracting CLIP features")
            if a.extractor == "repo":  # the repo's own extractor, one image at a time
                run([sys.executable, os.path.join(a.repo, "process_experiment.py"), "--root", root], cwd=a.repo)
            else:
                run([sys.executable, os.path.join(HERE, "extract_features.py"), "--root", root])
        original = os.path.join(root, "tensors", "features_original.pt")
        if first_original is None:
            first_original = original
        elif not torch.equal(torch.load(original), torch.load(first_original)):
            shutil.copyfile(first_original, original)
            print(f"  {name}: original features set to {os.path.basename(root_of(names[0]))}'s")


def main():
    a = parse_args()
    a.repo = os.path.abspath(a.repo)
    a.edit_dist = float(a.edit_dist)
    a.experiments = os.path.abspath(a.experiments)
    a.bundle = a.bundle or os.path.join(PARENT, "bundle", f"{a.domain}-{a.layer}-{a.application}")
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown method(s) {unknown}; choose from {', '.join(METHODS)}")
    experiments = {m: f"{a.domain}-{m}-{a.layer}-{a.application}" for m in methods}
    root_of = lambda name: os.path.join(a.experiments, name)
    print(f"device {a.device}" + (f" ({torch.cuda.get_device_name()})" if a.device == "cuda" else ""))
    torch.backends.cudnn.benchmark = False

    import_repo(a.repo)
    if not a.cuda_ops:
        use_reference_ops()
    if a.check:
        generator, _ = load_generator(a.domain, a.device)
        torch.manual_seed(a.seed)
        with torch.no_grad():
            image, _ = generator.forward(torch.randn(1, generator.z_dim).to(a.device), None, PSI, CUTOFF, noise_mode="const")
        save_batch(image, [os.path.join(HERE, "check.jpg")])
        print(f"generator works: wrote a {image.shape[-1]}px test image to {os.path.join(HERE, 'check.jpg')}")
        return
    generate(a, experiments, root_of)

    names = list(experiments.values())
    if "features" not in a.skip:
        extract(a, names, root_of)

    if "export" not in a.skip:
        print(f"exporting to {os.path.abspath(a.bundle)}")
        run([sys.executable, os.path.join(PARENT, "export_for_observable.py"), "--server", os.path.join(a.repo, "server"),
             "--data", a.experiments, "--experiments", ",".join(names), "--out", a.bundle, "--images", str(a.images),
             "--dtype", a.dtype])
        if a.parity:
            bundle = os.path.normpath(a.bundle)
            fixtures = f"{bundle}-fixtures.json"
            print("recording the original server's responses")
            run([sys.executable, os.path.join(PARENT, "parity", "record_fixtures.py"), "--server", os.path.join(a.repo, "server"),
                 "--data", a.experiments, "--experiments", ",".join(names), "--out", fixtures])
            script = os.path.join(PARENT, "parity", "parity.mjs")
            node, deno = shutil.which("node"), shutil.which("deno")
            if node:
                run([node, script, a.bundle, fixtures])
            elif deno:  # Deno runs parity.mjs unchanged (node:fs, process)
                run([deno, "run", "--allow-read", script, a.bundle, fixtures])
            else:
                print(f"Neither Node.js nor Deno is installed, so the comparison didn't run. Fixtures are in {fixtures}.")
    print("done")


if __name__ == "__main__":
    main()
