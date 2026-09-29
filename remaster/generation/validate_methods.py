"""Check generate.py's direction methods against independent references, on a (small) finished run.

  1. structure: unit norms, overlap between directions, local vs global determinism, step size in SD of W
  2. SeFA against a transcription of the official code (genforce/sefa utils.factorize_weight)
  3. GANSpace against brute-force PCA of real style vectors from the generator's affine modules (gains and biases included)
  4. vector arithmetic: every direction is a normalised difference of two k-means centroids, pairs distinct
  5. the CLIP models used by svm/clip have QuickGELU
  6. svm/clip concept faithfulness under a held-out CLIP (ViT-H-14 dfn5b, the analysis model, which neither method used):
     does direction j raise its own concept's score, and more than the other concepts?
  7. edit size: cosine distance between walked and original ViT-H-14 features, per method and application

A test run to check, e.g. (16 codes, 8 directions, both applications; about 25 minutes on an RTX 3090):
    python generate.py --n_codes 16 --n_directions 8 --application local  --concepts concepts-test.txt --experiments test/exp --bundle test/bundle-local
    python generate.py --n_codes 16 --n_directions 8 --application global --concepts concepts-test.txt --experiments test/exp --bundle test/bundle-global
    python validate_methods.py test/exp test/validation.json [--v1 <folder with the first version's s2_ffhq256-*-early-global experiments>]
"""
import argparse
import json
import os
import re
import sys

sys.dont_write_bytecode = True  # importing generate.py must not leave __pycache__ behind
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import generate as g

p = argparse.ArgumentParser()
p.add_argument("experiments")
p.add_argument("out")
p.add_argument("--v1", default=None, help="folder with the first version's experiments (repo methods, global) to compare with")
p.add_argument("--domain", default="s2_ffhq256")
p.add_argument("--layer", default="early")
a = p.parse_args()
EXP, V1, DOMAIN, LAYER = a.experiments, a.v1, a.domain, a.layer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
report = {}


def section(title):
    print(f"\n=== {title}")


def folder(method, app, root=None):
    return os.path.join(root or EXP, f"{DOMAIN}-{method}-{LAYER}-{app}")


def exists(method, app):
    return os.path.exists(os.path.join(folder(method, app), "setting.json"))


def load_dirs(method, app):
    return torch.load(os.path.join(folder(method, app), "directions.pt")).double()


def setting(method, app):
    with open(os.path.join(folder(method, app), "setting.json")) as f:
        return json.load(f)


def abs_cos_matrix(x, y):
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    return (x @ y.T).abs()


g.import_repo(g.default_repo())
g.use_reference_ops()
gen, layers_of = g.load_generator(DOMAIN, DEVICE)
layers = layers_of(LAYER)
style_names = list(gen.get_style_weights().keys())
print(f"{LAYER} layers", layers, "->", [style_names[i] for i in layers])
report["layers"] = [style_names[i] for i in layers]
methods = [m for m in g.METHODS if exists(m, "local") or exists(m, "global")]
apps = [x for x in ("local", "global") if any(exists(m, x) for m in methods)]

# ── 1. structure and determinism
section("1. structure: unit norm, overlap between directions, local vs global determinism, step size")
report["structure"] = {}
for m in methods:
    app = "local" if exists(m, "local") else "global"
    d = load_dirs(m, app)
    c = abs_cos_matrix(d, d).fill_diagonal_(0)
    signed = (d @ d.T).fill_diagonal_(0)
    row = {"n": len(d), "norm_error": float((d.norm(dim=1) - 1).abs().max()), "max_abs_cos": round(float(c.max()), 4),
           "pairs_abs_cos_gt_0.9": int((c > 0.9).sum() // 2), "near_opposite_pairs": int((signed < -0.9).sum() // 2),
           "step_in_sd": setting(m, app).get("diagnostics", {}).get("step_in_sd")}
    if m != "clip" and exists(m, "local") and exists(m, "global"):  # clip is optimised for the walked layers
        row["local_equals_global"] = bool(torch.equal(d, load_dirs(m, "global")))
    report["structure"][m] = row
    print(m, row)

# ── 2. SeFA against the official implementation
if "sefa" in methods:
    section("2. SeFA vs official factorize_weight")
    app = "local" if exists("sefa", "local") else "global"
    weights = [w.detach().cpu().numpy().T for i, w in enumerate(gen.get_style_weights().values()) if i in layers]
    W = np.concatenate(weights, axis=1).astype(np.float32)
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    vals, vecs = np.linalg.eig(W.dot(W.T))
    order = np.argsort(-vals.real)
    ref = torch.tensor(vecs.real[:, order].T.copy()).double()
    ours = load_dirs("sefa", app)
    match = [float(abs_cos_matrix(ours[i:i + 1], ref[i:i + 1])) for i in range(len(ours))]
    print("eigenvalues (top 10):", np.round(vals.real[order][:10], 3))
    print("|cos| ours vs official, per rank:", np.round(match, 5))
    report["sefa"] = {"abs_cos_vs_official": match, "top_eigenvalues": [float(v) for v in vals.real[order][:10]]}
    if V1:
        v1 = torch.load(os.path.join(folder("sefa", "global", V1), "directions.pt")).double()
        best = abs_cos_matrix(v1, ref[:64]).max(dim=1)
        c = abs_cos_matrix(v1, v1).fill_diagonal_(0)
        print(f"v1 SeFA: median best |cos| with an official top-64 eigenvector {best.values.median():.3f}; "
              f"distinct eigenvectors hit {len(set(best.indices.tolist()))}; pairs |cos|>0.9: {int((c > 0.9).sum() // 2)}")
        report["sefa"]["v1_distinct_eigenvectors_hit"] = len(set(best.indices.tolist()))
        report["sefa"]["v1_pairs_abs_cos_gt_0.9"] = int((c > 0.9).sum() // 2)

# ── 3. GANSpace against brute-force PCA of actual style vectors
if "ganspace" in methods:
    section("3. GANSpace vs PCA of actual style vectors")
    app = "local" if exists("ganspace", "local") else "global"
    affines = [(n, mod) for n, mod in gen.synthesis.named_modules() if re.search(r"conv\d\.affine$", n)]
    chosen = [affines[i] for i in layers]
    print("affine modules used:", [n for n, _ in chosen])
    w = g.sample_w(gen, 30000, DEVICE, seed=999).float()
    with torch.no_grad():
        styles = torch.cat([mod(w) for _, mod in chosen], dim=1).double()
    sc = styles - styles.mean(0)
    _, S, Vh = torch.linalg.svd(sc, full_matrices=False)
    ours = load_dirs("ganspace", app)
    k = len(ours)
    A_eff = torch.cat([(mod.weight * mod.weight_gain).detach() for _, mod in chosen]).double().cpu()
    d_ref = torch.linalg.lstsq(A_eff, Vh[:k].T.cpu(), driver="gelsd").solution.T
    match = [float(abs_cos_matrix(ours[i:i + 1], d_ref[i:i + 1])) for i in range(k)]
    align = [float(torch.nn.functional.cosine_similarity((A_eff @ ours[i])[None], Vh[i].cpu()[None]).abs()) for i in range(k)]
    var = (S ** 2 / (S ** 2).sum()).cpu()
    print("explained variance of style PCs 1-10:", np.round(var[:10].numpy(), 4))
    print("|cos| ours vs brute force, per rank:", np.round(match, 5))
    print("|cos|(A d_ours, style PC):", np.round(align, 5))
    report["ganspace"] = {"abs_cos_vs_bruteforce": match, "style_change_alignment": align,
                          "explained_variance_top10": [float(v) for v in var[:10]]}
    if V1:
        v1 = torch.load(os.path.join(folder("ganspace", "global", V1), "directions.pt")).double()
        d48 = torch.linalg.lstsq(A_eff, Vh[:48].T.cpu(), driver="gelsd").solution.T
        best = abs_cos_matrix(v1, d48).max(dim=1)
        c = abs_cos_matrix(v1, v1).fill_diagonal_(0)
        print(f"v1 GANSpace: median best |cos| with a top-48 PC {best.values.median():.3f}; distinct PCs hit "
              f"{len(set(best.indices.tolist()))} {sorted(set(best.indices.tolist()))}; pairs |cos|>0.9: {int((c > 0.9).sum() // 2)}")
        report["ganspace"]["v1_distinct_pcs_hit"] = len(set(best.indices.tolist()))
        report["ganspace"]["v1_pairs_abs_cos_gt_0.9"] = int((c > 0.9).sum() // 2)

# ── 4. Vector arithmetic
if "va" in methods:
    section("4. VA: directions are distinct centroid differences")
    from sklearn.cluster import KMeans

    app = "local" if exists("va", "local") else "global"
    s = setting("va", app)["generation"]
    seed = s["seed"] + 1
    wva = g.sample_w(gen, s["n_code_samples"], DEVICE, seed).cpu().numpy()
    cent = torch.tensor(KMeans(n_clusters=20, random_state=seed).fit(wva).cluster_centers_).double()
    names = [(i, j) for i in range(20) for j in range(20) if i != j]
    diffs = torch.stack([cent[i] - cent[j] for i, j in names])
    ours = load_dirs("va", app)
    best = ((ours / ours.norm(dim=1, keepdim=True)) @ (diffs / diffs.norm(dim=1, keepdim=True)).T).max(dim=1)
    pairs = [names[i] for i in best.indices.tolist()]
    print("best cos with a centroid difference:", np.round(best.values.numpy(), 6))
    print("pairs:", pairs, "| unordered pairs distinct:", len({frozenset(x) for x in pairs}) == len(pairs))
    report["va"] = {"cos_with_centroid_difference": best.values.tolist(), "pairs": pairs,
                    "unordered_distinct": len({frozenset(x) for x in pairs}) == len(pairs)}

# ── 5. CLIP activation
section("5. the svm/clip CLIP models use QuickGELU")
import open_clip

for name in ("ViT-B-32-quickgelu", "ViT-B-16-quickgelu"):
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained="openai")
    act = type(model.visual.transformer.resblocks[0].mlp.gelu).__name__
    print(name, "activation:", act)
    report[f"activation {name}"] = act
    del model

# ── 6. text methods under a held-out CLIP
text_methods = [m for m in ("svm", "clip") if m in methods]
if text_methods:
    section("6. concept faithfulness under a held-out CLIP (ViT-H-14 dfn5b)")
    first = setting(text_methods[0], "local" if exists(text_methods[0], "local") else "global")["generation"]
    concepts, prompts = first["concepts"], first["prompts"]
    mh, _, _ = open_clip.create_model_and_transforms("ViT-H-14-quickgelu", pretrained="dfn5b")
    mh = mh.to(DEVICE).eval()
    tok = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    T = g.encode_text(mh, tok, [prompts["template"].format(c) for c in concepts], DEVICE).cpu().double()
    N = g.encode_text(mh, tok, [prompts["neutral"]], DEVICE).cpu().double()
    del mh
    torch.cuda.empty_cache()

    def gains(path, dir_idx, n_codes):
        f = torch.load(os.path.join(path, "tensors", "features.pt"))[:n_codes][:, dir_idx].double()
        f0 = torch.load(os.path.join(path, "tensors", "features_original.pt"))[:n_codes].double()
        f, f0 = f / f.norm(dim=-1, keepdim=True), f0 / f0.norm(dim=-1, keepdim=True)
        score = lambda x: x @ T.T - x @ N.T  # concept score relative to the neutral prompt
        return score(f) - score(f0)[:, None, :]  # [code, direction, concept]

    def summarise(label, per_code):
        G = per_code.mean(0)
        own = torch.diagonal(G)
        hits = int((G.argmax(dim=1) == torch.arange(len(G))).sum())
        col_hits = int((G.argmax(dim=0) == torch.arange(len(G))).sum())
        frac = torch.diagonal(per_code, dim1=1, dim2=2).gt(0).double().mean(0)
        print(f"{label:18s} own gain mean {float(own.mean()):+.4f} | own concept tops the gains of {hits}/{len(G)} directions;"
              f" own direction tops {col_hits}/{len(G)} concepts")
        print(f"{'':18s} " + ", ".join(f"{c[:12]} {float(x):+.3f} ({float(fr):.0%})" for c, x, fr in zip(concepts, own, frac)))
        return {"own_gain": [float(x) for x in own], "own_gain_mean": float(own.mean()),
                "direction_raises_own_concept_most": hits, "concept_raised_most_by_own_direction": col_hits,
                "fraction_codes_with_gain": [float(x) for x in frac]}

    report["concepts"] = {}
    for m in text_methods:
        for app in apps:
            if exists(m, app):
                n_codes = setting(m, app)["generation"]["n_codes"]
                report["concepts"][f"{m}-{app}"] = summarise(f"{m} {app}", gains(folder(m, app), list(range(len(concepts))), n_codes))
    if V1 and os.path.exists(folder("svm", "global", V1)):
        v1_names, _ = g.read_concepts(os.path.join(HERE, "concepts-ffhq.txt"))
        if all(c in v1_names for c in concepts):
            idx = [v1_names.index(c) for c in concepts]  # the first version's SVM direction for each test concept
            n_codes = setting(text_methods[0], apps[0])["generation"]["n_codes"]  # same z, so the same first codes
            report["concepts"]["svm-v1-global"] = summarise("v1 svm global", gains(folder("svm", "global", V1), idx, n_codes))

# ── 7. edit size
section("7. edit size: mean cosine distance between walked and original ViT-H-14 features")
report["edit_size"] = {}
for m in methods:
    row = []
    for app in apps:
        if exists(m, app) and os.path.exists(os.path.join(folder(m, app), "tensors", "features.pt")):
            f = torch.load(os.path.join(folder(m, app), "tensors", "features.pt")).double()
            f0 = torch.load(os.path.join(folder(m, app), "tensors", "features_original.pt")).double()
            v = float((1 - torch.nn.functional.cosine_similarity(f, f0[:, None, :], dim=-1)).mean())
            report["edit_size"][f"{m}-{app}"] = v
            row.append(f"{app} {v:.4f}")
    print(f"{m:9s} " + "   ".join(row))

with open(a.out, "w") as f:
    json.dump(report, f, indent=1)
print(f"\nwrote {a.out}")
