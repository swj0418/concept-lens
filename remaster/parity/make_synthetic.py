"""Synthetic ConceptLens experiments in the original on-disk format.

resources/experiments/<domain>-<method>-<layer>-<application>/
    setting.json
    tensors/features.pt           [code, direction, feature]  (walked end images)
    tensors/features_original.pt  [code, feature]              (original images)
    codes/{i}.jpg, walked/{i}-{d}.jpg                          (simple procedural faces with --images)

Features imitate CLIP image embeddings: unit vectors that are all fairly similar
(faces of one domain), with walks that move along a handful of shared semantic
axes so that directions form real clusters.
"""
import argparse, json, os, uuid
import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--root", default="synthetic/experiments")
p.add_argument("--n_code", type=int, default=200)
p.add_argument("--n_direction", type=int, default=40)
p.add_argument("--dim", type=int, default=1024)
p.add_argument("--methods", default="sefa,ganspace,vector_arithmetic,supervised")
p.add_argument("--domain", default="s2_ffhq256")
p.add_argument("--images", action="store_true")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

center = unit(rng.normal(size=a.dim))
codes = unit(center * 4.0 + rng.normal(size=(a.n_code, a.dim)) * 0.35 / np.sqrt(a.dim) * 12)
axes = unit(rng.normal(size=(12, a.dim)))               # shared semantic axes
code_gain = rng.uniform(0.4, 1.6, size=(a.n_code, 1))    # some codes react more

for m_i, method in enumerate(a.methods.split(",")):
    name = f"{a.domain}-{method}-early-global"
    root = os.path.join(a.root, name)
    os.makedirs(os.path.join(root, "tensors"), exist_ok=True)
    # each direction mixes 1-2 axes; methods differ in how clean they are
    mix = np.zeros((a.n_direction, 12))
    for d in range(a.n_direction):
        k = rng.integers(0, 12)
        mix[d, k] = rng.uniform(0.6, 1.4) * rng.choice([-1, 1])
        if rng.random() < 0.3 + 0.15 * m_i:
            mix[d, rng.integers(0, 12)] += rng.uniform(0.2, 0.6)
    strength = rng.uniform(0.15, 0.55, size=(a.n_direction, 1))
    delta = (mix @ axes) * strength                                   # [direction, dim]
    noise = rng.normal(size=(a.n_code, a.n_direction, a.dim)) * (0.02 + 0.01 * m_i) / np.sqrt(a.dim) * 8
    walked = unit(codes[:, None, :] + code_gain[:, :, None] * delta[None] + noise)
    torch.save(torch.from_numpy(walked.astype(np.float32)), os.path.join(root, "tensors", "features.pt"))
    torch.save(torch.from_numpy(codes.astype(np.float32)), os.path.join(root, "tensors", "features_original.pt"))
    with open(os.path.join(root, "setting.json"), "w") as f:
        json.dump({"experiment_name": name, "n_code": a.n_code, "n_direction": a.n_direction, "n_method": 1,
                   "feature_space": "clip", "uid": str(uuid.UUID(int=int(rng.integers(0, 2**62))))}, f, indent=2)
    if a.images:
        # Simple procedural faces: each semantic axis nudges one visual trait, so images echo the features.
        from PIL import ImageDraw
        os.makedirs(os.path.join(root, "codes"), exist_ok=True)
        os.makedirs(os.path.join(root, "walked"), exist_ok=True)
        def face(seed_code, shift, path):
            r = np.random.default_rng(seed_code)
            skin = np.array([230, 190, 160]) + r.integers(-30, 20, 3)
            hair = r.integers(20, 140, 3)
            bg = r.integers(150, 235, 3)
            t = np.zeros(12) if shift is None else shift
            size = 128
            im = Image.new("RGB", (size, size), tuple(int(v) for v in np.clip(bg + 40 * t[0:3], 0, 255)))
            d = ImageDraw.Draw(im)
            w = max(20, 36 + 8 * t[3])
            d.ellipse([64 - w, 26, 64 + w, 112], fill=tuple(int(v) for v in np.clip(skin + 30 * t[4], 0, 255)))
            d.pieslice([64 - w - 4, 18, 64 + w + 4, 80], 180, 360, fill=tuple(int(v) for v in np.clip(hair + 90 * t[5], 0, 255)))
            ey = 60 + 4 * t[6]
            for ex in (50, 78):
                h = max(1.0, 4 + 2 * t[7]); d.ellipse([ex - 5, ey - h, ex + 5, ey + h], fill=(40, 40, 50))
            smile = 10 * t[8]
            d.arc([48, 80 - abs(smile), 80, 96 + abs(smile)], 20 if smile >= 0 else 200, 160 if smile >= 0 else 340, fill=(120, 40, 40), width=3)
            if t[9] > 0.3:
                d.rectangle([40, 54, 88, 58], fill=(30, 30, 30))
            im.save(path, quality=85)
        for c in range(a.n_code):
            if m_i == 0 or not os.path.exists(os.path.join(root, "codes", f"{c}.jpg")):
                face(1000 + c, None, os.path.join(root, "codes", f"{c}.jpg"))
            for d in range(a.n_direction):
                face(1000 + c, mix[d] * float(code_gain[c, 0]) * float(strength[d, 0]) * 2.5, os.path.join(root, "walked", f"{c}-{d}.jpg"))
    print(name, walked.shape)
