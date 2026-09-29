"""Export ConceptLens experiments as files that can be attached directly to an Observable notebook.

No server is involved afterwards: the notebook reads these files as FileAttachments and does
all the clustering and coherence math in the browser (conceptlens-core.js). Every file stays
under Observable's 50 MB per-attachment limit, so the same folder also works from any static host.

    manifest.json                      sizes, experiment order, file list
    features-000.f16, -001, ...        walked (end) CLIP features [code, direction, feature], float16,
                                       split by code into parts of at most --max-file-mb
    magnitude.f32                      ||walked - original|| per [code, direction] (exact, float32)
    direction-{metric}-{mask}.f64      all-code mean pairwise distances between directions, as the server
                                       computes them for the initial tree, for each combination of methods
                                       (mask bit i = experiment i; exact, condensed)
    code-euclidean.f64                 pairwise distances between codes (exact, condensed)
    sprite-{code}.jpg                  one image per code: its original image, then every walked image
    attachments.js                     a ready-to-paste notebook cell that attaches all of the above

The exact arrays come from the repo's own functions (server/conceptlens/utilities), so the
notebook's initial trees match the original server bit for bit.

usage:
    python export_for_observable.py --server <repo>/server --data <repo>/resources/experiments \\
        --experiments s2_ffhq256-sefa-early-global,s2_ffhq256-ganspace-early-global --out bundle/s2_ffhq256
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="path to the repo's server/ folder")
    p.add_argument("--data", required=True, help="folder that holds the experiment folders")
    p.add_argument("--experiments", required=True, help="comma-separated, in the order they should be combined")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="precision of the walked features (float32 doubles the size)")
    p.add_argument("--max-file-mb", type=float, default=45, help="split larger files (Observable allows 50 MB)")
    p.add_argument("--images", type=int, default=128, help="thumbnail size in pixels; 0 skips images")
    p.add_argument("--columns", type=int, default=16, help="thumbnails per row in each code's sprite")
    p.add_argument("--quality", type=int, default=80)
    a = p.parse_args()

    sys.path.insert(0, a.server)
    from conceptlens.utilities import features, readers  # the server's own code

    names = a.experiments.split(",")
    os.makedirs(a.out, exist_ok=True)
    limit = int(a.max_file_mb * 1e6)
    t0 = time.time()

    # Same loading path as the server (readers.py), including its reshape and concatenation.
    walk, cumulative = readers.read_walk_features(names, served_data_root=a.data)
    original = readers.read_original_features(names, served_data_root=a.data)
    walk, original = walk.float(), original.float()
    # The server takes original features from the first selected experiment; that is only
    # equivalent for every method combination if all experiments share the same codes.
    for name in names[1:]:
        other = readers.load_original_features(os.path.join(a.data, name)).float()
        if other.shape != original.shape or not torch.equal(other, original):
            print(f"warning: {name} has different original features than {names[0]}; "
                  "method combinations that start with it will not match the server exactly")
    n_code, n_direction, dim = walk.shape
    print(f"walk features {tuple(walk.shape)}, original {tuple(original.shape)}")

    # 1. Walked features, split by code so every part stays under the attachment limit.
    ext, np_dtype = ("f16", "<f2") if a.dtype == "float16" else ("f32", "<f4")
    stored = walk.numpy().astype(np_dtype)
    err = np.abs(stored.astype(np.float32) - walk.numpy())
    print(f"{a.dtype} rounding: max abs error {err.max():.2e}, mean abs error {err.mean():.2e}")
    per_code = n_direction * dim * stored.itemsize
    codes_per_part = max(1, limit // per_code)
    parts = []
    for i, lo in enumerate(range(0, n_code, codes_per_part)):
        name = f"features-{i:03d}.{ext}"
        stored[lo:lo + codes_per_part].tofile(os.path.join(a.out, name))
        parts.append({"path": name, "codes": [lo, min(n_code, lo + codes_per_part)]})

    # 2. Edit magnitude per (code, direction), exactly as postprocess_magnitude_features + torch.norm.
    magnitude = torch.norm(features.postprocess_magnitude_features(walk, original), dim=-1)
    magnitude.numpy().astype("<f4").tofile(os.path.join(a.out, "magnitude.f32"))

    # 3. Initial direction clustering distances, exactly as direction_hierarchy_initialization, for
    #    every combination of methods the notebook can show (the centering depends on the combination).
    w_mean = torch.mean(original, dim=0)
    ranges, start = [], 0
    for size in [int(s) for s in cumulative.tolist()]:
        ranges.append(list(range(start, size)))
        start = size
    k = len(names)
    masks = range(1, 2 ** k) if k <= 5 else sorted({2 ** k - 1} | {1 << i for i in range(k)})
    for mask in masks:
        dirs = [d for i in range(k) if mask >> i & 1 for d in ranges[i]]
        clustering = features.postprocess_clustering_features(walk[:, dirs], original, code_selection=None, mode="end",
                                                               prior_subset=True, centering="all_code_mean", w_mean=w_mean)
        for metric in ["cosine", "euclidean"]:
            d = features.compute_mean_features_pairwise_distances(clustering, dist_metric=metric)
            d.numpy().astype("<f8").tofile(os.path.join(a.out, f"direction-{metric}-{mask}.f64"))

    # 4. Code clustering distances: linkage(original_features) uses Euclidean pdist in float64.
    pdist(np.asarray(original.numpy(), dtype=np.float64), "euclidean").astype("<f8").tofile(
        os.path.join(a.out, "code-euclidean.f64"))

    sizes = [int(s) for s in cumulative.tolist()]
    experiments = []
    for i, name in enumerate(names):
        start = 0 if i == 0 else sizes[i - 1]
        setting_fp = os.path.join(a.data, name, "setting.json")  # optional: the generation scripts don't write it
        setting = json.load(open(setting_fp)) if os.path.exists(setting_fp) else None
        experiments.append({"name": name, "offset": start, "count": sizes[i] - start, "setting": setting})

    # 5. One sprite per code: cell 0 is the original image, cell 1 + d is direction d (global order).
    images = None
    if a.images:
        from PIL import Image

        s, cols = a.images, a.columns
        rows = math.ceil((n_direction + 1) / cols)
        missing = 0

        def thumb(path):
            nonlocal missing
            if not os.path.exists(path):
                missing += 1
                return None
            with Image.open(path) as im:
                im = im.convert("RGB")
                side = min(im.size)  # centre square, then scale
                left, top = (im.width - side) // 2, (im.height - side) // 2
                return im.crop((left, top, left + side, top + side)).resize((s, s), Image.LANCZOS)

        for c in range(n_code):
            sheet = Image.new("RGB", (cols * s, rows * s), (235, 234, 228))
            cells = [os.path.join(a.data, names[0], "codes", f"{c}.jpg")]
            for e in experiments:
                cells += [os.path.join(a.data, e["name"], "walked", f"{c}-{d}.jpg") for d in range(e["count"])]
            for i, path in enumerate(cells):
                t = thumb(path)
                if t is not None:
                    sheet.paste(t, ((i % cols) * s, (i // cols) * s))
            sheet.save(os.path.join(a.out, f"sprite-{c}.jpg"), quality=a.quality, optimize=True, progressive=True)
        images = {"sprite": "sprite-{code}.jpg", "cell": s, "columns": cols, "rows": rows,
                  "layout": "cell 0 = original image, cell 1 + d = walked image for direction d"}
        print(f"sprites: {n_code} written" + (f" ({missing} source images missing)" if missing else ""))

    manifest = {
        "format": "conceptlens-bundle/2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_code": n_code,
        "n_direction": n_direction,
        "feature_dim": dim,
        "experiments": experiments,
        "files": {
            "features": {"dtype": a.dtype, "shape": [n_code, n_direction, dim], "parts": parts},
            "magnitude": {"path": "magnitude.f32", "dtype": "float32", "shape": [n_code, n_direction]},
            "direction_distance": {"pattern": "direction-{metric}-{mask}.f64", "dtype": "float64",
                                   "masks": list(masks), "metrics": ["cosine", "euclidean"],
                                   "note": "mask bit i = experiment i in the order above"},
            "code_distance": {"path": "code-euclidean.f64", "dtype": "float64", "condensed": n_code},
        },
        "images": images,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 6. The notebook cell that attaches every file (Observable needs literal FileAttachment names).
    files = sorted(n for n in os.listdir(a.out) if n not in ("attachments.js",))
    files.sort(key=lambda n: (not n.startswith("manifest"), n.startswith("sprite-"),
                              int(n[7:-4]) if n.startswith("sprite-") else 0, n))
    with open(os.path.join(a.out, "attachments.js"), "w") as f:
        f.write("// Paste this as one notebook cell after uploading every file in this folder\n")
        f.write("// (drag them all onto the notebook's Files pane).\n")
        f.write("files = ({\n")
        f.write(",\n".join(f'  "{n}": FileAttachment("{n}")' for n in files))
        f.write("\n})\n")

    sizes_mb = {n: os.path.getsize(os.path.join(a.out, n)) / 1e6 for n in files}
    largest = max(sizes_mb, key=sizes_mb.get)
    print(f"wrote {len(files)} files, {sum(sizes_mb.values()):.1f} MB in total "
          f"(largest {largest}: {sizes_mb[largest]:.1f} MB) in {time.time() - t0:.1f}s")
    if sizes_mb[largest] > 50:
        print("warning: a file is larger than Observable's 50 MB attachment limit")


if __name__ == "__main__":
    main()
