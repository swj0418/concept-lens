"""CLIP features for one experiment folder, computed as the repo's process_experiment.py does, but in batches.

Same model (open_clip ViT-H-14-quickgelu, dfn5b), same preprocessing (the model's own transform applied to each
JPEG as loaded by PIL), same unnormalized encode_image output in float32, and the same files:
    tensors/features_original.pt   [n_code, 1024]               codes/{i}.jpg
    tensors/features.pt            [n_code, n_direction, 1024]  walked/{i}-{d}.jpg
The repo feeds one image at a time; batching only changes float rounding (see --compare).

usage:
    python extract_features.py --root <experiment folder> [--batch_size 64] [--workers 4]
    python extract_features.py --root <experiment folder> --compare 512   # check against existing tensors, write nothing
"""
import argparse
import os
import re
import sys
import time

import torch
from PIL import Image

MODEL, PRETRAINED = "ViT-H-14-quickgelu", "dfn5b"  # processing_utils.CLIPTransform


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.preprocess(image)


def dataset_size(walked_dir):
    """As process_experiment.infer_dataset_size: codes from the file names, directions counted for the first code."""
    pairs = [m.groups() for m in (re.fullmatch(r"(\d+)-(\d+)\.jpg", f) for f in os.listdir(walked_dir)) if m]
    codes = sorted({int(c) for c, _ in pairs})
    n_direction = sum(1 for c, _ in pairs if int(c) == codes[0])
    if len(pairs) != len(codes) * n_direction or codes != list(range(len(codes))):
        sys.exit(f"{walked_dir}: expected every code 0..n-1 to have the same number of walked images")
    return len(codes), n_direction


def encode(model, paths, preprocess, batch_size, workers, device, label):
    loader = torch.utils.data.DataLoader(Images(paths, preprocess), batch_size=batch_size, num_workers=workers,
                                         pin_memory=True, persistent_workers=False)
    out, done, t0, last = [], 0, time.time(), 0.0
    with torch.no_grad():
        for batch in loader:
            out.append(model.encode_image(batch.to(device, non_blocking=True)).float().cpu())
            done += len(batch)
            if time.time() - last > 30 or done == len(paths):
                last = time.time()
                rate = done / max(last - t0, 1e-9)
                print(f"  {label}: {done}/{len(paths)} images, {rate:.0f}/s", flush=True)
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--compare", type=int, default=0, help="encode this many walked images and compare with tensors/features.pt")
    a = p.parse_args()

    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=MODEL, pretrained=PRETRAINED)
    model = model.to(device).eval()

    walked, codes, tensors = (os.path.join(a.root, d) for d in ("walked", "codes", "tensors"))
    n_code, n_direction = dataset_size(walked)
    walk_paths = [os.path.join(walked, f"{c}-{d}.jpg") for c in range(n_code) for d in range(n_direction)]
    code_paths = [os.path.join(codes, f"{c}.jpg") for c in range(n_code)]
    print(f"{a.root}: {n_code} codes x {n_direction} directions", flush=True)

    if a.compare:
        ref = torch.load(os.path.join(tensors, "features.pt")).reshape(n_code * n_direction, -1)[:a.compare]
        new = encode(model, walk_paths[:a.compare], preprocess, a.batch_size, a.workers, device, "compare")
        diff = (new - ref).abs()
        rel = diff.max() / ref.abs().max()
        cos = torch.nn.functional.cosine_similarity(new, ref).min()
        print(f"max abs diff {diff.max():.3e}, mean abs diff {diff.mean():.3e}, max diff / max |feature| {rel:.3e}, "
              f"min cosine {cos:.8f}")
        return

    t0 = time.time()
    original = encode(model, code_paths, preprocess, a.batch_size, a.workers, device, "originals")
    walk = encode(model, walk_paths, preprocess, a.batch_size, a.workers, device, "walked")
    os.makedirs(tensors, exist_ok=True)
    # Written to temporary names first, so an interrupted run never leaves a complete-looking pair behind.
    torch.save(original, os.path.join(tensors, "features_original.pt.tmp"))
    torch.save(walk.reshape(n_code, n_direction, -1), os.path.join(tensors, "features.pt.tmp"))
    os.replace(os.path.join(tensors, "features_original.pt.tmp"), os.path.join(tensors, "features_original.pt"))
    os.replace(os.path.join(tensors, "features.pt.tmp"), os.path.join(tensors, "features.pt"))
    print(f"  wrote tensors/features.pt {tuple(walk.reshape(n_code, n_direction, -1).shape)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
