# ConceptLens remaster: notes for Claude

The ConceptLens paper tool (repo github.com/swj0418/concept-lens, IEEE TVCG 2025), remastered:
- a port to an Observable notebook that runs with no server;
- a data-generation pipeline with sound direction-finding methods.

README.md covers usage; this file is the working context.

## Where things are
- **In git:** branch `remaster` of swj0418/concept-lens. This folder is `remaster/`. The repository root holds the original tool: `data_generation/`, `server/` (Django), `client/` (React), `process_experiment.py`. The pipeline imports those from the enclosing repo; `generate.default_repo()` and setup.ps1 detect it.
- **Standalone copy** (e.g. the original C:\Work\Concept-Lens-ObservableHQ): setup clones the repo into `generation/concept-lens` instead.
- **Not in git** (large, regenerable):
  - `bundle/`, `generation/.venv`, `generation/experiments`;
  - checkpoints and old experiments under `<repo>/resources/`.

## What's here
- `conceptlens.notebook.js`: 18 notebook cells, with `conceptlens-core.js` inlined as the `core` cell.
- `conceptlens-core.js`: the Django server's math in JavaScript. Linkage and optimal leaf ordering match scipy exactly, and node magnitudes are rounded to bfloat16 as the server does.
- `conceptlens-preview.html`: the notebook as one offline page. Pick a bundle folder in it; it makes no network requests. To load a bundle served over HTTP (for automated checks): `window.__main.redefine("files", [], () => new Proxy({}, {has: () => true, get: (t, n) => typeof n === "string" ? {name: n, json: ..., arrayBuffer: ..., url: ...} : undefined}))` with fetches from the server.
- `export_for_observable.py`: turns experiment folders into a bundle (16-bit features split into parts under 45 MB, exact magnitudes and distances, sprites, `attachments.js`).
- `parity/`:
  - `record_fixtures.py` records the original Django views' responses.
  - `parity.mjs` checks the JS port against them. Run it with Node or `deno run --allow-read`.
- `generation/`:
  - `generate.py` (v2): directions, then renders, features, export and optional parity. Details below.
  - `extract_features.py`: batched CLIP ViT-H-14 dfn5b features in the repo's format. They match `process_experiment.py` to 1.4e-5 max abs.
  - `validate_methods.py`: checks the methods against independent references. See its docstring for the test run it expects.
  - `concepts-ffhq.txt` (48 concepts) and `concepts-test.txt` (8), with `# template:` and `# neutral:` prompt lines.
  - Windows-only scripts:
    - `setup.ps1`: venv with torch cu13x, checkpoint and CLIP downloads.
    - `run.ps1`: runs `generate.py`.
    - `start.cmd` / `start.ps1`: setup, then run, with a log.
- `sample-bundle/`: a tiny synthetic bundle for trying the preview.

## generate.py v2 methods
All directions are unit vectors in W. An edit is w + 5·d.

| Method | What v2 does | What the repo did |
|---|---|---|
| `sefa` | Official SeFA: eigenvectors of AᵀA, A = the chosen layers' style weights with unit-norm rows | SVDs of random 511-row subsets, KMeans, arbitrary signs, skipped the first eigenvector |
| `ganspace` | PCA of the chosen layers' style vectors, via A·cov(w)·Aᵀ over 100k samples, mapped back to W by least squares | Top-6 PCs of 8 random 32-1024-sample subsets |
| `va` | Differences of distinct unordered pairs of 20 k-means centroids | |
| `svm` | InterFaceGAN style. CLIP ViT-B-32 with QuickGELU (the repo got GELU), 10k samples, top vs bottom 20% of (concept − neutral) scores, LinearSVC in W | |
| `clip` | New. Per concept, a direction optimised with StyleGAN-NADA's directional loss: ViT-B-32 and ViT-B-16, two random crops, Adam, 150 steps, batch 8. The edit is applied exactly as rendered. | |

Pipeline changes:
- `--application local` (default) edits only the `--layer` layers. `global` edits all layers, as the paper and the repo's scripts did.
- Originals are rendered once per domain (`<experiments>/_codes-*`) and copied to every method, so every method has identical original images and features.
- Direction finding uses seed + 1, so no edited code is in any method's samples.
- `setting.json` gets a `diagnostics` block: the step size in SD of W and the maximum |cos| between directions.
- Experiments are named `<domain>-<method>-<layer>-<application>`. Bundles go to `bundle/<domain>-<layer>-<application>`.

## Status (2026-09-29)
- **The local PC can't do long GPU runs.** The Windows machine (RTX 3090) blue-screened twice under this workload, at 21:10 on 9/28 and 13:42 on 9/29. Windows has logged recurring 0x3B, 0x141 and 0x124 crashes since March. Work moves to a cloud box. Long GPU jobs must be resumable (they are: finished steps are skipped).
- **v1 results** (repo methods, global edits, the first `generate.py`) exist only on the local PC: `bundle/s2_ffhq256-early` (4 methods × 48 directions, 200 codes).
- **Parity on that real bundle, run with Deno:** float16 38/42 identical. The 4 misses are reclusters with near-tied merges, and one changes a tree shape. A float32 export gives 42/42, so the port is exact.
- **v1 bundle quirk:** the four methods had slightly different original images (GPU rendering isn't bit-exact). v2 fixes this.
- **v2 small test** (FFHQ-256, 16 codes, 8 directions, local edits) before the crash:

  | Method | Step (SD of W) | Max \|cos\| between directions |
  |---|---|---|
  | sefa | 2.98 | 0.0 |
  | ganspace | 2.77 | 0.06 |
  | va | 2.83 | 0.73 (pairs that share a centroid) |
  | svm | 19.3 (v1: 28.6) | 0.24 |

  - SeFA's local early-layer edits looked right: a consistent pose, face-shape, age or zoom change across faces, with colours and identity kept. The global v1 edits had replaced whole faces.
  - The PC crashed during `clip` optimisation. `clip` has never finished a run.
  - `validate_methods.py` hasn't been run yet.
- **Open question on svm:** even with 10k samples, the SVM normal lies along low-variance W axes, so a 5-unit step is about 19 SD. Check with `validate_methods.py` (held-out ViT-H-14) whether its edits show the concept. If they don't, a smaller C (the normal tends to the class-mean difference as C → 0) or a mean-difference direction may be the sound choice.
- **Not started:**
  - StyleSwin (the transformer StyleGAN the user wants), diffusion models (h-space directions; the user wants better image quality), and other StyleGAN domains (AFHQ wild/cat, CelebA-HQ, MetFaces; checkpoint names in setup.ps1).
  - Concept lists for those domains.
  - A job queue for unattended runs.
  - An earlier research workflow and a method-review workflow were cut off by the crashes; no results survived.

## Next steps (cloud box)
1. Linux setup. setup.ps1, run.ps1 and start.cmd are Windows-only. Write a setup.sh:
   - a venv with CUDA torch and `generation/requirements.txt`;
   - the StyleGAN2 checkpoint into `<repo>/resources/checkpoints` (URLs in setup.ps1);
   - the three CLIP models from setup.ps1 step 7.
2. Run the v2 test (commands in the `validate_methods.py` docstring), then `validate_methods.py`. Fix what it shows, especially svm and clip.
3. Full FFHQ-256 run (200 codes × 48 directions × 5 methods), then the other domains, StyleSwin and diffusion.
4. Load the bundles in the preview. Upload to Observable: paste the cells, attach the bundle files, replace the `files` cell with `attachments.js`.

## Decisions
- "Exact" parity with the original server, at 16-bit feature precision. The notebook must run with no server.
- The paper and the code differ on Ward vs complete linkage, cosine vs Euclidean code distance, and variance vs std. The user treats these as options to explore; the defaults follow the code.
- The user asked for sound methods: fix the repo's direction finding rather than reproduce it. Global edits stay available (`--application global`) to match the paper.
- Browser budget: about 200 codes × 400 directions. The Fig. 9 set (1000 × 384) is too large.
