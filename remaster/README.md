# ConceptLens on Observable (no server)

The Django server's work now runs in the browser. The notebook reads its data from files attached to the notebook itself, so there is no server, no S3 and no proxy.

On the `remaster` branch of swj0418/concept-lens, this project is the `remaster/` folder, and the generation scripts use the repository's own `data_generation/` and `server/`. In a standalone copy, `setup.ps1` clones the repository into `generation/concept-lens`.

`generation/generate.py` is at version 2: sound versions of SeFA, GANSpace, vector arithmetic and SVM, a new CLIP-guided method, and edits restricted to the chosen layers by default (`--application global` edits every layer, as the paper did). Its docstring lists the methods and the fixes. Only small tests of version 2 have run so far; see `generation/validate_methods.py`.

| File | What it is |
|---|---|
| `conceptlens.notebook.js` | The notebook: 18 cells to paste into Observable |
| `conceptlens-preview.html` | The same notebook as one offline page. Double-click it and choose a bundle folder; it makes no network requests. |
| `export_for_observable.py` | Turns experiment folders into attachable files (a *bundle*) |
| `conceptlens-core.js` | The server's math in JavaScript (also inside the notebook as the `core` cell) |
| `parity/record_fixtures.py`, `parity/parity.mjs` | Record the original server's responses on your data, then check the JavaScript against them |
| `parity/make_synthetic.py` | Fake experiments in the original folder format, for testing without real data |
| `generation/` | Scripts that generate the experiments on this PC and export the bundle in one run |
| `generation/extract_features.py` | Batched CLIP features, matching the repo's `process_experiment.py` to float rounding |
| `generation/validate_methods.py` | Checks the direction methods against independent references on a finished (test) run |
| `sample-bundle/` | A tiny synthetic bundle (24 codes, 2 methods, 460 KB) to try the preview right away |

## 1. Generate the data (this PC, NVIDIA GPU with 6 GB or more)

Double-click `generation\start.cmd`. It runs the two steps below and saves everything it shows to `generation\log.txt`. It also keeps the PC awake until its window closes. To run the steps yourself, in the `generation` folder:

```
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1 --parity
```

- **`setup.ps1`, once:** creates `.venv` with PyTorch for your CUDA driver, then downloads the concept-lens repo, NVIDIA's StyleGAN2 FFHQ-256 checkpoint and the CLIP weights (about 8 GB). It finishes by rendering a test image, `check.jpg`.
- **`run.ps1`:** generates four experiments on FFHQ 256 (early layers): SeFA, GANSpace, vector arithmetic and SVM, 48 directions each, all editing the same 200 codes. It then extracts CLIP features with the repo's own `process_experiment.py` and exports `bundle\s2_ffhq256-early`. With `--parity` it also records the original server's answers on the new data and checks the port against them (needs Node.js or Deno).
- **Resuming:** an interrupted run picks up where it stopped. Most of the time goes to feature extraction, which handles about 39,000 images one at a time.
- **Options:** see `.\run.ps1 --help`. For example, `--layer middle`, `--n_directions 32`, or `--domain s2_wild512 --concepts my-animal-concepts.txt` (run `setup.ps1 -Domain s2_wild512` first to get that checkpoint).
- **SVM concepts:** `concepts-ffhq.txt` holds the 48 text concepts behind the SVM directions. Edit it to change them.

`generation/generate.py` uses the repo's direction-finding code, with these fixes:

| Problem in `data_generation/` | Effect | Fix |
|---|---|---|
| Original images come from `generate_images(w_codes)`, but `forward` expects z | `codes/{i}.jpg` is a different face from the one the walks start from, so the Original column and every edit magnitude are wrong | Originals are rendered from the same z as the walks |
| `get_style_weights_and_biases` exists only for StyleGAN3 | GANSpace crashes on StyleGAN2 | Added for StyleGAN2 |
| SVM feeds images in [-1, 1] to `ToPILImage` | Negative pixel values wrap around, so CLIP labels corrupted images | Images are converted to 8-bit first |
| The SVM concept list is the animal one for every domain | Face directions come from concepts like "tiger" | Concepts come from a text file |
| Every walked image is kept in memory | About 7.5 GB of RAM per method at this size | Images are written as they're made |
| Custom CUDA ops need MSVC and the CUDA toolkit | Crash on a typical Windows PC | The reference PyTorch ops are used (`--cuda-ops` to compile) |
| SeFA and GANSpace aren't seeded | Directions change from run to run | Fully seeded |
| GANSpace always makes 96 directions; vector arithmetic can repeat a centroid pair | Counts don't follow `--n_directions`; duplicate directions | Exactly `--n_directions` distinct directions per method |

GANSpace works only on layers whose style vectors have 512 entries, which covers early and middle layers on FFHQ 256.

## 2. Export a bundle by hand (other data)

`run.ps1` does this for you. For experiments made some other way, run this in the server's Python environment (torch, scipy, Pillow):

```
python export_for_observable.py --server <repo>/server --data <folder with the experiments> \
    --experiments s2_ffhq256-sefa-early-global,s2_ffhq256-ganspace-early-global,... \
    --out bundle/s2_ffhq256
```

The output contains:
- the walked CLIP features at 16-bit, split into parts under 45 MB (Observable's limit is 50 MB per file)
- exact edit magnitudes
- exact initial distances for every combination of methods
- one image sprite per code
- `attachments.js`

Options:
- **Order:** the order of `--experiments` sets the direction ids, just as `experiment_names` did.
- **Precision:** `--dtype float32` doubles the feature files but makes reclustering bit-identical.
- **Images:** `--images 0` skips them; `--images 96` makes smaller thumbnails.

## 3. Check parity by hand

```
python parity/record_fixtures.py --server <repo>/server --data <folder> --experiments <same list> --out fixtures.json
node parity/parity.mjs bundle/s2_ffhq256 fixtures.json
```

With Deno instead of Node: `deno run --allow-read parity/parity.mjs bundle/s2_ffhq256 fixtures.json`.

## 4. Put it on Observable

1. Paste the cells from `conceptlens.notebook.js`, one per cell.
2. Drag every file in the bundle folder onto the notebook's Files pane.
3. Replace the `files` cell with the contents of `attachments.js`.

Uploads count toward Observable's quota of 1 GB per 28 days.

## Size at the paper's scale

Features take codes × directions × 1,024 dimensions × 2 bytes:

| Data | Features download |
|---|---|
| 200 codes × 192 directions (one method) | about 79 MB |
| Two methods (384 directions) | about 157 MB |
| The 1,000-code set of Fig. 9 | about 790 MB — too large for attachments and a browser tab; use a 200-code subset |

## Parity with the original server

The test replays recorded calls to the original Django views on full-size synthetic data (200 codes × 4 methods × 40 directions × 1,024 dimensions).

| Bundle | Result |
|---|---|
| float32 | All 42 calls identical: trees, leaf order, bfloat16 magnitudes, weave scores. Inconsistency within 2 × 10⁻⁷ relative. |
| float16 (as shipped) | Tree shapes and leaf order identical in all 42. In 1 of 12 reclusters, two merges at nearly equal heights are numbered in the opposite order. Inconsistency within 1.1 × 10⁻⁴ relative. |
| Two-method subset | All 30 calls identical. |

**Speed:** selections take 3–10 ms and reclusters about 0.6 s.

## Where the code and the paper differ

These are settings to explore rather than errors. The port keeps the code's defaults, so it reproduces what the tool actually computed.

| | Paper | Code (and this port) |
|---|---|---|
| Linkage | Ward (Sec. IV-B) | `complete` by default; Ward is an option |
| Code distance | cosine, Eq. (3) | Euclidean on raw, unnormalized CLIP features |
| Edit consistency | mean of per-dimension **variance**, Eq. (5) | mean of per-dimension **standard deviation** |
| Displacement | count of source nodes, Eq. (6) | sum of index gaps between disjoint pairs ÷ cluster size |

## Quirks in the original server

- **Clearing a selection crashes the server.** Empty selections fall back to `range(1024)`, which raises `IndexError`. The port uses everything.
- **Reclustering with no codes** gives NaN magnitudes. The port uses all codes.
- **Displacement mixes up methods.** Directions are keyed by `flatIdx`, their index within their own method, so directions from different methods with the same index count as one. The port reproduces this by default; `legacyWeave: false` fixes it.
- **The recluster response's `weaveScores` is always empty.** The scores on the tree nodes are correct.
- **Node magnitudes are rounded to bfloat16.** The port reproduces this.
