# Inference pipeline

Tile-wide burned-area inference for Sentinel-2 tile **T29TPG**. The pipeline
reads before/after image chips from the HDF5 cube, runs them through a
change-detection model, and reassembles the per-chip predictions into
georeferenced burned-area rasters.

It is **inference only**, so no training happens here. The model is treated as a
swappable component behind an adapter (see [Adding a model](#adding-a-model)).

> *Note*: An **adapter** is a small piece of code that converts a model into a standard format so you can swap models without changing other code.
---

## What it does, in one line

For one before/after scene pair, run every 256×256 chip of the tile through the
model and stitch the results into three GeoTIFFs: a 3-class map, a binary burned
mask, and an observed-coverage mask, all aligned to the ICNF label grid.

---

## Quick start

Run from the repository root:

```bash
# full tile, auto-selected scenes (~4 min on CPU)
python -m inference.run --device cpu

# quick smoke run (a handful of chips)
python -m inference.run --device cpu --max-chips 8

# pick the scene pair explicitly
python -m inference.run --device cpu --before-date 2025-07-07 --after-date 2025-10-15
```

Outputs land in `outputs/predictions/` (git-ignored).

### Command Line Interface (CLI) flags

| Flag | Purpose |
|---|---|
| `--device` | Force `cuda` / `mps` / `cpu`. Default: auto (cuda → mps → cpu). |
| `--batch-size N` | Chips per model batch (default from config). |
| `--max-chips N` | Process only the first N chips (smoke runs). |
| `--before-date` / `--after-date` | Override scene auto-selection (YYYY-MM-DD). Both or neither. |
| `--weights PATH` | Override the model checkpoint (must match the configured model). |
| `--resume` | Skip chips already recorded in the manifest. |
| `--out DIR` | Output directory (default: `inference.predictions_dir` from config). |

---

## Architecture

The pipeline is split so that everything except the model itself is
model-agnostic. Modules fall into four types:

| Type | Files | Role |
|---|---|---|
| **Execution & Configuration** | `run.py`, `utils/config.py` | drive the run, load settings |
| **Input Preparation** | `scene_select.py`, `chips.py` | choose scenes, build chips from the cube |
| **Model / Inference** | `adapters/__init__.py`, `adapters/swin_ynet.py` (+ external `bacdm_predict`) | load the model, run prediction |
| **Output / Mosaicking** | `mosaic.py` | accumulate + write the GeoTIFFs |

```
inference/
  run.py            scene_select.py       
  chips.py          mosaic.py
  adapters/
    __init__.py     ← registry (model.kind → adapter) + pick_device
    swin_ynet.py    ← the Swin-YNet model wrapper
```

`run.py` is the spine: it loads config, selects scenes, allocates the output
tile, loads the model adapter, then loops over chip batches and writes the
result.

---

## Data flow (Will make a flowchart)

```
HDF5 cube + weights + config
        │
  1. select before/after scenes        scene_select.py
        │
  2. reconstruct chip pair             chips.py        ┐
  3. preprocess + Swin-YNet + decide   adapters/*      │ per batch of 8 chips,
  4. mask + place on tile              mosaic.py       ┘ repeated over 959 chips
        │
  5. write 3 GeoTIFFs                   mosaic.py
```

### 1. Scene selection (`scene_select.py`)
Each of the 79 acquisitions is classified from per-scene metadata
(`cloud_cover_pt`, `count_orbit_pixels_pt`, `pixel_count_pt`) into
`full_clear` / `partial_clear` / `edge_pass` / `clouded`. By default only
**`full_clear`** scenes (full swath coverage AND ≤10% cloud) are eligible. The
before scene is the cleanest baseline in `before_window`; the after scene is the
latest `full_clear` in `after_window`. Defaults select **2025-07-07 → 2025-10-15**.

### 2. Chip reconstruction (`chips.py`)
The cube stores `values` as `(79, 10, 62_849_024)` = (timestamps, bands,
pixels), where each 65 536-pixel block on the flat pixel axis is one 256×256
chip. `read_chip()` reads a chip's `(10, 65536)` slab for a timestamp and
**scatters** each pixel into a dense `(256, 256, 10)` array using its UTM
coordinate (`xs_new`/`ys_new`) and the chip's grid position
(`chip_x_bin`/`chip_y_bin`). Missing/padding pixels stay NODATA (65535). It
returns the chip plus two masks: `footprint` (belongs to the tile) and
`observed` (has real data in this scene). This runs twice per chip: once for
the before timestamp, once for the after.

### 3. Model inference (`adapters/`)
The adapter is the only model-specific code. `adapters/swin_ynet.py` wraps the
local `bacdm_predict` package: `load()` loads the checkpoint once;
`predict()` hands a `(B, 256, 256, 10)` uint16 batch to the model, which
preprocesses it (percentile stretch, band select, normalize), runs the
Swin-YNet forward pass (two encoder branches compare before vs after), and
returns `(B, 256, 256)` labels. Classes: **0 = Background, 1 = Cuts,
2 = Fires (burned)**. The adapter declares `BURNED_CLASS = 2`.

### 4 & 5. Mask + mosaic (`mosaic.py`)
For each chip, `build_blocks()` keeps a prediction only where the pixel was
**observed in both** the before and after scenes; elsewhere it writes NODATA.
The blocks are placed into three in-memory tile arrays and, at the end, written
once as GeoTIFFs. (See [Why in-memory](#why-in-memory-accumulation).)

---

## Outputs

Three single-band uint8 GeoTIFFs (EPSG:32629, 10 m, nodata 255), stem
`T29TPG_<model>_<before>_<after>`:

| File | Values | Meaning |
|---|---|---|
| `…_pred.tif` | 0 / 1 / 2 / 255 | full 3-class map (Background / Cuts / Fires) |
| `…_burned.tif` | 0 / 1 / 255 | binary burned mask = `(pred == BURNED_CLASS)` |
| `…_observed.tif` | 0 / 1 / 255 | 1 = observed in both scenes (trusted), 0 = in-footprint but unobserved, 255 = outside footprint |
| `…_manifest.json` | n/a | run parameters + completed chip ids (for `--resume`) |

The grid is identical to `data/processed/icnf_burned_labels_t29tpg_2025.tif`
(10980×6304), so predictions overlay the ICNF ground truth pixel-for-pixel, and
`run.py` asserts this at startup. `_observed.tif` lets evaluation exclude
unobserved pixels rather than scoring them as correct rejections.

---

## Configuration

Set in `config.yaml` (`model:` and `inference:` blocks), loaded via
`utils/config.py`:

```yaml
model:
  kind: swin_ynet                 # selects the adapter
  weights_path: ./models/...pth   # checkpoint (git-ignored, local only)
  package_dir:  ./models/updated_model/bacdm_predict
inference:
  predictions_dir: ./outputs/predictions
  batch_size: 8
  usable_categories: ["full_clear"]          # scene quality filter
  before_window: ["2025-07-01", "2025-07-22"]
  after_window:  ["2025-10-01", "2025-12-31"]
```

The "which class is burned" index is **not** here. It is a property of the
model and lives in the adapter (`BURNED_CLASS`).

---

## Adding a model

The pipeline supports any before/after change-detection model that takes a
`(B, 256, 256, 10)` chip batch. To add one (e.g. EfficientNet-B2):

1. Write `inference/adapters/<name>.py` exposing the adapter interface:
   `load(weights_path, package_dir, device) → (handle, model)`,
   `predict(handle, before, after, model, device) → (B,H,W) uint8`,
   plus `NAME` and `BURNED_CLASS`.
2. Register it in `inference/adapters/__init__.py` (`ADAPTERS` dict).
3. Set `model.kind` (and the weights/package paths) in config.

Nothing in the shared pipeline changes; chip reconstruction, scene selection,
masking, mosaicking and georeferencing are reused as is.

---

## Design notes

### Why in-memory accumulation
An earlier version streamed each chip straight to disk as a windowed write into
a compressed, tiled GeoTIFF. GDAL cannot rewrite an already-flushed compressed
block, so out-of-spatial-order chip writes were silently dropped, and the tile
collapsed to a single column. The fix: accumulate the whole tile in memory
(3 × ~69 MB uint8 arrays) and write each GeoTIFF in one pass. Checkpoints are
saved every 20 batches so `--resume` has state.

### Scene quality filter (currently full_clear only)
For now the pipeline uses only full-coverage (`full_clear`) scenes, so clouds
and orbit-edge gaps don't corrupt the chips. This is just the setting used for
initial testing, not a fixed rule. It is controlled by `usable_categories` in
config and can be relaxed later, for example to allow `partial_clear` scenes or
to pair each fire with its own nearby clear scenes. One limit to keep in mind
with the current setting: the last `full_clear` acquisition is 2025-10-15, so a
single before/after pair cannot capture fires that end after that date.

### Local-only model
The `models/` folder (model package + weights) is git-ignored and kept local.
The adapter imports it by adding its directory to `sys.path`, so the tracked
`inference/` code needs that folder present on the machine to run.

---

## Verification

```bash
# 1. smoke run: confirms the model loads and the path works
python -m inference.run --device cpu --max-chips 8

# 2. check the burned raster value distribution
python -c "
import rasterio, numpy as np
p='outputs/predictions/T29TPG_swin_ynet_20250707_20251015_burned.tif'
with rasterio.open(p) as r:
    v,c=np.unique(r.read(1),return_counts=True); print(dict(zip(v.tolist(),c.tolist())))
"
```

A healthy run prints `[grid] ... OK`, the chosen scenes, and
`[done] 959/959 chips`. For visual validation, overlay `…_burned.tif` on
`data/shapefiles/ground_truth_ICNF/ardida_2025.shp` in QGIS. Burned pixels
should sit on the known large fires.