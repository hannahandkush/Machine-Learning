# Machine-Learning

Final project for the **Practical Machine Learning** module, Master's in Green Data
Science 2025–2026, Instituto Superior de Agronomia (ULisboa). Instructor: Manuel
Campagnolo. Team: Hannah Nathanson and Danilo III Ortañez Gonzales.

## Project overview

**Problem.** ICNF, the Portuguese forestry authority, currently maps burned-area
perimeters (`ardida`) by manual satellite image interpretation. The lag between a
fire occurring and a published perimeter delays aerial firefighting resource
allocation, emergency logistics, and legal/insurance triggers, and every
downstream use of a burn perimeter — post-fire erosion and flood-risk modelling,
reforestation prioritisation, LULUCF carbon accounting — inherits that delay.
This project tests whether a deep-learning change-detection model can reproduce
ICNF's delineation directly from Sentinel-2 imagery.

**Data.** Sentinel-2 tile **T29TPG** (northeastern Portugal, Trás-os-Montes —
districts of Bragança and Vila Real), 79 scenes from January–December 2025, 10
spectral bands (B2–B12) at 10 m resolution, stored as one ~32 GB HDF5 cube. Ground
truth is ICNF's 2025 `ardida` shapefile (2,084 fire perimeters nationally, 182
within the tile), rasterised to the same 10 m grid with a 1-pixel interior erosion
to remove boundary-ambiguous mixed pixels (`data/processed/icnf_burned_labels_t29tpg_2025.tif`).

**Models compared.** Both are pre-trained, used for inference only (no
fine-tuning on T29TPG data), provided by Prof. Campagnolo's MLC research group:

| Model | Architecture | Classes | Burned class |
|---|---|---|---|
| **Swin-YNet** | dual-temporal Swin Transformer encoder, U-Net-style decoder (fine-tuned MSR-BACD) | Background / Cuts / Fires | `Fires` |
| **EfficientNet-B2** | U-Net with an EfficientNet-B2 encoder (~10M params) | Background / Burned | `Burned` |

Both take stacked before/after 10×256×256 uint16 chips and return a per-pixel
class map; predictions are reduced to a binary burned mask for comparison.

**Inference pipeline** (`inference/`, see `inference/README.md` for the full
architecture): classify each of the 79 scenes by cloud cover and swath coverage,
pick a single global before/after pair (baseline **2025-07-07**, post-fire
**2025-10-15**, both cloud ≤1%), reconstruct each of the tile's 959 chips from the
cube, batch them through the model adapter, and mosaic the per-chip predictions
into three georeferenced GeoTIFFs (3-class map, binary burned mask, observed-pixel
mask). A sliding-window overlap mode (`inference/run_overlap.py`) re-runs the tile
at 0/25/50/75% window overlap and lets a pixel's class be decided by majority (or
stricter) vote across overlapping windows, trading runtime for fewer false
alarms.

**Evaluation.** Full reasoning in `documents/evaluation_protocol.md`. Overall
accuracy and ROC/AUC are excluded as misleading under severe class imbalance
(burned ≈1.7% of the tile) and a hard-label output respectively; **MCC** (Matthews
Correlation Coefficient) is the primary metric because it uses all four confusion
matrix cells and is not inflated by the imbalance, with precision/recall/F1 on the
burned class as supporting metrics.

**Headline results.**

| Setting | Precision | Recall | F1 | MCC |
|---|---|---|---|---|
| EfficientNet-B2, no overlap | 0.55 | 0.91 | 0.69 | 0.70 |
| Swin-YNet, no overlap | 0.36 | 0.91 | 0.51 | 0.55 |
| EfficientNet-B2, 50% overlap, majority vote | 0.65 | 0.91 | 0.76 | 0.76 |
| EfficientNet-B2, 75% overlap, 80% vote (best tested) | 0.72 | 0.89 | 0.80 | 0.80 |
| EfficientNet-B2, 50% overlap, 75% vote (practical alternative, ⅓ the runtime) | 0.65 | 0.91 | 0.76 | 0.76 |

EfficientNet-B2 outperforms Swin-YNet at every overlap and vote setting tested,
mainly on precision: Swin-YNet produces roughly 1.8× more false positives for
comparable recall. Sliding-window overlap with a majority vote acts as a
false-alarm filter (removes scattered seasonal/noise detections while keeping
real fires); a stricter vote pushes this further at the cost of run time
(`notebooks/overlap_experiment.ipynb`).

**False positive review** (`notebooks/false_positive_review.ipynb`): manually
cross-checked the largest false-positive clusters against Sentinel-2 true-colour
chips, OpenStreetMap, and Google Maps. Most are small noise; the two notable
exceptions are real, persistent stream channels misread as burned ground by both
models (a watercourse spectrally resembles bare/burned soil), not ICNF mapping
omissions — a model failure mode worth naming explicitly rather than a ground-truth
error.

### Repository layout

```
inference/        tile-wide inference pipeline (model-agnostic; models are adapters)
models/           model weights + packages — git-ignored, kept local (see below)
data/             ICNF shapefiles, boundary files, processed labels — HDF5 cube git-ignored
notebooks/        exploration, evaluation, overlap/voting experiments, false-positive review
outputs/          prediction rasters + figures — git-ignored, reproducible by re-running inference
utils/            shared config loading + Sentinel-2 Process API client (for the app)
documents/        project proposal, evaluation protocol, report draft, course PDF
app.py            Streamlit app for exploring predictions interactively (see below)
```

## Running the interactive app locally

`app.py` is a local Streamlit app for exploring the burned-area predictions: pick
a model, a before/after scene pair, the window overlap, and the voting
strictness, and see the result on an interactive map, plus a focal-zone inspector
for spot-checking individual true/false positives against Sentinel-2 imagery and
OpenStreetMap.

### Why you need the full project folder, not just a GitHub clone

The app reads prediction rasters from `outputs/predictions/`, and (to run new
configurations) model weights from `models/` and the Sentinel-2 cube from
`data/hdf5/`. All three are listed in `.gitignore` and are **not on GitHub** —
they're large (the HDF5 cube alone is ~32 GB) and, for the model weights,
not ours to redistribute. They do exist on the [project OneDrive](https://ulisboa-my.sharepoint.com/:f:/g/personal/hnathanson_office365_ulisboa_pt/IgCdVlyvCI7QQp0_-fKiW2SiAQl6qzbKoUDUVgMWYLcjLOg),
which is the folder this whole repository is normally worked in (it's
OneDrive-synced, with `.git` alongside the data). So:

- **To explore the bundled prediction runs** (several model/overlap/date
  combinations already included under `outputs/predictions/`): you need a local,
  fully-synced copy of the **OneDrive** `Final_Project` folder, not just a plain
  `git clone` of the GitHub repo. A `git clone` gets you the code; OneDrive sync
  gets you the data and outputs sitting alongside it. The cleanest setup is to
  clone the GitHub repo *into* the synced OneDrive folder (so `git` and the
  OneDrive sync share the same directory), which is how this project was
  developed.
- **To run new model/date/overlap combinations from the app** (the **Run**
  button that appears for a configuration not already in `outputs/predictions/`):
  you additionally need `models/` (weights) and `data/hdf5/T29TPG.h5` present
  locally — see `inference/README.md` and, if working from a machine without
  direct OneDrive access, `documents/onedrive_rclone_access_setup.md` for how the
  HDF5 cube was pulled down via `rclone`.

### CDSE credentials (for the focal-zone inspector's Sentinel-2 chips)

The inspector's before/after Sentinel-2 panels are fetched live from the
Copernicus Data Space Ecosystem (CDSE) Process API (`utils/sentinel_hub.py`).
This needs a CDSE OAuth `client_id`/`client_secret`, which — like the data above —
must **never be committed**: they belong in `config.local.yaml` (copy
`config.yaml` to `config.local.yaml` and add a `cdse:` block), a file that is
git-ignored for exactly this reason. A working `config.local.yaml` with real
credentials already exists in the OneDrive project folder; if you only have the
GitHub clone, you'll need to register your own CDSE client at
[dataspace.copernicus.eu](https://dataspace.copernicus.eu/) and add it yourself.
Every other part of the app works without this — the inspector just shows
"Needs CDSE credentials" for the Sentinel-2 panels instead.

### Requirements

- The `veg-s2s` conda environment (`environment.yml`), which includes
  `streamlit`, `folium`, and `streamlit-folium`:
  ```bash
  conda env update -f environment.yml      # or: conda env create -f environment.yml
  conda activate veg-s2s
  ```
- The prediction rasters in `outputs/predictions/` (see above — present if
  you're working inside the synced OneDrive folder).
- For *new* configurations only: `models/` and the HDF5 cube (see above).

### Run it

From the repository root, with the environment active:

```bash
streamlit run app.py
```

It opens in your browser at http://localhost:8501. To keep it reachable only from
your own machine (not the local network), bind it to localhost:

```bash
streamlit run app.py --server.address 127.0.0.1
```

### Using it

- **Map viewer** tab: model, before/after date, window overlap, and a
  voting-strictness slider in the sidebar. Moving the voting-strictness slider
  re-thresholds the existing run instantly (no model run). Picking a combination
  not already in `outputs/predictions/` shows a **Run** button that launches the
  model with a live progress bar. Toggle the ICNF ground truth and Portugal
  boundary as overlays; use the inset map to locate the view.
- **Processed outputs** tab: browse every run that exists locally, switch
  between the "Burned area" and "Error map (TP/FP/FN)" overlay modes, and use the
  **focal-zone inspector** at the bottom — click any point on the map for a ~3 km
  chip shown four ways (Sentinel-2 before/after, OpenStreetMap, and that run's
  error map), the same manual check used in
  `notebooks/false_positive_review.ipynb`.

## How the work was divided


**Phase 1: setup**

| Hannah | Danilo |
|---|---|
| Reviewed agricultural/land-cover data and sourced INE land-use data (discarded) | Derived the T29TPG tile boundary from HDF5 metadata and pushed it to GitHub (blocking task — done first) |
| Redrafted the project proposal with the new evaluation plan and references | Obtained both model weights (Zenodo pre-fine-tuned + course SharePoint fine-tuned) |
| Set up remote HDF5 access via `rclone` + Colab | Set up inference environment with Identify framework/dependency versions and environment.yml / document|

**Phase 2: data prep & deployment**

| Hannah | Danilo |
|---|---|
|Filtered/translated the ICNF shapefile, rasterised ground-truth labels to the chip grid (`icnf_burned_labels_t29tpg_2025.tif`) | Built the initial tile-wide inference pipeline and the Swin-YNet adapter (`inference/`)|
| Adapted the pipeline to add an EfficientNet-B2 adapter, ran both models over the full tile, and produced the initial comparative analysis |  Ran Swin-YNet over T29TPG and exported prediction outputs |

**Phase 3: evaluation & comparison**

| Hannah | Danilo |
|---|---|
| Validated EfficientNet-B2 vs Swin-YNet against the agreed metrics; summarised limitations | Ran the overlap (0/25/50/75%) and voting-strictness experiments across both models, full metrics tables |
| Re-ran validation at the optimal setting (50% overlap, 75% voting strictness) | Produced the evalutation_protocol.md |
| Wrote the false-positive review notebook and investigated flagged clusters (river/stream confluences, noise) | Designed and built the interactive Streamlit app |
| Added the TP/FP/FN error-map layer and the focal-zone inspector (click-to-compare Sentinel-2 before/after, OpenStreetMap, and error map) to the app | Added a new tab to (`app.py`) with previous runs |
| 

**Phase 4: Write-up**

| Hannah | Danillo |
|---|---|
| Analysis: EfficientNet-B2 vs Swin-ynet model performance comparison section | Introduction, data, data management and methods |
Deployment: EfficientNet-B2 optimisation decision evaluation section | Report finalisation, contributions and appendix


### Notebooks

| Notebook | Author |
|---|---|
| `hdf5_data_exploration.ipynb` | Danilo |
| `burned_area_date_comparison.ipynb` | Danilo |
| `fire_incidence_2025.ipynb` | Danilo |
| `overlap_experiment.ipynb` | Danilo |
| `Model_comparison.ipynb` | Hannah |
| `min_fire_size.ipynb` | Hannah |
| `false_positive_review.ipynb` | Hannah |
| `efficientnet_evaluation.ipynb` | Joint

## Report and submission

The full written report lives in `documents/Report_Final.docx` (sections:
introduction, data, methods, results, analysis, deployment, contributions,
references — per the course's [project guidelines](documents/project_pml_2025_2026_.pdf)).
Supporting write-ups: `documents/project_proposal.md`,
`documents/evaluation_protocol.md`. Project repo: <https://github.com/hannahandkush/Machine-Learning>.
