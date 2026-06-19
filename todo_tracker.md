# Project To-Do Tracker — due 15 July

## Reference links
- Repo: https://github.com/hannahandkush/Machine-Learning
- Pre-fine-tuned model (Zenodo, trained on Asian fire data): https://zenodo.org/records/15336666
- Fine-tuned model (Manuel's version, burned + cuttings): course SharePoint folder (`danilo_hannah`)
- Project OneDrive (data exchange): SharePoint link shared above

## Data on hand
1. HDF5 cube — Sentinel-2, tile T29TPG
2. ICNF shapefile — ground truth burned-area polygons (`ardida_2025`)

## Working objectives
- **Main goal**: deploy MLC's vegetation-change-detection model on Portuguese data
- **Side quest**: extend it to distinguish agricultural change from fire-driven change (using NUTS II / INE land-use data: https://www.ine.pt/scripts/db_ra_2019.html)
- **Validation plan**: (1) run the pre-fine-tuned model (Asia-trained) over Portugal and check accuracy against ICNF ground truth; (2) run the fine-tuned model and compare the two

---

## Side quest — natural vs. agricultural fire classification

| Task | Owner | Status | Notes |
|---|---|---|---|
| Check Hannah's data on agricultural changes | Hannah | ☐ | Confirm what dataset this refers to and where it lives |
| Source an agricultural / forest land-cover map | Hannah | ☐ | NUTS II / INE land-use data is one candidate source — moved to Hannah, doesn't touch the HDF5 cube |
| Upload tile boundary map (T29TPG) to GitHub | Third | ☐ | **Blocks** the ICNF filtering task below — flag as critical-path |
| Filter ICNF shapefile to study area, export attribute table to CSV, translate (PT→EN presumably), upload to GitHub | Hannah | ☐ | Blocked on the tile boundary file above; you said you'll provide/derive the boundary — once it lands, this is a clean GIS task (clip → `to_csv` → translate columns/values → commit) |

## Data processing

| Task | Owner | Status | Notes |
|---|---|---|---|
| Upload HDF5 to Hannah's OneDrive | Third | ☐ | **Worth re-checking this item itself** — if the file is too large for Hannah to open/store, moving the raw cube to her OneDrive may not actually be useful. Consider replacing with "Third shares derived/processed outputs (clipped chips, prediction rasters) with Hannah" instead — much smaller, and avoids redundant copies of a multi-GB file |
| Filter data | Third | ☐ | Reassigned to Third — clipping/masking operates directly on the HDF5 cube, which only Third can currently open. Vague as written — suggested concrete sub-steps below |

**Proposed breakdown for "Filter data"** (based on what `data_exploration.ipynb` already surfaces):
- [ ] Clip HDF5 tile extent to the study-area boundary
- [ ] Mask cloud/nodata pixels using the per-timestamp completeness categories (`full_clear` / `partial_clear` / `edge_pass` / `clouded`) already computed in the notebook
- [ ] Align ICNF ground-truth polygons to the pixel grid (rasterize to the 10 m / 256×256 chip layout)
- [ ] Define train / validation / test split (spatial split recommended over random, to avoid leakage between adjacent chips)

## Model deployment *(section was empty — drafted from the stated objectives)*

| Task | Owner | Status | Notes |
|---|---|---|---|
| Obtain both model weights (Zenodo pre-fine-tuned + Manuel's fine-tuned version) | Third | ☐ | pairs naturally with the file-handling/upload tasks above |
| Set up inference environment (framework/deps matching how each model was trained) | Third | ☐ | |
| Build an inference pipeline: load HDF5 tile → preprocess (NDVI/indices, chip extraction) → run model → output change/burn maps | Third | ☐ | builds directly on `data_exploration.ipynb`; reassigned to Third since it requires loading the cube directly |
| Run the pre-fine-tuned model over T29TPG | Third | ☐ | feeds objective 1 — Third exports prediction rasters/vectors (small) for Hannah to evaluate |
| Run the fine-tuned model over T29TPG | Third | ☐ | feeds objective 2 — same export-for-evaluation handoff |

## Model evaluation *(section was empty — drafted from the stated objectives)*

| Task | Owner | Status | Notes |
|---|---|---|---|
| Define evaluation metrics against ICNF ground truth (e.g. IoU, F1, precision/recall, confusion matrix) | Hannah | ☐ | |
| Validate pre-fine-tuned model output vs. ground truth | Hannah | ☐ | objective 1 — works from Third's exported prediction outputs + the ICNF ground truth, no cube access needed |
| Validate fine-tuned model output vs. ground truth | Hannah | ☐ | objective 2 — same handoff pattern |
| Compare pre-fine-tuned vs. fine-tuned performance | Hannah | ☐ | the actual "fine-tuning helped/didn't" finding; natural fit alongside the two validation tasks above |
| (Side quest) Evaluate agricultural vs. natural-fire classification against NUTS II / INE reference | Hannah | ☐ | continues the agricultural-data thread from the side quest |

---

## Workload balance
Still 8 tasks each — but now split along the **HDF5-access line**, since Hannah can't open the cube (too large):

- **Third** owns everything that touches the raw HDF5 directly: tile boundary upload, HDF5 transfer, data filtering/clipping, model setup, inference pipeline, and running both models. He exports the *outputs* of that pipeline (prediction rasters/vectors — much smaller than the cube) for Hannah to work with.
- **Hannah** owns everything that works from those derived outputs plus the ground-truth side: ICNF shapefile prep, agricultural-data sourcing, evaluation-metric definition, validating both models against ICNF ground truth, the pre- vs. fine-tuned comparison, and the agricultural-vs-natural-fire side quest.

This isn't just a fairness split — it's also better practice for a project with one multi-GB asset: keep the heavy data and the compute that needs it on one machine, and pass around only the small derived products (predictions, metrics, vectors) rather than duplicating the cube.

## Flags to resolve with the team
- **"Upload HDF5 to Hannah's OneDrive" may need rethinking.** If Hannah can't access the file once it's there either, moving a multi-GB cube just to have a second copy isn't obviously useful — consider replacing it with "Third exports and shares the processed/derived outputs Hannah actually needs" (see note in the Data Processing table).
- "Filter data" and the "Model Deployment"/"Model Evaluation" sections had no concrete sub-items — the breakdowns above are my best read of the stated objectives; worth a quick team check before committing to them.
- The ICNF-filtering task (Hannah) is **blocked** on Third's tile-boundary upload — worth chasing early given the 15 July deadline.

## Suggested critical path to 15 July
1. Tile boundary upload (Third) → unblocks ICNF shapefile filtering (Hannah)
2. In parallel: Third works the HDF5 side (filter/clip data → set up environment → build inference pipeline); Hannah works the ground-truth side (ICNF filtering, agricultural map sourcing, defining evaluation metrics)
3. Third runs both models over T29TPG and exports prediction outputs to Hannah (small files, not the cube)
4. Hannah validates both models against ICNF ground truth, compares pre- vs. fine-tuned performance, and folds in the agricultural side-quest evaluation
