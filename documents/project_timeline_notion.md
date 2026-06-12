# Vegetation Change Detection — Project Timeline (due 26 June)

Even split: 8 tasks each for Hannah and Third. Third takes the early sequencing/blocking work (8–10 June) so nothing downstream stalls; both run in parallel after that. Import this file into Notion (File → Import → Markdown) — checkboxes, headings, and nested sub-items convert directly to to-do blocks.

> ⚠️ **Compressed schedule.** The original plan assumed ~5 weeks (to 15 July); this version fits the same 16 tasks into 18 days. Phases now overlap tightly with very little slack — see the note at the bottom on what to cut first if the team falls behind.

---

## Phase 1 — Setup & unblocking — 8–10 June

**Third (sequencing-critical — these unblock everyone else)**

- [ ] **Upload tile boundary map (T29TPG) to GitHub** — *due 9 June*
    - [ ] Export/derive boundary from HDF5 metadata (`bounds_left/right/top/bottom`)
    - [ ] Convert to a shareable vector format (GeoJSON/Shapefile)
    - [ ] Push to repo
    - ⚠️ Blocks Hannah's ICNF filtering task below — top priority, do this first
- [ ] **Confirm/share HDF5 access with Hannah** — *due 8 June*
    - [ ] Verify Hannah can open and read the cube directly
    - [ ] Note access method/path in `data/README.md`
- [ ] **Obtain both model weights (Zenodo + Manuel's fine-tuned version)** — *due 10 June*
    - [ ] Download pre-fine-tuned weights from Zenodo
    - [ ] Retrieve fine-tuned weights from course SharePoint
    - [ ] Confirm both load correctly in a test script
- [ ] **Set up inference environment** — *due 10 June*
    - [ ] Identify framework/dependency versions used to train each model
    - [ ] Add to `environment.yml` / document any extra requirements
    - [ ] Smoke-test environment loads both models

**Hannah (parallel — not blocked by anything above)**

- [ ] **Check Hannah's data on agricultural changes** — *due 9 June*
    - [ ] Locate and review the dataset
    - [ ] Note format, coverage, and relevance to side quest
- [ ] **Source agricultural / forest land-cover map** — *due 10 June*
    - [ ] Check NUTS II / INE land-use data (ine.pt/scripts/db_ra_2019.html)
    - [ ] Confirm spatial coverage matches T29TPG study area
- [ ] **Define evaluation metrics against ICNF ground truth** — *due 10 June*
    - [ ] Shortlist metrics (IoU, F1, precision/recall, confusion matrix)
    - [ ] Decide on per-pixel vs. per-polygon comparison approach
    - [ ] Document the chosen protocol so both validation tasks use the same yardstick

---

## Phase 2 — Data prep & deployment — 11–17 June

**Hannah**

- [ ] **Filter ICNF shapefile, export CSV, translate, upload to GitHub** — *due 13 June*
    - [ ] Clip ICNF shapes to the study-area boundary (now available from Third)
    - [ ] Export attribute table to CSV
    - [ ] Translate fields/values (PT → EN)
    - [ ] Commit to repo
- [ ] **Filter data** — *due 17 June*
    - [ ] Clip HDF5 tile extent to study-area boundary
    - [ ] Mask cloud/nodata pixels using completeness categories from `data_exploration.ipynb`
    - [ ] Rasterize ICNF ground-truth polygons to the pixel grid
    - [ ] Define train/validation/test split (spatial split, not random — avoids chip-adjacency leakage)

**Third**

- [ ] **Build inference pipeline** — *due 14 June*
    - [ ] Load HDF5 tile → preprocessing (NDVI/NDWI/NBR, chip extraction)
    - [ ] Wire up model inference call
    - [ ] Output change/burn maps in a shareable format (raster/vector)
- [ ] **Run pre-fine-tuned model over T29TPG** — *due 17 June*
    - [ ] Execute pipeline with Zenodo weights
    - [ ] Export prediction outputs for Hannah's evaluation step

**Hannah**

- [ ] **Run fine-tuned model over T29TPG** — *due 17 June*
    - [ ] Execute pipeline with Manuel's weights
    - [ ] Export prediction outputs

---

## Phase 3 — Evaluation & comparison — 18–23 June

**Hannah**

- [ ] **Validate pre-fine-tuned model output vs. ground truth** — *due 20 June*
    - [ ] Apply the agreed metrics from Phase 1
    - [ ] Summarise accuracy on Portuguese data vs. its original (Asia-trained) performance
- [ ] **Compare pre- vs. fine-tuned performance** — *due 22 June*
    - [ ] Tabulate both validation results side by side
    - [ ] Draft the "did fine-tuning help, and how much" finding

**Third**

- [ ] **Validate fine-tuned model output vs. ground truth** — *due 20 June*
    - [ ] Apply the same agreed metrics for a fair comparison
- [ ] **(Side quest) Evaluate agricultural vs. natural-fire classification** — *due 22 June*
    - [ ] Cross-reference burned-area predictions with the agricultural/forest land-cover map
    - [ ] Classify detected changes as fire-driven vs. agricultural
    - [ ] Sanity-check against NUTS II / INE reference data

---

## Phase 4 — Write-up & submission — 24–26 June

**Both**

- [ ] Compile results, figures, and comparison tables
- [ ] Write up methodology, findings, and limitations
- [ ] Final review of repo (README, notebooks run cleanly, `.gitignore` covers large files)
- [ ] Submit by **26 June**

---

## Notes
- **Critical path:** Third's tile-boundary upload (by 9 June) and HDF5 access confirmation (by 8 June) are now the make-or-break items — both block Hannah's Phase 2 work, and there's no slack left to absorb a delay here. Treat these as the very first things tackled.
- **This schedule is tight — roughly 2.5x faster than the original.** If the team starts slipping, the cleanest things to descope (in order) are: (1) the agricultural-vs-natural-fire side quest — it was explicitly framed as a "side quest" relative to the main deployment/evaluation goal; (2) the depth of the model comparison write-up (a solid table + short discussion beats an exhaustive analysis under time pressure); (3) consider running both models on a smaller sub-region of T29TPG first to get *something* validated early, then scale up if time allows.
- The "filter data," "model deployment," and "model evaluation" sections had no sub-items in the original list — the breakdowns above are my best read of the project's stated objectives; worth a quick gut-check with the team before locking them in.
- Dates are working targets to hit 26 June, not hard commitments — but given how little buffer remains, slippage in Phase 1 will cascade directly into the deadline.
