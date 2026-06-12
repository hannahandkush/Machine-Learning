# Evaluation Protocol and Methodological Justifications

This document records the decisions made during evaluation design and the explicit rationale for each. It is intended as both an internal reference and supporting material for the methods section of the final report.

---

## 1. Model inference context

The model is a pre-trained Swin Transformer (Liu et al., 2021) provided via a `predict(before_chip, after_chip, path_to_weights)` interface. It accepts two 10x256x256 uint16 arrays and returns a binary 256x256 array (0 = not burned, 1 = burned). No training or fine-tuning is performed on T29TPG data during this project. This determines several downstream evaluation choices, as noted below.

---

## 2. Metric selection

### 2.1 ROC/AUC — excluded

**Justification.** ROC analysis requires a continuous probability estimate for each prediction, not a hard class label (Fawcett, 2006). The ROC curve is constructed by sweeping a decision threshold across the probability output and recording the true positive rate against the false positive rate at each threshold. Without a probability surface there is no threshold to sweep and the curve is not defined. The course lecture (Campagnolo, 2025, at 20:47) states this explicitly: "the model needs to give a probability estimate, not just a hard class label, for the ROC curve to work." The `predict()` function returns a hard binary array (0 or 1). Inclusion of ROC/AUC would be methodologically incorrect given this output format.

### 2.2 Overall accuracy — excluded

**Justification.** Burned pixels are a small and spatially irregular fraction of any Sentinel-2 tile. A trivial classifier that assigns every pixel to the "not burned" class would achieve greater than 95% overall accuracy while detecting nothing. This reflects a well-documented limitation of overall accuracy under severe class imbalance (Stehman & Foody, 2019; Campagnolo, 2025): the metric is dominated by the majority class and does not reflect performance on the class of interest. Using it as a headline metric would misrepresent model performance.

### 2.3 Precision, recall, and F1 — included, computed on the burned class only

**Justification.** Precision (1 - commission error) measures the fraction of pixels predicted as burned that are actually burned. Recall (1 - omission error) measures the fraction of truly burned pixels that the model detects. Both are operationally meaningful for fire mapping: high commission error propagates false alarms into downstream risk assessments; high omission error means fires go undetected. F1 is their harmonic mean and provides a single balanced score for the burned class. All three are computed on the burned class only, consistent with the binary imbalanced evaluation framework recommended by Raschka et al. (2022, Ch. 6) and applied in burned area remote sensing literature (Roteta et al., 2019).

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# y_true and y_pred are flattened 1D arrays of 0/1 labels
precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
```

### 2.4 Matthews Correlation Coefficient (MCC) — primary single-number summary

**Justification.** MCC was introduced by Matthews (1975) and incorporates all four cells of the confusion matrix:

```
MCC = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Unlike F1, MCC is symmetric with respect to class assignment: swapping positive and negative labels returns the same coefficient. Chicco & Jurman (2020) demonstrate empirically that MCC is a more reliable single-number summary than F1 or accuracy for imbalanced binary classification, specifically because F1 ignores the true negative count and can be artificially inflated when the model predicts few positives in an imbalanced setting. Raschka et al. (2022, Ch. 6) and Campagnolo (2025) both identify MCC as the preferred summary metric for this class of problem. It is used as the headline metric in the final comparison table.

```python
from sklearn.metrics import matthews_corrcoef, confusion_matrix

mcc = matthews_corrcoef(y_true, y_pred)

# Full confusion matrix for reporting
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
```

### 2.5 Per-acquisition breakdown — included

Metrics are also reported per before/after acquisition pair to assess whether performance degrades with increasing temporal gap between the acquisition date and the fire date, or with rising cloud fraction in the after scene. This is diagnostic rather than a tuning step, and follows the practice of stratifying accuracy assessment by data quality strata recommended in Stehman & Foody (2019).

---

## 3. Ground truth rasterisation

### 3.1 No train/test split

**Justification.** The model is used purely for inference. No parameters are estimated from T29TPG data, so there is no risk of overfitting and no need to partition the data. All evaluable chips within T29TPG are used for evaluation. This is consistent with independent accuracy assessment of pre-trained models against external reference data (Chuvieco et al., 2019).

### 3.2 Polygon interior erosion before rasterisation

**Justification.** Pixels straddling the fire perimeter boundary contain a mix of burned and unburned land cover within the same 10 m footprint. Assigning a hard label to these edge pixels introduces systematic label noise at the boundary, which inflates error rates regardless of model quality. Interior erosion by one pixel (10 m) removes the ambiguous boundary ring before rasterisation. This is a standard practice in remote sensing accuracy assessment to avoid mixed-pixel boundary effects (Stehman & Foody, 2019).

```python
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

# Load ICNF polygons and reproject to tile CRS (EPSG:32629)
fires = gpd.read_file("data/shapefiles/ground_truth_ICNF/ardida_2025.shp")
fires = fires.to_crs("EPSG:32629")

# Erode each polygon by 1 pixel (10 m) to remove boundary ambiguity
fires["geometry_eroded"] = fires.geometry.buffer(-10)

# Remove any polygons that collapsed to empty after erosion (very small fires)
fires_valid = fires[~fires["geometry_eroded"].is_empty].copy()
fires_valid = fires_valid.set_geometry("geometry_eroded")

# Rasterise against the chip grid
# (chip_transform and chip_shape are derived from HDF5 spatial metadata)
label_array = rasterize(
    [(geom, 1) for geom in fires_valid.geometry],
    out_shape=chip_shape,
    transform=chip_transform,
    fill=0,
    dtype="uint8",
)
```

---

## 4. Fire size and chip scale

### 4.1 Fire size distribution

**Observation.** The ICNF dataset contains 182 fire events within T29TPG. The distribution is heavily right-skewed, consistent with the known power-law behaviour of wildfire size distributions (Chuvieco et al., 2019):

| Statistic | Value |
|---|---|
| Total events | 182 |
| Median event size | 1.98 ha |
| Mean event size | 70.1 ha |
| Max event size | 4,196.6 ha |

A 256x256 chip at 10 m resolution covers 655 ha. A 1.98 ha fire occupies approximately 0.3% of one chip, meaning roughly 3 pixels out of 65,536 would be labeled burned. Per-pixel metrics become unreliable at this within-chip imbalance, and MCC and F1 are undefined when there are no positive predictions.

```python
# Reproduce the fire size distribution summary
fires_in_tile = gpd.clip(fires, tile_t29tpg)

print(fires_in_tile["AreaHaSIG"].describe())
print(f"\nChip area at 10m: {256 * 10 * 256 * 10 / 10000:.0f} ha")
print(f"Median fire as % of chip: {fires_in_tile['AreaHaSIG'].median() / 655 * 100:.2f}%")
```

### 4.2 Evaluable fire threshold

**Decision.** Fires below 65 ha are excluded from quantitative evaluation. 65 ha represents approximately 10% of chip area (655 ha * 0.10 = 65.5 ha), the practical minimum for per-pixel metrics to be interpretable. This yields 15 evaluable events. The headline evaluation set is the 4 events above 945 ha, which span multiple chips and allow robust spatial statistics. The threshold is consistent with the minimum mapping unit considerations discussed in Roteta et al. (2019) for Sentinel-2 burned area products.

```python
CHIP_AREA_HA = (256 * 10) ** 2 / 1e4          # 655.36 ha
MIN_COVERAGE_FRACTION = 0.10
SIZE_THRESHOLD_HA = CHIP_AREA_HA * MIN_COVERAGE_FRACTION  # ~65 ha

large_fires = (
    fires_in_tile[fires_in_tile["AreaHaSIG"] > SIZE_THRESHOLD_HA]
    .sort_values("AreaHaSIG", ascending=False)
    .reset_index(drop=True)
)

print(f"Evaluable fires (>{SIZE_THRESHOLD_HA:.0f} ha): {len(large_fires)}")
print(f"Headline fires (>945 ha): {(large_fires['AreaHaSIG'] > 945).sum()}")
```

---

## 5. Before/after pair selection

> **Note on sections 5.1-5.3.** The classification logic and selection strategy are finalised. The specific acquisition dates cannot be confirmed until the HDF5 metadata has been extracted on a system with full file access. Pending outputs are indicated in italics throughout. See `scene_selection_request.md` for the script and the outstanding request to Third.

### 5.1 Scene quality classification

Scenes are labelled using per-timestamp orbit coverage fraction and cloud percentage stored in the HDF5 metadata. Thresholds follow the completeness categories defined in the data exploration notebook and are consistent with scene selection practice in Sentinel-2 time series analysis (Roteta et al., 2019). The classification logic is fixed; *the resulting per-scene label table is pending extraction from the full HDF5 archive.*

```python
import h5py
import pandas as pd

with h5py.File("data/hdf5/T29TPG.h5", "r") as f:
    ts_raw = f["timestamps"][:]
    timestamps = pd.to_datetime(
        [t.decode("utf-8") if isinstance(t, bytes) else t for t in ts_raw]
    )
    df_scenes = pd.DataFrame({
        "date":      timestamps,
        "cloud_pct": f["cloud_cover_pt"][:],
        "total_px":  f["pixel_count_pt"][:],
        "orbit_px":  f["count_orbit_pixels_pt"][:],
    })

df_scenes["swath_coverage"] = df_scenes["orbit_px"] / df_scenes["total_px"]

def classify_scene(row):
    if row.swath_coverage >= 0.95 and row.cloud_pct <= 10:
        return "full_clear"
    if row.swath_coverage < 0.50:
        return "edge_pass"
    if row.swath_coverage >= 0.50 and row.cloud_pct <= 20:
        return "partial_clear"
    return "clouded"

df_scenes["label"] = df_scenes.apply(classify_scene, axis=1)
```

| Label | Condition | Inference eligible |
|---|---|---|
| `full_clear` | swath >= 95% AND cloud <= 10% | Yes |
| `partial_clear` | swath >= 50% AND cloud <= 20% | Yes |
| `edge_pass` | swath < 50% | No |
| `clouded` | swath >= 50% AND cloud > 20% | No |

`edge_pass` scenes are excluded regardless of cloud cover: a chip from a partially observed tile may contain NoData (65535) across a large fraction of its pixels, which corrupts the uint16-to-uint8 normalisation step inside `predict()`.

### 5.2 Shared July baseline for August/September fires

**Observation.** Seven of the 15 large fires occur within a 24-day window from August 7 to August 30. A fire-specific before scene for each event would risk including an earlier August burn scar in the baseline for a later event in the same region, suppressing recall. Baseline contamination is a known source of evaluation error in multi-event change detection studies (Chuvieco et al., 2019).

**Strategy (confirmed).** A single shared pre-season baseline from July 1-25 is used for all August and September fires. The window ends July 25 to remain before the first large fire (July 23, Montalegre, 237 ha). If no `full_clear` scene exists in this window, the cleanest June scene is used as a fallback.

*Pending: confirmation of whether a `full_clear` scene exists in July 1-25, and if not, the specific June date with the lowest cloud fraction. The baseline acquisition date will be updated here once Third's output is received.*

```python
JULY_START, JULY_END = "2025-07-01", "2025-07-25"
JUNE_START, JUNE_END = "2025-06-01", "2025-06-30"

USABLE = {"full_clear", "partial_clear"}

july_candidates = df_scenes[
    (df_scenes["date"] >= JULY_START) &
    (df_scenes["date"] <= JULY_END) &
    (df_scenes["label"].isin(USABLE))
].sort_values("cloud_pct")

if not july_candidates.empty:
    baseline_date = july_candidates.iloc[0]["date"]
    print(f"[+] July baseline: {baseline_date.date()} ({july_candidates.iloc[0]['label']})")
else:
    # Fallback: cleanest June scene
    june_candidates = df_scenes[
        (df_scenes["date"] >= JUNE_START) &
        (df_scenes["date"] <= JUNE_END) &
        (df_scenes["label"].isin(USABLE))
    ].sort_values("cloud_pct")
    baseline_date = june_candidates.iloc[0]["date"]
    print(f"[!] June fallback baseline: {baseline_date.date()} (seasonal penalty applies)")
```

**June fallback justification.** June introduces a seasonal offset relative to August (higher NDVI, fuller canopy cover). This is accepted because fire-driven change in SWIR and NIR bands far exceeds seasonal spectral variance (Roteta et al., 2019), and a partially observed or cloud-contaminated July scene degrades spatial coverage more severely than a temporally offset but complete June scene.

### 5.3 After-fire scene selection

The ICNF field `DH_Fim` (datetime64) marks the official fire end date. The after acquisition is the earliest usable scene on or after `DH_Fim`. A 30-day search window is applied; events with no usable scene within 30 days are excluded from quantitative evaluation.

*Pending: for each of the 5 largest fires, confirmation of whether a `full_clear` or `partial_clear` scene exists within 30 days of `DH_Fim`. Any events without a usable after scene will be removed from the evaluable set and the final event count updated accordingly.*

```python
AFTER_WINDOW_DAYS = 30

def get_after_scene(fire_end, df_scenes):
    candidates = df_scenes[
        (df_scenes["date"] >= fire_end) &
        (df_scenes["date"] <= fire_end + pd.Timedelta(days=AFTER_WINDOW_DAYS)) &
        (df_scenes["label"].isin(USABLE))
    ].sort_values("date")
    if candidates.empty:
        return None
    return candidates.iloc[0]

for _, fire in large_fires.iterrows():
    after = get_after_scene(fire["DH_Fim"], df_scenes)
    if after is None:
        print(f"[SKIP] {fire['AreaHaSIG']:.0f} ha in {fire['PI_Conc']} - no usable after scene in 30 days")
    else:
        print(f"[OK]   {fire['AreaHaSIG']:.0f} ha in {fire['PI_Conc']} - after: {after['date'].date()} ({after['label']})")
```

---

## 6. Summary decision table

| Decision | Choice | Primary justification |
|---|---|---|
| ROC/AUC | Excluded | `predict()` returns hard binary 0/1; no probability output (Fawcett, 2006; Campagnolo, 2025) |
| Overall accuracy | Excluded | Dominated by majority class; trivial classifier scores >95% (Stehman & Foody, 2019) |
| Primary metric | MCC | Symmetric, uses all four confusion matrix cells (Chicco & Jurman, 2020; Raschka et al., 2022) |
| Supporting metrics | Precision, recall, F1 (burned class only) | Commission/omission operationally meaningful for fire mapping (Roteta et al., 2019) |
| Train/test split | Not applicable | Inference-only model; no parameters estimated from T29TPG |
| Ground truth rasterisation | Polygon interior erosion (1 px = 10 m) | Removes boundary label ambiguity (Stehman & Foody, 2019) |
| Fire size threshold | >65 ha (~10% of chip area) | Minimum for reliable per-pixel metrics; derived from chip geometry |
| Evaluation set size | 15 fires (headline: 4 > 945 ha) | Threshold applied to 182-event T29TPG distribution |
| Baseline period | July 1-25 (June fallback if no full_clear) | Pre-dates all August fires; avoids burn scar contamination (Chuvieco et al., 2019) |
| Scene eligibility | `full_clear` or `partial_clear` only | `edge_pass`/`clouded` chips degrade or corrupt model input |
| Pair matching fields | `DH_Inicio` (before), `DH_Fim` (after) | Explicit fire start/end timestamps from ICNF |
| After-scene window | 30 days from `DH_Fim` | Balances cloud-free probability against temporal proximity |
| **Baseline acquisition date** | *Pending HDF5 output from Third* | *July date TBC; June fallback TBC if no full_clear in July* |
| **Final evaluable event count** | *Pending HDF5 output from Third* | *Up to 15; events without a usable after scene will be excluded* |

---

## References

Campagnolo, M.L. (2025). *Lecture T6: Classification and Accuracy Assessment*. Practical Machine Learning, Instituto Superior de Agronomia, Universidade de Lisboa.

Chicco, D. & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6. https://doi.org/10.1186/s12864-019-6413-7

Chuvieco, E., Mouillot, F., van der Werf, G.R., San Miguel, J., Tanasse, M., Koutsias, N., Garcia, M., Yebra, M., Padilla, M., Gitas, I., Heil, A., Hawbaker, T.J. & Doblas-Reyes, F.J. (2019). Historical background and current developments for mapping burned area from satellite Earth observation. *Remote Sensing of Environment*, 225, 45-64. https://doi.org/10.1016/j.rse.2019.02.013

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. & Guo, B. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012-10022. https://doi.org/10.1109/ICCV48922.2021.00986

Matthews, B.W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta (BBA) - Protein Structure*, 405(2), 442-451. https://doi.org/10.1016/0005-2795(75)90109-9

Raschka, S., Liu, Y.H. & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing.

Roteta, E., Bastarrika, A., Padilla, M., Storm, T. & Chuvieco, E. (2019). Development of a Sentinel-2 burned area algorithm: Generation of a small fire database for sub-Saharan Africa. *Remote Sensing of Environment*, 222, 1-17. https://doi.org/10.1016/j.rse.2018.12.011

Stehman, S.V. & Foody, G.M. (2019). Key issues in rigorous accuracy assessment of land cover products. *Remote Sensing of Environment*, 231, 111199. https://doi.org/10.1016/j.rse.2019.05.018
