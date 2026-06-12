# Change Detection Candidate Selection — T29TPG 2025

This document records the data exploration outputs already produced locally and the logic used to translate them into evaluation decisions. The final step requires running the scene quality script against the full HDF5 archive on a system with sufficient storage.

---

## 1. Fire Data Exploration (already run locally)

The ICNF `ardida_2025` shapefile was clipped to the T29TPG tile boundary and filtered for events large enough to be evaluable at chip scale (one 256x256 chip at 10 m = 655 ha; threshold set at >65 ha to capture fires spanning at least ~10% of a chip).

### Summary statistics

| Statistic | Value |
|---|---|
| Total fire events in tile | 182 |
| Total burned area | 13,035.4 ha |
| Median event size | 1.98 ha |
| Mean event size | 70.1 ha |
| Max event size | 4,196.6 ha (Montalegre, Aug) |
| Peak month by area | August (9,166 ha) |
| Events > 65 ha | 15 |
| Events > 945 ha | 4 |

The median being 1.98 ha (sub-chip) means the bulk of events are too small for pixel-level evaluation. The evaluation focuses on the 15 events above 65 ha; headline results use the 4 events above 945 ha.

### Large fire events (>65 ha)

| Start Date | End Date | Area (ha) | Municipality |
|---|---|---|---|
| 2025-08-18 | 2025-08-21 | 4,196.59 | Montalegre |
| 2025-08-28 | 2025-08-30 | 3,159.85 | Vinhais |
| 2025-09-20 | 2025-09-20 | 1,101.88 | Montalegre |
| 2025-08-08 | 2025-08-09 | 945.34 | Ribeira de Pena |
| 2025-09-08 | 2025-09-09 | 554.38 | Chaves |
| 2025-08-07 | 2025-08-10 | 489.53 | Vila Pouca de Aguiar |
| 2025-03-31 | 2025-03-31 | 313.74 | Bragança |
| 2025-10-10 | 2025-10-10 | 264.84 | Montalegre |
| 2025-07-25 | 2025-07-25 | 248.54 | Bragança |
| 2025-10-19 | 2025-10-19 | 247.89 | Montalegre |
| 2025-07-23 | 2025-07-24 | 237.08 | Montalegre |
| 2025-08-13 | 2025-08-15 | 146.46 | Vinhais |
| 2025-09-22 | 2025-09-24 | 128.64 | Montalegre |
| 2025-04-28 | 2025-04-28 | 109.31 | Montalegre |
| 2025-08-13 | 2025-08-14 | 90.33 | Macedo de Cavaleiros |

---

## 2. Impact on evaluation decisions

**Why a shared July baseline.** Seven of the 15 large fires fall within a 24-day August window (Aug 7-30). If a fire-specific before scene were used for each, a burn scar from an earlier August fire would appear in the "before" chip of a later one. A single pre-season baseline from July eliminates this contamination for all August and September events.

**Why July 1-25 (not later).** The first large fire starts July 23. A baseline acquired on or after July 23 risks containing partial burn signal. The July window is therefore capped at July 25 to allow the day-25 Braganca fire to be evaluated with a before scene from earlier that month.

**June fallback.** If no `full_clear` scene exists in July, the cleanest June scene is used. June introduces a seasonal offset (higher NDVI, greener canopy) but the Swin Transformer is designed for large spectral discontinuities; the offset is acceptable given the magnitude of fire-driven change.

**Scene quality classification.** Scenes are labelled using per-timestamp orbit coverage and cloud fraction stored in the HDF5 metadata:

| Label | Condition |
|---|---|
| `full_clear` | swath coverage >= 95% AND cloud % <= 10 |
| `partial_clear` | swath coverage >= 50% AND cloud % <= 20 |
| `edge_pass` | swath coverage < 50% (tile only partially observed) |
| `clouded` | everything else |

Only `full_clear` or `partial_clear` scenes are passed to the model. `edge_pass` scenes are excluded because a chip drawn from a tile edge may contain NoData (encoded 65535) across a significant fraction of its pixels, which corrupts the model input regardless of cloud cover.

---

## 3. Script (run from project root on HDF5 system)

```python
import geopandas as gpd
import pandas as pd
import h5py
import numpy as np
import os

# ==============================================================================
# 1. FIRE DATA EXPLORATION (ICNF 2025)
# ==============================================================================
print("--- 1. LOADING FIRE DATA ---")
fire_path = "data/shapefiles/ground_truth_ICNF/ardida_2025.shp"
tile_path = "data/shapefiles/boundary_files/sentinel2_tiles_PT_terra_tm06.shp"

fires = gpd.read_file(fire_path)
print(f"Columns: {fires.columns.tolist()}")
print("\nFirst 5 rows:")
print(fires.head())
print("\nData Types:")
print(fires.dtypes)

# Filter for Tile T29TPG and Large Fires (> 65 ha)
tiles = gpd.read_file(tile_path)
tile_t29tpg = tiles[tiles["Name"] == "T29TPG"].to_crs(fires.crs)

fires_in_tile = gpd.clip(fires, tile_t29tpg)
large_fires = fires_in_tile[fires_in_tile["AreaHaSIG"] > 65].sort_values("AreaHaSIG", ascending=False)

print(f"\nFound {len(fires_in_tile)} fire events in T29TPG.")
print(f"Filtering for events > 65 ha (Total: {len(large_fires)})")

# ==============================================================================
# 2. SATELLITE METADATA & QUALITY CLASSIFICATION
# ==============================================================================
print("\n--- 2. ANALYZING HDF5 METADATA ---")
hdf5_path = "data/hdf5/T29TPG.h5"

if not os.path.exists(hdf5_path):
    print(f"ERROR: HDF5 file not found at {hdf5_path}")
    exit()

with h5py.File(hdf5_path, "r") as f:
    ts_raw = f["timestamps"][:]
    timestamps = pd.to_datetime([t.decode('utf-8') if isinstance(t, bytes) else t for t in ts_raw])

    # Extract quality metrics stored in HDF5 datasets
    df_scenes = pd.DataFrame({
        "date":      timestamps,
        "cloud_pct": f["cloud_cover_pt"][:],
        "total_px":  f["pixel_count_pt"][:],
        "orbit_px":  f["count_orbit_pixels_pt"][:],
    })

# Define classification logic for "usable" scenes
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

# ==============================================================================
# 3. BASELINE SELECTION (JULY WITH JUNE FALLBACK)
# ==============================================================================
print("\n--- 3. BASELINE SEARCH (BEFORE FIRE) ---")
july_baseline = df_scenes[(df_scenes["date"] >= "2025-07-01") & (df_scenes["date"] <= "2025-07-25")]
june_fallback = df_scenes[(df_scenes["date"] >= "2025-06-01") & (df_scenes["date"] <= "2025-06-30")]

print("July Candidates:")
print(july_baseline[['date', 'label', 'cloud_pct']].to_string(index=False))

if not any(july_baseline["label"] == "full_clear"):
    print("\n[!] No 'full_clear' in July. Checking June Fallback (Green Seasonal Penalty):")
    print(june_fallback[['date', 'label', 'cloud_pct']].to_string(index=False))
else:
    print("\n[+] Ideal July baseline found.")

# ==============================================================================
# 4. IMPACT SELECTION (AFTER FIRE)
# ==============================================================================
print("\n--- 4. IMPACT SEARCH (AFTER FIRE) ---")
for _, fire in large_fires.head(5).iterrows():
    fire_end = fire['DH_Fim']
    after_window = df_scenes[(df_scenes["date"] > fire_end) &
                             (df_scenes["date"] <= fire_end + pd.Timedelta(days=30))]

    print(f"\nEVENT: {fire['AreaHaSIG']:.1f} ha in {fire['PI_Conc']} (Ends: {fire_end.date()})")
    print("Candidate 'After' scenes (30-day window):")
    print(after_window[['date', 'label', 'cloud_pct']].to_string(index=False))
```

---

## 4. Request

Hi Third,

The fire data exploration above was run locally. The final step requires the full HDF5 file (`T29TPG.h5`, 32 GB) which I cannot access due to storage and sync constraints.

Please run the script from the project root on your system and paste the full console output. From the output I need to know:

1. Is there a `full_clear` row in the July 1-25 candidate list? If yes, which date?
2. If not, which June date has the lowest `cloud_pct`?
3. For each of the 5 largest fires, are there `full_clear` or `partial_clear` scenes within 30 days after the fire end date?

These answers lock in the specific timestamps passed to the model. Without them we cannot extract the before/after chip pairs and cannot proceed to inference.

Thanks
