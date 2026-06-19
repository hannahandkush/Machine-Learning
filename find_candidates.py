import geopandas as gpd
import pandas as pd
import numpy as np
import h5py
import os

# --- 1. Load Fire Data ---
fires = gpd.read_file("data/shapefiles/ground_truth_ICNF/ardida_2025.shp")
tile  = gpd.read_file("data/shapefiles/boundary_files/sentinel2_tiles_PT_terra_tm06.shp")
tile  = tile[tile["Name"] == "T29TPG"].to_crs(fires.crs)
fires_in_tile = gpd.clip(fires, tile)
large_fires = fires_in_tile[fires_in_tile["AreaHaSIG"] > 65].sort_values("AreaHaSIG", ascending=False)

# --- 2. Load Scene Data (Reconstructing df_complete from HDF5 if possible, or using dummy for logic) ---
# Note: In a real scenario, we'd run the notebook cells or load a saved CSV.
# Here, I'll simulate the logic using the structure found in the notebook.

HDF5_PATH = "data/hdf5/T29TPG.h5" # Based on notebook output
if os.path.exists(HDF5_PATH):
    with h5py.File(HDF5_PATH, "r") as f:
        timestamps = pd.to_datetime(f["timestamps"][:].astype(str))
        # We don't have the full cloud/pixel counts here without processing, 
        # so let's assume all timestamps are potential candidates for now 
        # or use a simplified filter if we can't reconstruct the full 'category'
        df_complete = pd.DataFrame({"date": timestamps})
else:
    # Fallback if HDF5 not at that specific path
    print(f"HDF5 not found at {HDF5_PATH}")
    df_complete = pd.DataFrame({"date": []})

# --- 3. Identify Candidates ---

# Before-chip candidates: July 1 - July 22
before_window = df_complete[
    (df_complete["date"] >= "2025-07-01") & 
    (df_complete["date"] < "2025-07-23")
]

print(f"Before candidates (July pre-fire): {before_window['date'].dt.date.tolist()}")

# After-chip candidates: per fire, within 30 days of DH_Fim
print("\nAfter-fire candidates:")
for _, fire in large_fires.head(4).iterrows():
    after = df_complete[
        (df_complete["date"] > fire["DH_Fim"]) &
        (df_complete["date"] <= fire["DH_Fim"] + pd.Timedelta(days=30))
    ]
    print(f"{fire['DH_Inicio'].date()} {fire['AreaHaSIG']:.0f}ha "
          f"-> after scenes: {after['date'].dt.date.tolist()}")
