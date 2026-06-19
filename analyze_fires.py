import geopandas as gpd
import pandas as pd

fires = gpd.read_file("data/shapefiles/ground_truth_ICNF/ardida_2025.shp")
tile  = gpd.read_file("data/shapefiles/boundary_files/sentinel2_tiles_PT_terra_tm06.shp")
tile  = tile[tile["Name"] == "T29TPG"].to_crs(fires.crs)

fires_in_tile = gpd.clip(fires, tile)

print(f"Events in tile: {len(fires_in_tile)}")
print(f"Date range: {fires_in_tile['DH_Inicio'].min()} to {fires_in_tile['DH_Inicio'].max()}")
print(f"\nArea distribution (AreaHaSIG):")
print(fires_in_tile["AreaHaSIG"].describe())
print(f"\nTotal burned area: {fires_in_tile['AreaHaSIG'].sum():.1f} ha")
print(f"\nFires > 65 ha (at least 10% of one chip): {(fires_in_tile['AreaHaSIG'] > 65).sum()}")
print(f"Fires > 655 ha (full chip): {(fires_in_tile['AreaHaSIG'] > 655).sum()}")

# Monthly distribution
fires_in_tile["month"] = fires_in_tile["DH_Inicio"].dt.month
print(f"\nFires by month:\n{fires_in_tile.groupby('month')['AreaHaSIG'].agg(['count','sum'])}")
