import geopandas as gpd
import pandas as pd

fires = gpd.read_file("data/shapefiles/ground_truth_ICNF/ardida_2025.shp")
tile  = gpd.read_file("data/shapefiles/boundary_files/sentinel2_tiles_PT_terra_tm06.shp")
tile  = tile[tile["Name"] == "T29TPG"].to_crs(fires.crs)

fires_in_tile = gpd.clip(fires, tile)

large_fires = fires_in_tile[fires_in_tile["AreaHaSIG"] > 65].sort_values("AreaHaSIG", ascending=False)
print(large_fires[["DH_Inicio", "DH_Fim", "AreaHaSIG", "PI_Conc"]].to_string())
