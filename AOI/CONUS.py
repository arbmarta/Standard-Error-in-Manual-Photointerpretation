# Creates a polygon of the contiguous US using TIGER shapefile from https://www2.census.gov/geo/tiger/TIGER2025/STATE/

import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Load TIGER states shapefile
states = gpd.read_file("tl_2025_us_state/tl_2025_us_state.shp")

# Filter contiguous US (exclude Alaska, Hawaii, territories)
conus = states[~states['STUSPS'].isin(['AK', 'HI', 'PR', 'GU', 'VI', 'MP', 'AS'])]

# Merge all states into a single polygon
conus = gpd.GeoDataFrame(geometry=[unary_union(conus.geometry)], crs=conus.crs)

# Reproject to WGS 84
conus = conus.to_crs(epsg=3857)

# Save merged polygon to GeoPackage
conus.to_file("conus.gpkg", layer="conus", driver="GPKG")

# Load the GeoPackage
conus_gpkg = gpd.read_file("conus.gpkg", layer="conus")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
conus_gpkg.boundary.plot(ax=ax, color='black', linewidth=1)
conus_gpkg.plot(ax=ax, color='lightgreen', edgecolor='black', alpha=0.5)

ax.set_title("Contiguous US States")
ax.set_axis_off()  # optional: remove axes
plt.show()