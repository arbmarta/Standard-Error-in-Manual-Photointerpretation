import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
from pyproj import Transformer

## --------------------------------------------- DOWNLOAD META TILE INDEX ----------------------------------------------

# S3 bucket and key
bucket_name = 'dataforgood-fb-data'
key = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'
local_file = 'tiles.geojson'

# Initialize unsigned S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

# Download tiles.geojson if not already present
if not Path(local_file).exists():
    print(f"Downloading {local_file} from S3...")
    s3.download_file(bucket_name, key, local_file)
    print(f"Downloaded {local_file}")
else:
    print(f"Using existing {local_file}")

# Load AOI and reproject
aoi_gdf = gpd.read_file("conus.gpkg", layer="conus").to_crs(epsg=3857)
aoi_geom = aoi_gdf.union_all()

## ------------------------------------------------ GENERATE MIDWEST GRID ------------------------------------------------

# Define Midwest bounding box
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
lon_min, lon_max = -100, -92  # West to East
lat_min, lat_max = 43, 48     # South to North

minx, miny = transformer.transform(lon_min, lat_min)
maxx, maxy = transformer.transform(lon_max, lat_max)

print(f"\nMidwest bounding box:")
print(f"  Lat/Lon: ({lat_min}°N to {lat_max}°N, {lon_min}°W to {lon_max}°W)")

# Generate 100km grid
cell_size_km = 100
cell_size_m = cell_size_km * 1000

cells = []
x_range = range(int(minx), int(maxx), cell_size_m)
y_range = range(int(miny), int(maxy), cell_size_m)

for x in tqdm(x_range, desc=f"Generating {cell_size_km}km grid"):
    for y in y_range:
        cell = box(x, y, x + cell_size_m, y + cell_size_m)
        # Only keep cells that intersect the test area
        if aoi_geom.contains(cell):
            cells.append(cell)

# Create GeoDataFrame
grid_gdf = gpd.GeoDataFrame(
    {"cell_id": range(1, len(cells) + 1)},
    geometry=cells,
    crs="EPSG:3857"
)

# Save grid
filename = f"grid_{cell_size_km}km.gpkg"
grid_gdf.to_file(filename, driver="GPKG")
print(f"✅ Saved {len(grid_gdf)} cells to {filename}")

# Create union of all grid cells
grid_union = grid_gdf.union_all()

## ------------------------------------------------ FILTER TILES BY GRID ------------------------------------------------

# Load and filter tiles - only keep tiles intersecting the 100km grid
tiles = gpd.read_file(local_file).to_crs(epsg=3857)
tiles_in_grid = tiles[tiles.intersects(grid_union)]

# Save filtered tiles
tiles_in_grid.to_file("tiles_in_aoi_test.geojson", driver="GeoJSON")
tiles_in_grid["tile"].to_csv("tiles_in_aoi_test.txt", index=False, header=False)
print(f"Saved {len(tiles_in_grid)} tiles within 100km grid")

## ------------------------------------------------ VISUALIZE ------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))

# Plot CONUS boundary
aoi_plot_gdf = gpd.GeoDataFrame([1], geometry=[aoi_geom], crs="EPSG:3857")
aoi_plot_gdf.boundary.plot(ax=ax, color='black', linewidth=2, label='CONUS')

# Plot tiles
tiles_in_grid.boundary.plot(ax=ax, color='blue', linewidth=0.3, alpha=0.5, label='Tiles')

# Plot grid
grid_gdf.boundary.plot(ax=ax, color='red', linewidth=0.8, alpha=0.7, label=f'{cell_size_km}km Grid')

# Zoom into the grid extent with some padding
bounds = grid_gdf.total_bounds  # minx, miny, maxx, maxy
padding = 50000  # 50km padding in meters
ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
ax.set_ylim(bounds[1] - padding, bounds[3] + padding)

# Styling
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"{cell_size_km}km Midwest Grid\n({len(grid_gdf):,} cells, {len(tiles_in_grid):,} tiles)",
             fontsize=14, fontweight='bold')
ax.legend(loc='lower left', frameon=False, fontsize=10)
ax.grid(False)

plt.tight_layout()
plt.show()

# Summary
print("\n" + "=" * 60)
print("GRID SUMMARY")
print("=" * 60)
print(f"  • Number of cells: {len(grid_gdf):,}")
print(f"  • Cell area: {cell_size_km}×{cell_size_km} = {cell_size_km**2} km²")
print(f"  • Total coverage: ~{len(grid_gdf) * cell_size_km**2:,} km²")
print(f"  • Number of tiles: {len(tiles_in_grid):,}")