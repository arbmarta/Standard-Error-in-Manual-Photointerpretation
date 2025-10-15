import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm

## --------------------------------------------- DOWNLOAD META TILE INDEX ----------------------------------------------
#region

# S3 bucket and key
bucket_name = 'dataforgood-fb-data'
key = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'
local_file = 'tiles.geojson'
reprojected_file = 'tiles_us_albers.geojson'
aoi_filtered_file = 'tiles_in_aoi.geojson'

# Initialize unsigned S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

# Download tiles.geojson if not already present
if not Path(local_file).exists():
    print(f"Downloading {local_file} from S3...")
    s3.download_file(bucket_name, key, local_file)
    print(f"Downloaded {local_file}")
else:
    print(f"Using existing {local_file}")

# Load AOI
aoi_gdf = gpd.read_file("conus.gpkg", layer="conus").to_crs(epsg=5070)
aoi_geom = aoi_gdf.union_all()  # dissolve to single polygon

# Reproject tiles directly
tiles = gpd.read_file(local_file).to_crs(epsg=5070)

# Spatial filter: tiles fully within AOI
tiles_in_aoi = tiles[tiles.within(aoi_geom)]

# Save AOI-filtered tiles
tiles_in_aoi.to_file("tiles_in_aoi.geojson", driver="GeoJSON")
print(f"Saved {len(tiles_in_aoi)} tiles in AOI to tiles_in_aoi.geojson")

# Save list of tile IDs to text file
tiles_in_aoi["tile"].to_csv("tiles_in_aoi.txt", index=False, header=False)

#endregion

## ----------------------------------------------- MAP THE AOI AND TILES -----------------------------------------------
#region

fig, ax = plt.subplots(figsize=(10, 8))

# Plot tiles as green lines
tiles_in_aoi.boundary.plot(ax=ax, color='green', linewidth=1, label=f'Tiles ({len(tiles_in_aoi)})')

# Plot AOI outline in black
aoi_plot_gdf = gpd.GeoDataFrame([1], geometry=[aoi_geom])
aoi_plot_gdf.boundary.plot(ax=ax, color='black', linewidth=2, label='Contiguous United States')

ax.set_xticks([]) # Remove ticks
ax.set_yticks([]) # Remove ticks
ax.set_title("") # Remove title
ax.set_xlabel("") # Remove axis labels
ax.set_ylabel("") # Remove axis labels
ax.grid(False)

ax.legend(loc='lower left', frameon=False, fontsize=17) # Move legend to bottom left and remove box

plt.tight_layout()
plt.savefig("Tiles in AOI.pdf", format="pdf")
plt.show()

#endregion

## ------------------------------------------------ GENERATE GRID CELLS ------------------------------------------------
#region

# Load AOI-filtered tiles
tiles_in_aoi = gpd.read_file("tiles_in_aoi.geojson").to_crs(epsg=5070)

# Merge all polygons to one big AOI polygon
aoi_union = tiles_in_aoi.union_all()
grid_sizes = [3, 24, 54] # Grid sizes in km

for cell_size_km in grid_sizes:
    cell_size_m = cell_size_km * 1000
    grid_label = f"{cell_size_km}km"
    print(f"\n\n{'🔲 PROCESSING ' + grid_label + ' GRID':=^60}")

    try:
        # Get AOI bounds
        minx, miny, maxx, maxy = aoi_union.bounds

        # Generate grid cells with live tqdm
        cells = []
        x_range = range(int(minx), int(maxx), cell_size_m)
        y_range = range(int(miny), int(maxy), cell_size_m)

        for x in tqdm(x_range, desc=f"Grid {grid_label} X-loop"):
            for y in y_range:
                cell = box(x, y, x + cell_size_m, y + cell_size_m)
                if aoi_union.contains(cell):
                    cells.append(cell)

        grid_gdf = gpd.GeoDataFrame(geometry=cells, crs="EPSG:5070")

        # Create GeoDataFrame with explicit cell_id
        grid_gdf = gpd.GeoDataFrame(
            {"cell_id": range(1, len(cells)+1)},
            geometry=cells,
            crs="EPSG:5070"
        )

        # Save as GeoPackage
        filename = f"grid_{cell_size_km}km.gpkg"
        grid_gdf.to_file(filename, driver="GPKG")
        print(f"✅ Saved {len(grid_gdf)} {grid_label} cells to {filename}")

    except Exception as e:
        print(f"❌ Error generating {grid_label} grid: {e}")

#endregion

## ------------------------------------------------ VISUALIZE GRIDS WITH CONUS ------------------------------------------------

print(f"\n\nLoad .gpkg files to plot...")
# Load CONUS AOI
aoi_gdf = gpd.read_file("conus.gpkg", layer="conus").to_crs(epsg=5070)
aoi_geom = aoi_gdf.union_all()

# Load grid files
grid_sizes = [3, 24, 54]  # km
grids = {}
for size in grid_sizes:
    filename = f"grid_{size}km.gpkg"
    grids[size] = gpd.read_file(filename)
    print(f"Loaded {filename}: {len(grids[size])} cells, CRS: {grids[size].crs}")

# Create multi-panel figure
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, size in enumerate(grid_sizes):
    ax = axes[idx]

    # Plot CONUS boundary
    aoi_plot_gdf = gpd.GeoDataFrame([1], geometry=[aoi_geom], crs="EPSG:5070")
    aoi_plot_gdf.boundary.plot(ax=ax, color='black', linewidth=2, label='CONUS')

    # Plot grid cells
    grids[size].boundary.plot(ax=ax, color='red', linewidth=0.5, alpha=0.7, label=f'{size}km Grid')

    # Styling
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{size}km Grid\n({len(grids[size]):,} cells)", fontsize=14, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.legend(loc='lower left', frameon=False, fontsize=10)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("GRID SUMMARY")
print("=" * 60)
for size in grid_sizes:
    grid = grids[size]
    total_area_km2 = (len(grid) * size * size)
    print(f"\n{size}km Grid:")
    print(f"  • Number of cells: {len(grid):,}")
    print(f"  • Cell area: {size}×{size} = {size ** 2} km²")
    print(f"  • Total coverage: ~{total_area_km2:,} km²")
    print(f"  • CRS: {grid.crs}")