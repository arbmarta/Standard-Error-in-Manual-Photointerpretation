import geopandas as gpd
from functions import generate_systematic_sample_points, plot_sample_points_map

# ----------------------------------------------------------------------------------------------------------------------
# LOAD EXISTING TILE DATA (Optional - for visualization only)
# ----------------------------------------------------------------------------------------------------------------------

# If you want to visualize tile boundaries on the map, load them here
# Otherwise, this is optional since we're working directly with raster files
try:
    tiles_in_aoi = gpd.read_file("tiles_in_aoi.geojson")
    print(f"Loaded {len(tiles_in_aoi)} tiles from tiles_in_aoi.geojson")
except FileNotFoundError:
    print("tiles_in_aoi.geojson not found - skipping tile visualization")
    tiles_in_aoi = None

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE SAMPLE POINTS WITH RASTER VALIDATION
# ----------------------------------------------------------------------------------------------------------------------

# Path to your downloaded Meta CHM raster files (binary .tif files)
raster_folder = "Meta CHM Binary"

# Load your grid (example shown here)
grid_gdf = gpd.read_file("AOI/grid_10km.gpkg")

# Generate sample points with raster data validation
sample_points = generate_systematic_sample_points(
    grid_gdf,
    points_per_cell=10,
    raster_folder=raster_folder  # Add this to check for raster coverage
)

# Save sample points
sample_points.to_file("sample_points_10km.gpkg", driver="GPKG")
print(f"\nSaved sample points to sample_points_10km.gpkg")

# Visualize
plot_sample_points_map(
    sample_points,
    tiles_gdf=tiles_in_aoi,
    grid_size_km=10
)