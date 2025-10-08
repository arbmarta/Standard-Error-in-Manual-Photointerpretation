import geopandas as gpd
import os
from functions import process_grid_cells_with_raster_association

# ATTENTION: This works best if you pip install rtree and import into functions.py

USE_TEST_SETTINGS = True

# Import the grids
grid_1 = gpd.read_file('AOI/grid_1km.gpkg')
grid_20 = gpd.read_file('AOI/grid_20km.gpkg')
grid_40 = gpd.read_file('AOI/grid_40km.gpkg')
grid_100 = gpd.read_file('AOI/grid_100km.gpkg')

grids = [grid_100] if USE_TEST_SETTINGS else [grid_1, grid_20, grid_40]

# Main execution
if __name__ == "__main__":
    # Configuration
    raster_folder = "Meta CHM Binary"
    output_directory = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Choose which grids to process
    grid_paths = (
        ["AOI/grid_100km.gpkg"] if USE_TEST_SETTINGS
        else ["AOI/grid_1km.gpkg", "AOI/grid_20km.gpkg", "AOI/grid_40km.gpkg"]
    )

    # Process each grid size
    for grid_path in grid_paths:
        grid = gpd.read_file(grid_path)

        # Skip if grid has no cells
        if grid.empty:
            print(f"⚠️ Skipping {grid_path}: no cells found.")
            continue

        # Ensure cell_id exists
        if "cell_id" not in grid.columns:
            grid = grid.reset_index(drop=True)
            grid["cell_id"] = grid.index

        # Determine AOI size
        if "1km" in grid_path:
            aoi_size = "1km"
        elif "20km" in grid_path:
            aoi_size = "20km"
        elif "40km" in grid_path:
            aoi_size = "40km"
        elif "100km" in grid_path:
            aoi_size = "100km"
        else:
            aoi_size = "unknown"

        print(f"\n=== Processing {aoi_size} grid ===")

        # Process grid cells
        results_df = process_grid_cells_with_raster_association(
            grid,
            raster_folder,
            aoi_size,
            output_directory,
            intersection_method='mosaic'  # Change this if needed
        )

        # Print summary statistics
        print(f"\nSummary for {aoi_size}:")
        print(f"Total cells processed: {len(results_df)}")
        print(f"Average canopy extent: {results_df['canopy_extent'].mean():.2f}%")
        print(f"Cells with no intersecting rasters: {sum(results_df['num_intersecting_rasters'] == 0)}")
        print(f"Cells with multiple intersecting rasters: {sum(results_df['num_intersecting_rasters'] > 1)}")