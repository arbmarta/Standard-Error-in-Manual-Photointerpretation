# This is the file that processes the CHM files

# =============================================================================
# SELECT WHICH PROCESSES TO RUN
# =============================================================================
# region

# Run through pre-processing with test settings (low complexity, fast run)
USE_TEST_SETTINGS = False  # Set to True to run a test of the code using a larger BBOX and AOI

# Explore the CHM tiles
EXPLORE_S3_STRUCTURE = False  # Set to True to explore S3 bucket structure
CHECK_CHM_FILE_SIZES = True  # Set to True to analyze CHM file sizes
SHOW_TILES_PLOT = True  # Set to True to visualize the tiles

# Grid generation and mapping
GENERATE_GRIDS = False  # Set to True to generate grids

# Sample point generation and mapping
GENERATE_SAMPLE_POINTS = False  # Set to True to generate systematic sample points
SHOW_SAMPLE_POINTS_MAP = False  # Set to True to show a map of the sample points (run only with test settings)

# Download the CHMs and merge
DOWNLOAD_CHM = False  # Set to True to downlaod the Meta CHM tiles, converted to binary
CREATE_CHM_MOSAIC = False  # Set to True to merge the Meta CHM tiles

# endregion

# =============================================================================
# CONFIGURATION OPTIONS - Modify these as needed
# =============================================================================
# region

# Geographic settings
CONUS_GPKG = "conus.gpkg"  # Path to AOI GeoPackage file
TEST_BBOX = [-80, 40, -70, 45]  # Small NY/New England region for testing

# Grid specifications
GRID_SIZES = [1, 20, 40]  # Neighbourhood (1 km), city (20 km), and region (35 km) grids
TEST_GRID_SIZES = [100]  # Large 100km grids for speed

# Sample point specifications
base_points = 10000  # Number of points per 1 sq km area
SAMPLE_POINTS_CONFIG = {1: base_points,
                        20: base_points,
                        40: base_points,
                        100: base_points}

# S3 data source configuration
BUCKET_NAME = 'dataforgood-fb-data'
TILES_KEY = 'forests/v1/alsgedi_global_v6_float/tiles.geojson'

# CHM download configuration
CHM_OUTPUT_DIR = "chm_binary"  # Output directory
CHM_BINARY_THRESHOLD = 2.0  # Binary threshold value

# endregion

from functions import *


def load_conus_boundary(conus_gpkg_path):
    """
    Load AOI boundary from GeoPackage file

    Parameters:
    -----------
    conus_gpkg_path : str or Path
        Path to the AOI GeoPackage file

    Returns:
    --------
    gpd.GeoDataFrame : AOI boundary geometry
    tuple : Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    try:
        from pathlib import Path
        import geopandas as gpd

        conus_path = Path(conus_gpkg_path)
        if not conus_path.exists():
            print(f"❌ AOI file not found: {conus_path}")
            print("Please ensure the conus.gpkg file is in the same directory as this script")
            return None, None

        print(f"📍 Loading AOI boundary from {conus_path}")
        conus_gdf = gpd.read_file(conus_path)

        # Ensure it's in WGS84 for bbox calculation
        if conus_gdf.crs != 'EPSG:4326':
            conus_gdf = conus_gdf.to_crs('EPSG:4326')

        # Get bounding box
        bounds = conus_gdf.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]  # [min_lon, min_lat, max_lon, max_lat]

        print(f"✅ AOI boundary loaded successfully")
        print(f"   Bounding box: {bbox}")
        print(f"   Number of features: {len(conus_gdf)}")

        return conus_gdf, bbox

    except Exception as e:
        print(f"❌ Error loading AOI boundary: {str(e)}")
        return None, None


def get_active_boundary_and_bbox(use_test_settings, conus_gpkg_path, test_bbox):
    """
    Get the active boundary geometry and bounding box based on settings

    Parameters:
    -----------
    use_test_settings : bool
        Whether to use test settings
    conus_gpkg_path : str
        Path to AOI GeoPackage file
    test_bbox : list
        Test bounding box [min_lon, min_lat, max_lon, max_lat]

    Returns:
    --------
    gpd.GeoDataFrame or None : Boundary geometry (None for test bbox)
    list : Bounding box [min_lon, min_lat, max_lon, max_lat]
    str : Description of the boundary being used
    """
    if use_test_settings:
        print("🧪 Using test settings with simplified bounding box")
        return None, test_bbox, f"Test region: {test_bbox}"
    else:
        conus_gdf, conus_bbox = load_conus_boundary(conus_gpkg_path)
        if conus_gdf is not None and conus_bbox is not None:
            return conus_gdf, conus_bbox, f"AOI boundary from {conus_gpkg_path}"
        else:
            print("⚠️  Failed to load AOI boundary, falling back to test bbox")
            return None, test_bbox, f"Fallback test region: {test_bbox}"


if __name__ == "__main__":

    # Determine which settings to use
    boundary_gdf, active_bbox, boundary_description = get_active_boundary_and_bbox(
        USE_TEST_SETTINGS,
        CONUS_GPKG,
        TEST_BBOX
    )
    active_grid_sizes = TEST_GRID_SIZES if USE_TEST_SETTINGS else GRID_SIZES
    mode = "TESTING" if USE_TEST_SETTINGS else "PRODUCTION"

    print("Access and Process Meta CHM Script")
    print("=" * 50)
    print(f"Target area: {boundary_description}")
    print(f"Bounding box: {active_bbox}")
    print(f"Configuration:")
    print(f"  - Test run: {USE_TEST_SETTINGS}")
    print(f"  - Explore S3: {EXPLORE_S3_STRUCTURE}")
    print(f"  - Show tiles plot: {SHOW_TILES_PLOT}")
    print(f"  - Generate grids: {GENERATE_GRIDS}")
    print(f"  - Generate sample points: {GENERATE_SAMPLE_POINTS}")
    print(f"  - Download CHM: {DOWNLOAD_CHM}")
    print(f"  - Merge CHM: {CREATE_CHM_MOSAIC}")
    print(f"  - Grid sizes: {active_grid_sizes} km")
    print("=" * 50)
    print("\n")

    success = True
    tiles_gdf = None

    # Step 1: Optional S3 exploration
    if EXPLORE_S3_STRUCTURE:
        print("\n🔍 EXPLORING S3 BUCKET STRUCTURE...")
        list_s3_directories(BUCKET_NAME, 'forests/v1/alsgedi_global_v6_float/')
        print("\n" + "=" * 60 + "\n")

    # Step 2: Load tiles data (needed for grids, plots, or analysis)
    if GENERATE_GRIDS or SHOW_TILES_PLOT or DOWNLOAD_CHM or GENERATE_SAMPLE_POINTS or (
            not GENERATE_GRIDS and not EXPLORE_S3_STRUCTURE and not SHOW_TILES_PLOT):
        print("📥 LOADING TILES DATA...")

        # Pass boundary information to the download function
        tiles_gdf = download_tiles_geojson(
            BUCKET_NAME,
            TILES_KEY,
            bbox=active_bbox,
            boundary_gdf=boundary_gdf,  # Pass the boundary geometry if available
            show_plot=SHOW_TILES_PLOT
        )

        if tiles_gdf is None:
            print("❌ Failed to load tiles data. Exiting.")
            success = False
        else:
            print(f"\n✅ Tiles loaded successfully")
            print(f"   Tiles extent: {tiles_gdf.total_bounds}")
            print(f"   Number of tiles: {len(tiles_gdf)}")

    # Step 3: Optional CHM file size check
    if CHECK_CHM_FILE_SIZES and tiles_gdf is not None:
        print(f"\n📊 CHECKING CHM FILE SIZES...")
        from functions import check_chm_file_sizes

        file_stats = check_chm_file_sizes(
            tiles_gdf,
            BUCKET_NAME,
            sample_size=2942
        )

    # Step 4: Generate grids if requested
    if GENERATE_GRIDS and success and tiles_gdf is not None:
        print(f"\n🔲 GENERATING GRIDS...")

        # Create config for grid generation (now includes boundary info)
        grid_config = {
            'bbox': active_bbox,
            'boundary_gdf': boundary_gdf,
            'grid_sizes': active_grid_sizes,
            'generate_grids': True  # Force True for this step
        }

        from functions import create_grid, spatial_filter_grid, save_grid, _get_filename_from_km

        successful_grids = []
        bounds = tiles_gdf.total_bounds

        for cell_size_km in active_grid_sizes:
            # Convert km to meters for internal calculations
            cell_size_meters = int(cell_size_km * 1000)
            grid_label = f"{cell_size_km}km"

            print(f"\n{'🔲 PROCESSING ' + grid_label + ' GRID':=^60}")

            try:
                # Create grid
                print(f"⚙️  Creating {grid_label} grid...")
                grid = create_grid(bounds, cell_size_meters, crs=tiles_gdf.crs)

                # Spatial filter (use boundary if available, otherwise use tiles)
                print(f"🔍 Spatially filtering {grid_label} grid...")
                if boundary_gdf is not None:
                    # Filter using AOI boundary for more precise results
                    boundary_proj = boundary_gdf.to_crs(tiles_gdf.crs)
                    filtered_grid = spatial_filter_grid(grid, boundary_proj)
                else:
                    # Filter using tiles as before
                    filtered_grid = spatial_filter_grid(grid, tiles_gdf)

                # Save grid
                print(f"💾 Saving {grid_label} grid...")
                filename = _get_filename_from_km(cell_size_km)
                save_grid(filtered_grid, filename, cell_size_meters)

                successful_grids.append(filename)
                print(f"✅ {grid_label} grid completed successfully!")

                # Clear memory
                del grid, filtered_grid

            except Exception as e:
                print(f"❌ Error processing {grid_label} grid: {e}")
                success = False
                continue

        # Grid generation summary
        if successful_grids:
            print(f"\n{'🎉 GRID GENERATION COMPLETE':=^60}")
            print("✅ Successfully created grids:")
            for filename in successful_grids:
                print(f"   📄 {filename}")
        else:
            print("❌ No grids were created successfully")
            success = False

    # Step 5: Run analysis if grids are disabled
    if not GENERATE_GRIDS:
        # If we haven't loaded tiles yet (when all flags are False), load them now for analysis
        if tiles_gdf is None:
            print("📥 LOADING TILES DATA FOR ANALYSIS...")
            tiles_gdf = download_tiles_geojson(
                BUCKET_NAME,
                TILES_KEY,
                bbox=active_bbox,
                boundary_gdf=boundary_gdf,
                show_plot=False  # Don't show plot here since SHOW_TILES_PLOT was False
            )

        if tiles_gdf is not None:
            print(f"\n🔬 RUNNING ANALYSIS...")
            config = {
                'bbox': active_bbox,
                'boundary_gdf': boundary_gdf,
                'grid_sizes': active_grid_sizes,
                'generate_grids': GENERATE_GRIDS
            }
            analysis_success = placeholder_analysis(tiles_gdf, config)
            success = success and analysis_success
        else:
            print("❌ Could not load tiles for analysis")
            success = False

    # Step 6: Generate sample points if requested
    if GENERATE_SAMPLE_POINTS:
        print(f"\n🎯 GENERATING SAMPLE POINTS...")

        # Only process grids that exist for current mode
        active_sample_config = {size: SAMPLE_POINTS_CONFIG[size]
                                for size in active_grid_sizes
                                if size in SAMPLE_POINTS_CONFIG}

        if active_sample_config:
            # Generate sample points for all active grid sizes
            # Pass boundary info to the sample generation function
            sample_results = generate_sample_points_for_grids(
                active_sample_config,
                active_bbox,
                boundary_gdf=boundary_gdf
            )

            # Optional: Show sample points map
            if SHOW_SAMPLE_POINTS_MAP and sample_results:
                print(f"\n🗺️  CREATING SAMPLE POINTS MAP...")
                from functions import plot_sample_points_map

                for grid_size_km, points_gdf in sample_results.items():
                    if points_gdf is not None and len(points_gdf) > 0:
                        # Use the new function that plots all points
                        plot_sample_points_map(
                            points_gdf,
                            tiles_gdf=tiles_gdf,
                            bbox=active_bbox,
                            boundary_gdf=boundary_gdf,  # Pass boundary for better visualization
                            grid_size_km=grid_size_km
                        )
        else:
            print("   No sample point configuration found for active grid sizes")

    # Step 7: Download and process CHM tiles if requested
    if DOWNLOAD_CHM and tiles_gdf is not None:
        print(f"\n🌲 DOWNLOADING CHM TILES...")

        from functions import download_chm, create_chm_mosaic

        # Download and process CHM tiles
        chm_results = download_chm(
            tiles_gdf,
            BUCKET_NAME,
            output_dir=CHM_OUTPUT_DIR,
            binary_threshold=CHM_BINARY_THRESHOLD
        )

        if chm_results and chm_results['processed'] > 0:
            print(f"✅ CHM processing completed: {chm_results['processed']} tiles processed")

            # Optionally create mosaic
            if CREATE_CHM_MOSAIC:
                print(f"\n🗺️  CREATING CHM MOSAIC...")
                mosaic_path = f"{CHM_OUTPUT_DIR}/chm_binary_mosaic.tif"
                mosaic_result = create_chm_mosaic(CHM_OUTPUT_DIR, mosaic_path, tiles_gdf)

                if mosaic_result:
                    print(f"✅ CHM mosaic created: {mosaic_result}")
                else:
                    print("❌ Failed to create CHM mosaic")
        else:
            print("❌ CHM processing failed or no tiles were processed")
            success = False

    # Final summary
    print(f"\n{'📋 EXECUTION SUMMARY':=^60}")
    operations = []
    if EXPLORE_S3_STRUCTURE:
        operations.append("✅ S3 exploration")
    if SHOW_TILES_PLOT:
        operations.append("✅ Tiles visualization")
    if GENERATE_GRIDS:
        operations.append("✅ Grid generation")
    if not GENERATE_GRIDS and tiles_gdf is not None:
        operations.append("✅ Analysis placeholder")
    if DOWNLOAD_CHM:
        operations.append("✅ CHM tile download and processing")

    if operations:
        print("Completed operations:")
        for op in operations:
            print(f"   {op}")
    else:
        print("No operations were configured to run")

    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")

    # Exit with appropriate code
    exit(0 if success else 1)