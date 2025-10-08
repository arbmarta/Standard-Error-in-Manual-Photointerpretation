from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import queue
import rasterio
import rasterio.mask
from scipy import ndimage
from scipy.spatial.distance import pdist
from shapely.geometry import Point
from skimage.measure import label, regionprops
import time
from tqdm import tqdm
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------- Functions for download meta.py -------------------------------------------
#region

def main_download_workflow(quadkeys, raw_dir="Meta CHM Raw", binary_dir="Meta CHM Binary"):
    """
    Main workflow that validates files in S3, then downloads and converts them in parallel.
    Raw downloaded files are deleted after successful conversion.
    """

    # Create S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

    bucket_name = 'dataforgood-fb-data'
    base_prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

    # --- Step 1: Validate that files exist in S3 ---
    print("Step 1: Validating file existence in S3...")
    valid_paths, invalid_paths, file_info = validate_quadkey_paths(
        quadkeys, bucket_name, base_prefix, s3
    )

    print(f"✓ Found {len(valid_paths)} valid files")
    if invalid_paths:
        print(f"✗ {len(invalid_paths)} files not found at expected locations")
        # (Optional: Add logic here to find alternatives for missing files if needed)

    if not valid_paths:
        print("❌ No valid files found to download!")
        return

    # --- Step 2: Check what's already downloaded locally ---
    print(f"\nStep 2: Checking local files...")
    files_to_download = []
    already_converted = 0

    # We check the FINAL destination (binary directory) to see if work is already done
    for s3_path in valid_paths:
        key = s3_path.replace(f"s3://{bucket_name}/", "")
        quadkey = file_info[key]['quadkey']

        # Check if the FINAL binary file already exists and is valid
        final_binary_path = Path(binary_dir) / f"{quadkey}.tif"
        if final_binary_path.exists() and final_binary_path.stat().st_size > 1024:  # Basic integrity check
            already_converted += 1
        else:
            files_to_download.append((s3_path, key))

    print(f"✓ {already_converted} files already converted and exist in '{binary_dir}'")
    print(f"📥 {len(files_to_download)} files need downloading and conversion")

    if not files_to_download:
        print("✅ All files are already processed!")
        return

    # --- Step 3: Calculate total download size ---
    total_size_mb = sum(file_info[key]['size_mb'] for _, key in files_to_download)

    print(f"\nStep 3: Download summary")
    print(f"Total download size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.1f} GB)")

    # --- Step 4: Setup directories and parallel processing ---
    print("\nStep 4: Starting parallel download and conversion...")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(binary_dir).mkdir(parents=True, exist_ok=True)

    conversion_queue = queue.Queue()
    HEIGHT_THRESHOLD = 5  # meters

    # Define number of parallel workers for each task
    num_download_workers = 5
    num_conversion_workers = max(1, os.cpu_count() - 1)  # Use most CPU cores for conversion

    successful_downloads = 0
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=num_conversion_workers, thread_name_prefix='Converter') as conversion_executor, \
            ThreadPoolExecutor(max_workers=num_download_workers, thread_name_prefix='Downloader') as download_executor:

        # 1. Start the conversion workers. They will wait for items in the queue.
        for _ in range(num_conversion_workers):
            conversion_executor.submit(conversion_worker, conversion_queue, binary_dir, HEIGHT_THRESHOLD)

        # 2. Submit all download tasks
        future_to_file = {}
        for s3_path, key in files_to_download:
            quadkey = Path(key).stem
            local_path = Path(raw_dir) / f"{quadkey}.tif"
            future = download_executor.submit(
                download_tile_boto3, s3, bucket_name, key, str(local_path)
            )
            future_to_file[future] = local_path

        # 3. Process downloads as they complete
        with tqdm(total=len(files_to_download), desc="Downloading & Converting") as pbar:
            for future in as_completed(future_to_file):
                local_path = future_to_file[future]
                success, message = future.result()

                if success:
                    successful_downloads += 1
                    # Add the successfully downloaded file path to the queue for conversion
                    conversion_queue.put(str(local_path))
                else:
                    failed_downloads += 1
                    logger.error(f"Download failed: {message}")

                pbar.update(1)

        # 4. All downloads are done. Wait for the conversion queue to be empty.
        print("\nAll downloads complete. Waiting for conversions to finish...")
        conversion_queue.join()

        # 5. Signal the conversion workers to stop by sending 'None'
        for _ in range(num_conversion_workers):
            conversion_queue.put(None)

    # --- Step 5: Final summary ---
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Binary rasters saved to: {Path(binary_dir).absolute()}")
    print(f"🧹 Raw files directory '{Path(raw_dir).absolute()}' should now be empty.")

    if failed_downloads > 0:
        print(f"\n⚠ {failed_downloads} downloads failed. Check logs for details.")

#endregion

# ------------------------------------------ Functions for canopy metrics.py -------------------------------------------
#region

def process_grid_cells_with_raster_association(grid_gdf, raster_folder, aoi_size, output_dir='.',
                                               intersection_method='mosaic'):
    """
    Process grid cells and calculate landscape metrics for each, handling multiple intersecting rasters.

    Parameters:
    -----------
    grid_gdf : geopandas.GeoDataFrame
        Grid with cell_id column and geometry
    raster_folder : str
        Path to folder containing raster files
    aoi_size : str
        AOI size identifier for CSV naming
    output_dir : str
        Output directory for CSV files
    intersection_method : str
        Method to handle multiple intersecting rasters:
        - 'mosaic': Mosaic all intersecting rasters
        - 'largest': Use the raster with largest intersection area
        - 'first': Use the first intersecting raster found

    Returns:
    --------
    pandas.DataFrame : Results dataframe
    """
    results_list = []

    print(f"Processing {len(grid_gdf)} grid cells for AOI size: {aoi_size}")

    # Build spatial index for efficient raster lookup
    print("Building spatial index for raster files...")
    spatial_index, raster_info = build_raster_spatial_index(raster_folder)

    if spatial_index is not None:
        print(f"Using spatial index with {len(raster_info)} rasters")
    else:
        print("Using brute force intersection checking")

    for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc=f"Processing {aoi_size}"):
        cell_id = row['cell_id']
        geometry = row.geometry

        try:
            # Find intersecting rasters using spatial index
            if spatial_index is not None:
                intersecting_rasters = get_intersecting_rasters_indexed(
                    geometry, spatial_index, raster_info
                )
            else:
                intersecting_rasters = get_intersecting_rasters_bruteforce(
                    geometry, raster_folder
                )

            if not intersecting_rasters:
                print(f"No intersecting rasters found for cell_id {cell_id}")
                # Add a record with zero values
                metrics = {
                    'canopy_extent': 0,
                    'morans_i': 0,
                    'join_count_bb': 0,
                    'hansens_uniformity': 0.5,
                    'geary_c': 1.0,
                    'edge_density': 0,
                    'clumpy': 0,
                    'number_of_patches': 0,
                    'avg_patch_size': 0,
                    'patch_size_std': 0,
                    'patch_size_median': 0,
                    'patch_size_min': 0,
                    'patch_size_max': 0,
                    'normalized_lsi': 0,
                    'landscape_type': 'no_data',
                    'cell_id': cell_id,
                    'num_intersecting_rasters': 0
                }
                results_list.append(metrics)
                continue

            # Handle intersection method
            if intersection_method == 'mosaic' or len(intersecting_rasters) > 1:
                # Mosaic intersecting rasters
                raster_data, transform = mosaic_rasters(
                    intersecting_rasters,
                    geometry,
                    grid_gdf.crs
                )
            else:
                # Single raster case
                with rasterio.open(intersecting_rasters[0]) as src:
                    out_image, transform = rasterio.mask.mask(
                        src, [geometry], crop=True, all_touched=False
                    )
                    raster_data = out_image[0]

            # Skip if no valid data
            if raster_data is None or raster_data.size == 0:
                print(f"No valid raster data for cell_id {cell_id}")
                continue

            # Calculate cell size from transform
            cell_size = abs(transform.a)  # Pixel width

            # Calculate metrics
            metrics = calculate_landscape_metrics(raster_data, cell_size=cell_size)
            metrics['cell_id'] = cell_id
            metrics['num_intersecting_rasters'] = len(intersecting_rasters)

            results_list.append(metrics)

        except Exception as e:
            print(f"Error processing cell_id {cell_id}: {e}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_filename = f"canopy_metrics_{aoi_size}.csv"
    output_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results_df)} grid cells successfully")

    return results_df

#endregion

# ----------------------------------------- Functions for sample points.py ------------------------------------------
#region

def generate_systematic_sample_points(grid_gdf, points_per_cell, raster_folder=None):
    """Generate systematic sample points within each grid cell."""
    print(f"🎯 GENERATING {points_per_cell:,} SYSTEMATIC SAMPLE POINTS PER GRID CELL...")
    print(f"   Processing {len(grid_gdf):,} grid cells...")

    # Calculate grid layout for points (as square as possible)
    n_points = points_per_cell
    cols = int(math.ceil(math.sqrt(n_points)))
    rows = int(math.ceil(n_points / cols))
    print(f"   Point pattern: {rows} rows × {cols} columns")

    x_positions = np.linspace(0, 1, cols)
    y_positions = np.linspace(0, 1, rows)

    relative_points = []
    point_count = 0
    for y in y_positions:
        for x in x_positions:
            if point_count < n_points:
                relative_points.append((x, y))
                point_count += 1

    print(f"   Using {len(relative_points)} points per cell with consistent positioning")

    # Build spatial index if raster folder provided
    spatial_index, raster_info = None, None
    if raster_folder:
        print(f"\n   Building spatial index for raster validation...")
        spatial_index, raster_info = build_raster_spatial_index(raster_folder)

    all_points, all_grid_ids, all_point_ids = [], [], []
    all_has_raster = []

    for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="Generating points"):
        grid_cell = row.geometry
        grid_id = row['grid_id']

        # Check if grid cell has intersecting rasters
        has_raster = False
        if spatial_index is not None:
            intersecting = get_intersecting_rasters_indexed(grid_cell, spatial_index, raster_info)
            has_raster = len(intersecting) > 0

        minx, miny, maxx, maxy = grid_cell.bounds
        cell_width, cell_height = maxx - minx, maxy - miny

        for point_idx, (rel_x, rel_y) in enumerate(relative_points):
            abs_x = minx + (rel_x * cell_width)
            abs_y = miny + (rel_y * cell_height)
            point = Point(abs_x, abs_y)

            if grid_cell.contains(point) or grid_cell.intersects(point):
                all_points.append(point)
                all_grid_ids.append(grid_id)
                all_point_ids.append(f"{grid_id}_p{point_idx}")
                all_has_raster.append(has_raster)

    points_gdf = gpd.GeoDataFrame({
        'point_id': all_point_ids,
        'grid_id': all_grid_ids,
        'sample_type': 'systematic',
        'points_per_cell': points_per_cell,
        'has_raster_data': all_has_raster
    }, geometry=all_points, crs=grid_gdf.crs)

    print(f"✅ Generated {len(points_gdf):,} total sample points")
    print(f"   Points per cell: {len(points_gdf) / len(grid_gdf):.1f} (target: {points_per_cell})")
    if spatial_index is not None:
        points_with_data = sum(all_has_raster)
        print(f"   Points with raster data: {points_with_data:,} ({100 * points_with_data / len(points_gdf):.1f}%)")

    return points_gdf


def plot_sample_points_map(points_gdf, tiles_gdf=None, bbox=None, grid_size_km=None):
    """Create sample points map visualization."""
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    print(f"   Creating sample points map for {grid_size_km}km grid...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    if tiles_gdf is not None:
        tiles_gdf.plot(ax=ax, facecolor='lightgreen', edgecolor='darkgreen',
                       alpha=0.2, linewidth=0.3)

    # Color points by whether they have raster data
    if 'has_raster_data' in points_gdf.columns:
        points_with_data = points_gdf[points_gdf['has_raster_data']]
        points_without_data = points_gdf[~points_gdf['has_raster_data']]

        if len(points_with_data) > 0:
            points_with_data.plot(ax=ax, color='red', markersize=2, alpha=0.7, label='With data')
        if len(points_without_data) > 0:
            points_without_data.plot(ax=ax, color='gray', markersize=1, alpha=0.3, label='No data')

        print(f"   Points with data: {len(points_with_data):,}")
        print(f"   Points without data: {len(points_without_data):,}")
    else:
        points_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.7)

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                         linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(min_lon - 1, max_lon + 1)
        ax.set_ylim(min_lat - 0.5, max_lat + 0.5)

    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    title = f'Sample Points Distribution'
    if grid_size_km is not None:
        title += f' - {grid_size_km}km Grid'
    title += f'\n{len(points_gdf):,} total points'
    ax.set_title(title, fontsize=16, pad=20)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen',
               markersize=10, alpha=0.2, label='Meta CHM Tiles'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=5, alpha=0.7, label='Sample Points (with data)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=5, alpha=0.3, label='Sample Points (no data)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='AOI Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#endregion

# -------------------------------------------------- Helper Functions --------------------------------------------------

# S3 AND BOTO
#region
def check_s3_file_exists(s3_client, bucket, key):
    """
    Check if a file exists in S3 bucket without downloading it
    Returns: (exists: bool, size: int, last_modified: datetime or None)
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return True, response['ContentLength'], response['LastModified']
    except s3_client.exceptions.NoSuchKey:
        return False, 0, None
    except Exception as e:
        logger.warning(f"Error checking S3 file {key}: {str(e)}")
        return False, 0, None


def batch_check_s3_files(s3_client, bucket, keys, max_workers=10):
    """
    Check existence of multiple S3 files in parallel
    Returns: dict mapping key -> (exists, size, last_modified)
    """
    results = {}

    def check_single_file(key):
        exists, size, modified = check_s3_file_exists(s3_client, bucket, key)
        return key, (exists, size, modified)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_single_file, key): key for key in keys}

        for future in tqdm(as_completed(futures), total=len(keys), desc="Checking S3 files"):
            key, result = future.result()
            results[key] = result

    return results


def validate_quadkey_paths(quadkeys, bucket_name, base_prefix, s3_client):
    """
    Validate that the expected S3 paths exist for given quadkeys
    Returns: (valid_paths, invalid_paths, file_info)
    """
    # Generate expected S3 keys
    expected_keys = []
    for qk in quadkeys:
        # Meta CHM files are typically stored as: chm/{quadkey}.tif
        key = f"{base_prefix}{qk}.tif"
        expected_keys.append(key)

    print(f"Checking existence of {len(expected_keys)} files in S3...")

    # Batch check all files
    file_check_results = batch_check_s3_files(s3_client, bucket_name, expected_keys)

    # Separate valid and invalid paths
    valid_paths = []
    invalid_paths = []
    file_info = {}

    for key, (exists, size, modified) in file_check_results.items():
        if exists:
            valid_paths.append(f"s3://{bucket_name}/{key}")
            file_info[key] = {
                'size_mb': size / (1024 * 1024),
                'last_modified': modified,
                'quadkey': Path(key).stem
            }
        else:
            invalid_paths.append(f"s3://{bucket_name}/{key}")

    return valid_paths, invalid_paths, file_info


def download_tile_boto3(s3_client, bucket, key, local_path, max_retries=3):
    """
    Download a single tile using boto3 with pre-download existence check
    """
    # First, verify the file exists and get its info
    exists, size, modified = check_s3_file_exists(s3_client, bucket, key)
    if not exists:
        return False, f"✗ {Path(local_path).name}: File does not exist in S3"

    # Check if local file already exists and is complete
    if Path(local_path).exists():
        local_size = Path(local_path).stat().st_size
        if local_size == size:
            return True, f"✓ {Path(local_path).name} (already complete)"
        else:
            logger.warning(f"Local file size mismatch for {Path(local_path).name}: {local_size} vs {size}")
            # Remove incomplete file
            Path(local_path).unlink(missing_ok=True)

    # Create directory if it doesn't exist
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Download with retries
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket, key, local_path)

            # Verify download completed successfully
            if check_file_integrity(local_path, size):
                return True, f"✓ {Path(local_path).name} ({size / (1024 * 1024):.1f} MB)"
            else:
                return False, f"✗ {Path(local_path).name}: Downloaded file failed integrity check"

        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"✗ {Path(local_path).name}: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

    return False, f"✗ {Path(local_path).name}: Max retries exceeded"


def check_file_integrity(file_path, expected_size=None, min_size_mb=0.1):
    """
    Enhanced file integrity check with optional size validation
    """
    try:
        if not Path(file_path).exists():
            return False

        actual_size = Path(file_path).stat().st_size
        size_mb = actual_size / (1024 * 1024)

        # Check minimum size
        if size_mb < min_size_mb:
            logger.warning(f"File {file_path} is too small: {size_mb:.2f} MB")
            return False

        # Check expected size if provided
        if expected_size is not None and actual_size != expected_size:
            logger.warning(f"File {file_path} size mismatch: {actual_size} vs expected {expected_size}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking file integrity for {file_path}: {str(e)}")
        return False


def reproject_raster(input_path, output_path, target_crs='EPSG:5070'):
    """Reprojects a raster to the target CRS using Nearest Neighbor resampling."""
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    try:
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'lzw' # Good practice to add compression
            })

            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest # Crucial for binary data
                )
        return True, f"Reprojected to {output_path}"
    except Exception as e:
        logger.error(f"Failed to reproject {input_path}: {e}")
        return False, str(e)

#endregion

# RASTER PROCESSING
#region
def build_raster_spatial_index(raster_folder):
    """
    Build a spatial index of all raster files for efficient intersection queries.

    Parameters:
    -----------
    raster_folder : str
        Path to folder containing raster files

    Returns:
    --------
    tuple : (rtree_index, raster_bounds_dict)
        - rtree_index: R-tree spatial index
        - raster_bounds_dict: Dictionary mapping index IDs to raster info
    """
    try:
        from rtree import index
    except ImportError:
        print("Warning: rtree not available. Install with: pip install rtree")
        print("Falling back to brute force intersection checking...")
        return None, None

    # Create R-tree index
    idx = index.Index()
    raster_info = {}

    # Get all GeoTIFF files
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"Building spatial index for {len(raster_files)} raster files...")

    for i, raster_path in enumerate(tqdm(raster_files, desc="Indexing rasters")):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                # Insert bounding box into spatial index
                idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top))

                raster_crs = src.crs
                print(f"The CRS of the raster is: {raster_crs}")

                # Store raster information
                raster_info[i] = {
                    'path': raster_path,
                    'bounds': bounds,
                    'crs': src.crs,
                    'transform': src.transform,
                    'shape': src.shape
                }

        except Exception as e:
            print(f"Warning: Could not index {raster_path}: {e}")
            continue

    print(f"Spatial index built with {len(raster_info)} rasters")
    return idx, raster_info


def get_intersecting_rasters_indexed(grid_cell_geometry, spatial_index, raster_info, buffer_distance=0):
    """
    Find all raster files that intersect with a given grid cell geometry using spatial index.

    Parameters:
    -----------
    grid_cell_geometry : shapely.geometry
        The geometry of the grid cell
    spatial_index : rtree.index.Index
        R-tree spatial index of raster bounds
    raster_info : dict
        Dictionary mapping index IDs to raster information
    buffer_distance : float
        Buffer distance to expand grid cell for intersection (optional)

    Returns:
    --------
    list : List of raster file paths that intersect the grid cell
    """
    if spatial_index is None:
        # Fallback to brute force method
        return get_intersecting_rasters_bruteforce(grid_cell_geometry, raster_info, buffer_distance)

    intersecting_rasters = []

    # Buffer the grid cell geometry if specified
    if buffer_distance > 0:
        search_geometry = grid_cell_geometry.buffer(buffer_distance)
    else:
        search_geometry = grid_cell_geometry

    # Get bounding box of search geometry
    minx, miny, maxx, maxy = search_geometry.bounds

    # Query spatial index for potentially intersecting rasters
    candidate_ids = list(spatial_index.intersection((minx, miny, maxx, maxy)))

    # Perform detailed intersection check for candidates
    for raster_id in candidate_ids:
        try:
            raster_bounds = raster_info[raster_id]['bounds']

            # Create raster geometry
            from shapely.geometry import box
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)

            # Check detailed intersection
            if search_geometry.intersects(raster_geom):
                intersecting_rasters.append(raster_info[raster_id]['path'])

        except Exception as e:
            print(f"Warning: Error checking intersection for raster {raster_id}: {e}")
            continue

    return intersecting_rasters


def get_intersecting_rasters_bruteforce(grid_cell_geometry, raster_folder_or_info, buffer_distance=0):
    """
    Fallback brute force method for finding intersecting rasters.
    """
    intersecting_rasters = []

    # Buffer the grid cell geometry if specified
    if buffer_distance > 0:
        search_geometry = grid_cell_geometry.buffer(buffer_distance)
    else:
        search_geometry = grid_cell_geometry

    # Handle different input types
    if isinstance(raster_folder_or_info, dict):
        # Called from indexed method as fallback
        raster_files = [info['path'] for info in raster_folder_or_info.values()]
    else:
        # Called directly with folder path
        raster_pattern = os.path.join(raster_folder_or_info, "*.tif*")
        raster_files = glob.glob(raster_pattern)

    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                # Get raster bounds and create geometry
                bounds = src.bounds
                from shapely.geometry import box
                raster_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Check intersection
                if search_geometry.intersects(raster_geom):
                    intersecting_rasters.append(raster_path)

        except Exception as e:
            print(f"Warning: Could not check intersection for {raster_path}: {e}")
            continue

    return intersecting_rasters


def create_binary_raster(input_path, output_path, threshold, nodata_value=None):
    """
    Converts a CHM raster to a binary raster based on a height threshold.
    """
    try:
        with rasterio.open(input_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            source_nodata = src.nodata

            # Pixels > threshold become 1, all others become 0
            binary_data = (data > threshold).astype(np.uint8)

            # Preserve the original NoData values
            if source_nodata is not None:
                binary_data[data == source_nodata] = source_nodata

            meta.update(dtype='uint8', count=1, compress='lzw')
            if nodata_value is not None:
                meta['nodata'] = nodata_value

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(binary_data, 1)

        return True, f"Created binary raster: {Path(output_path).name}"
    except Exception as e:
        logger.error(f"Failed to create binary raster for {input_path}: {e}")
        return False, str(e)


def conversion_worker(q, binary_dir, reprojected_dir, threshold):
    """
    Pulls a file, converts to binary, reprojects it, and deletes the raw file.
    """
    Path(binary_dir).mkdir(parents=True, exist_ok=True)
    Path(reprojected_dir).mkdir(parents=True, exist_ok=True)

    while True:
        try:
            raw_path_str = q.get()
            if raw_path_str is None:
                break

            raw_path = Path(raw_path_str)
            binary_path = Path(binary_dir) / raw_path.name
            reprojected_path = Path(reprojected_dir) / raw_path.name

            # Step 1: Create binary raster
            success_bin, msg_bin = create_binary_raster(str(raw_path), str(binary_path), threshold)

            if success_bin:
                # Step 2: Reproject the new binary raster
                success_reproj, msg_reproj = reproject_raster(str(binary_path), str(reprojected_path))

                if success_reproj:
                    # Clean up intermediate binary file if desired
                    os.remove(binary_path)

                    # Step 3: Always remove the raw file after processing
            os.remove(raw_path)

        except Exception as e:
            logger.error(f"Error in conversion worker for {raw_path_str}: {e}")
        finally:
            q.task_done()


def mosaic_rasters(raster_paths, target_geometry, target_crs, target_resolution=None):
    """
    Mosaic multiple rasters that intersect a target geometry.

    Parameters:
    -----------
    raster_paths : list
        List of raster file paths to mosaic
    target_geometry : shapely.geometry
        Target geometry to clip the mosaic to
    target_crs : CRS
        Target coordinate reference system
    target_resolution : float, optional
        Target pixel resolution. If None, uses the finest resolution from input rasters

    Returns:
    --------
    tuple : (mosaicked_array, transform)
    """
    if not raster_paths:
        return None, None

    if len(raster_paths) == 1:
        # Single raster case
        with rasterio.open(raster_paths[0]) as src:
            out_image, out_transform = rasterio.mask.mask(
                src, [target_geometry], crop=True, all_touched=False
            )
            return out_image[0], out_transform

    # Multiple rasters - need to mosaic
    try:
        from rasterio.merge import merge

        # Open all rasters
        src_files = [rasterio.open(path) for path in raster_paths]

        # Determine target resolution
        if target_resolution is None:
            target_resolution = min([abs(src.transform.a) for src in src_files])

        # Merge rasters
        mosaic, out_trans = merge(
            src_files,
            res=target_resolution,
            method='max'  # Use max value where rasters overlap
        )

        # Close source files
        for src in src_files:
            src.close()

        # Create temporary in-memory dataset for clipping
        from rasterio.io import MemoryFile

        with MemoryFile() as memfile:
            with memfile.open(
                    driver='GTiff',
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    count=1,
                    dtype=mosaic.dtype,
                    crs=src_files[0].crs,  # Assuming all rasters have same CRS
                    transform=out_trans
            ) as dataset:
                dataset.write(mosaic[0], 1)

                # Clip to target geometry
                out_image, out_transform = rasterio.mask.mask(
                    dataset, [target_geometry], crop=True, all_touched=False
                )

                return out_image[0], out_transform

    except Exception as e:
        print(f"Error in mosaicking: {e}")
        return None, None


#endregion

# LANDSCAPE METRICS
#region
def calculate_landscape_metrics(raster_data, cell_size=None):
    """
    Calculate comprehensive landscape metrics for a 1-bit binary raster dataset.

    METRICS CALCULATED:
    ==================

    LANDSCAPE-LEVEL METRICS:
    -----------------------
    1. canopy_extent: Percentage of landscape covered by canopy (0-100%)
    2. edge_density: Total edge length per unit area (length/area units)
    3. clumpy: Aggregation index (0-1)
    4. landscape_type: Categorical description ('mixed', 'all_canopy', 'all_non_canopy', 'no_data')

    SPATIAL AUTOCORRELATION METRICS:
    ---------------------------------------
    5. morans_i: Moran's I (NOT recommended for binary data - kept for comparison)
    6. join_count_bb: Join Count BB statistic (RECOMMENDED for binary data)
    7. hansens_uniformity: Hansen's Uniformity Index (0-1)
    8. geary_c: Geary's C (alternative to Moran's I for binary data)

    PATCH-LEVEL METRICS:
    -------------------
    9. number_of_patches: Count of discrete canopy patches
    10. avg_patch_size: Mean patch area
    11. patch_size_std: Standard deviation of patch areas
    12. patch_size_median: Median patch area
    13. patch_size_min: Smallest patch area
    14. patch_size_max: Largest patch area
    15. normalized_lsi: Normalized Landscape Shape Index (whole grid cell)

    Parameters:
    -----------
    raster_data : numpy.ndarray
        2D binary array (0=non-canopy, 1=canopy) from 1-bit raster
    cell_size : float, optional
        Size of each raster cell in appropriate units (e.g., meters)
        If None, calculations will be in pixel units

    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """
    # Handle invalid data
    if raster_data is None or raster_data.size == 0:
        return {
            'canopy_extent': 0,
            'morans_i': 0,
            'join_count_bb': 0,
            'hansens_uniformity': 0.5,
            'geary_c': 1.0,
            'edge_density': 0,
            'clumpy': 0,
            'number_of_patches': 0,
            'avg_patch_size': 0,
            'patch_size_std': 0,
            'patch_size_median': 0,
            'patch_size_min': 0,
            'patch_size_max': 0,
            'normalized_lsi': 0,
            'landscape_type': 'no_data'
        }

    # Data is already binary (1-bit rasters can only be 0 or 1)
    binary_raster = raster_data.astype(int)

    # Handle cell size
    if cell_size is None:
        cell_size = 1.0
        area_multiplier = 1.0
    else:
        area_multiplier = cell_size ** 2

    results = {}

    # Check for uniform landscapes (all 0s or all 1s)
    unique_values = np.unique(binary_raster)
    total_cells = binary_raster.size
    canopy_cells = np.sum(binary_raster)

    # 1. CANOPY EXTENT
    results['canopy_extent'] = (canopy_cells / total_cells) * 100  # Percentage

    # Handle special cases for uniform landscapes
    if len(unique_values) == 1:
        if unique_values[0] == 0:
            # All zeros - no canopy
            results.update({
                'morans_i': 0,  # Not meaningful for uniform data
                'join_count_bb': 0,  # No BB joins possible
                'hansens_uniformity': 1.0,  # Perfect uniformity (all same)
                'geary_c': 1.0,  # No spatial autocorrelation possible
                'edge_density': 0,  # No edges
                'clumpy': 0,  # No aggregation possible
                'number_of_patches': 0,  # No patches
                'avg_patch_size': 0,
                'patch_size_std': 0,
                'patch_size_median': 0,
                'patch_size_min': 0,
                'patch_size_max': 0,
                'normalized_lsi': 0,  # No shape complexity
                'landscape_type': 'all_non_canopy'
            })
        else:
            # All ones - complete canopy coverage
            total_area = total_cells * area_multiplier
            results.update({
                'morans_i': 1.0,  # Perfect positive autocorrelation
                'join_count_bb': float('inf'),  # All possible joins are BB
                'hansens_uniformity': 1.0,  # Perfect uniformity (all same)
                'geary_c': 0.0,  # Perfect clustering (no differences)
                'edge_density': 0,  # No internal edges in uniform landscape
                'clumpy': 1.0,  # Maximum aggregation
                'number_of_patches': 1,  # Single large patch
                'avg_patch_size': total_area,
                'patch_size_std': 0,  # No variation in patch size
                'patch_size_median': total_area,
                'patch_size_min': total_area,
                'patch_size_max': total_area,
                'normalized_lsi': 1.0,  # Minimum shape complexity (circular)
                'landscape_type': 'all_canopy'
            })
        return results

    # Mixed landscape - calculate all metrics normally
    results['landscape_type'] = 'mixed'

    # 2. SPATIAL AUTOCORRELATION METRICS
    results['morans_i'] = calculate_morans_i(binary_raster)  # Keep for comparison but not recommended
    results['join_count_bb'] = calculate_join_count_bb(binary_raster)  # RECOMMENDED for binary data
    results['hansens_uniformity'] = calculate_hansens_uniformity(binary_raster)  # Hansen's measure
    results['geary_c'] = calculate_geary_c_binary(binary_raster)  # Alternative to Moran's I

    # 3. EDGE DENSITY
    results['edge_density'] = calculate_edge_density(binary_raster, cell_size)

    # 4. CLUMPY INDEX
    results['clumpy'] = calculate_clumpy_index(binary_raster)

    # Patch-based metrics using consistent raster approach
    patch_metrics = calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size)
    results.update(patch_metrics)

    return results


def calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size):
    """
    Calculate patch-based landscape metrics using consistent raster approach.

    Patch Metrics Calculated:
    ------------------------
    1. number_of_patches: Total count of discrete canopy patches (8-connected)
    2. avg_patch_size: Mean area of all patches
    3. patch_size_std: Standard deviation of patch areas
    4. patch_size_median: Median patch area
    5. patch_size_min: Smallest patch area
    6. patch_size_max: Largest patch area
    7. normalized_lsi: Normalized Landscape Shape Index (shape complexity)

    Parameters:
    -----------
    binary_raster : numpy.ndarray
        Binary raster (0=non-canopy, 1=canopy)
    area_multiplier : float
        Multiplier to convert pixel count to real area units
    cell_size : float
        Size of each pixel in real units

    Returns:
    --------
    dict : Dictionary of patch metrics
    """
    # Label connected components (patches) using 8-connectivity
    # 8-connectivity means diagonal neighbors are considered connected
    labeled_patches, num_patches = label(binary_raster, connectivity=2, return_num=True)

    results = {'number_of_patches': num_patches}

    if num_patches > 0:
        # Get patch properties using scikit-image regionprops
        patch_props = regionprops(labeled_patches)

        # Calculate patch areas in real units
        patch_areas = [prop.area * area_multiplier for prop in patch_props]

        # Basic patch area statistics
        results.update({
            'avg_patch_size': np.mean(patch_areas),
            'patch_size_std': np.std(patch_areas, ddof=1) if len(patch_areas) > 1 else 0,
            'patch_size_median': np.median(patch_areas),
            'patch_size_min': np.min(patch_areas),
            'patch_size_max': np.max(patch_areas)
        })

        # Calculate normalized LSI (shape complexity measure)
        results['normalized_lsi'] = calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size)

    else:
        # No patches found (all zeros)
        results.update({
            'avg_patch_size': 0,
            'patch_size_std': 0,
            'patch_size_median': 0,
            'patch_size_min': 0,
            'patch_size_max': 0,
            'normalized_lsi': 0
        })

    return results


def calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size):
    """Calculate Normalized Landscape Shape Index using raster approach."""
    try:
        total_area = 0
        total_perimeter = 0

        for prop in patch_props:
            # Area in real units
            area = prop.area * (cell_size ** 2)

            # Calculate perimeter using more accurate method
            # Count edge pixels by checking 4-connectivity boundaries
            patch_mask = (labeled_patches == prop.label)

            # Use convolution to count edge cells
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 4-connectivity
            edges = ndimage.convolve(patch_mask.astype(int), kernel, mode='constant', cval=0)
            # Edge cells are those with at least one non-patch neighbor
            edge_cells = np.sum((patch_mask) & (edges < 4))

            # Convert to actual perimeter length
            perimeter = edge_cells * cell_size

            total_area += area
            total_perimeter += perimeter

        if total_area > 0:
            # Standard LSI calculation
            lsi = total_perimeter / (2 * np.sqrt(np.pi * total_area))
            return lsi
        else:
            return 0

    except Exception as e:
        warnings.warn(f"Could not calculate normalized LSI: {e}")
        return 0


def calculate_morans_i(raster_data):
    """
    Calculate Moran's I spatial autocorrelation index.
    NOTE: This is not recommended for binary data - kept for comparison purposes only.
    """
    try:
        rows, cols = raster_data.shape
        n = rows * cols

        # Flatten the raster
        values = raster_data.flatten()

        # Create spatial weights matrix (8-connectivity)
        coordinates = np.array([[i, j] for i in range(rows) for j in range(cols)])

        # Use a subset for large rasters to avoid memory issues
        if n > 10000:
            sample_size = 5000
            indices = np.random.choice(n, sample_size, replace=False)
            coordinates = coordinates[indices]
            values = values[indices]
            n = sample_size

        # Calculate distances and create weights
        distances = pdist(coordinates, metric='chebyshev')  # Max distance (8-connectivity)
        weights = (distances <= 1.5).astype(float)  # Adjacent cells

        # Convert to square matrix
        from scipy.spatial.distance import squareform
        W = squareform(weights)
        np.fill_diagonal(W, 0)  # No self-weights

        # Calculate Moran's I
        W_sum = np.sum(W)
        if W_sum == 0:
            return 0

        mean_val = np.mean(values)
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > 0:
                    numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2

        if denominator == 0:
            return 0

        morans_i = (n / W_sum) * (numerator / denominator)
        return morans_i

    except Exception as e:
        warnings.warn(f"Could not calculate Moran's I: {e}")
        return 0


def calculate_edge_density(binary_raster, cell_size):
    """Calculate edge density (total edge length per unit area)."""
    try:
        # Create edge map using morphological operations
        from scipy.ndimage import binary_erosion

        # Find edges by subtracting eroded image from original
        eroded = binary_erosion(binary_raster)
        edges = binary_raster - eroded

        # Calculate total edge length
        edge_cells = np.sum(edges)
        edge_length = edge_cells * cell_size

        # Total landscape area
        total_area = binary_raster.size * (cell_size ** 2)

        # Edge density (length per unit area)
        edge_density = edge_length / total_area if total_area > 0 else 0

        return edge_density

    except Exception as e:
        warnings.warn(f"Could not calculate edge density: {e}")
        return 0


def calculate_clumpy_index(binary_raster):
    """Calculate CLUMPY index (measure of aggregation)."""
    try:
        # Calculate proportion of landscape
        P = np.sum(binary_raster) / binary_raster.size

        if P == 0 or P == 1:
            return 0

        # Calculate observed proportion of like adjacencies
        rows, cols = binary_raster.shape
        like_adjacencies = 0
        total_adjacencies = 0

        # Check horizontal adjacencies
        for i in range(rows):
            for j in range(cols - 1):
                if binary_raster[i, j] == binary_raster[i, j + 1] == 1:
                    like_adjacencies += 1
                total_adjacencies += 1

        # Check vertical adjacencies
        for i in range(rows - 1):
            for j in range(cols):
                if binary_raster[i, j] == binary_raster[i + 1, j] == 1:
                    like_adjacencies += 1
                total_adjacencies += 1

        if total_adjacencies == 0:
            return 0

        # Observed proportion of like adjacencies
        G_observed = like_adjacencies / total_adjacencies

        # Expected proportion under random distribution
        G_expected = P

        # CLUMPY index
        if P < 0.5:
            clumpy = (G_observed - G_expected) / (P)
        else:
            clumpy = (G_observed - G_expected) / (1 - P)

        return clumpy

    except Exception as e:
        warnings.warn(f"Could not calculate CLUMPY index: {e}")
        return 0


def calculate_join_count_bb(binary_raster):
    """
    Calculate Join Count BB statistic for binary spatial autocorrelation.
    This is the appropriate measure for binary data instead of Moran's I.

    Join Count BB measures the number of Black-Black joins (canopy-canopy adjacencies)
    relative to what would be expected under spatial randomness.

    Returns:
    --------
    float : Standardized Join Count BB statistic
        - Positive values = clustering (more BB joins than expected)
        - Negative values = dispersion (fewer BB joins than expected)
        - Zero = random spatial pattern
    """
    try:
        rows, cols = binary_raster.shape

        # Count actual Black-Black (1-1) joins using 4-connectivity
        bb_joins = 0
        total_joins = 0

        # Horizontal adjacencies
        for i in range(rows):
            for j in range(cols - 1):
                if binary_raster[i, j] == 1 and binary_raster[i, j + 1] == 1:
                    bb_joins += 1
                total_joins += 1

        # Vertical adjacencies
        for i in range(rows - 1):
            for j in range(cols):
                if binary_raster[i, j] == 1 and binary_raster[i + 1, j] == 1:
                    bb_joins += 1
                total_joins += 1

        if total_joins == 0:
            return 0

        # Calculate expected BB joins under random distribution
        p = np.sum(binary_raster) / binary_raster.size  # Proportion of 1s
        expected_bb = total_joins * p * p

        # Calculate variance (simplified version)
        variance_bb = total_joins * p * p * (1 - p * p)

        if variance_bb <= 0:
            return 0

        # Standardized Join Count BB
        standardized_bb = (bb_joins - expected_bb) / np.sqrt(variance_bb)

        return standardized_bb

    except Exception as e:
        warnings.warn(f"Could not calculate Join Count BB: {e}")
        return 0


def calculate_hansens_uniformity(binary_raster):
    """
    Calculate Hansen's Uniformity Index for binary spatial patterns.

    This measures the uniformity of the spatial distribution of binary values.
    Based on the variance in local density across the landscape.

    Returns:
    --------
    float : Hansen's Uniformity Index (0-1)
        - 0 = maximum clustering/segregation
        - 1 = maximum uniformity/mixing
        - 0.5 = random distribution
    """
    try:
        rows, cols = binary_raster.shape

        # Use a moving window to calculate local densities
        window_size = min(5, min(rows, cols) // 3)  # Adaptive window size
        if window_size < 3:
            window_size = 3

        local_densities = []

        # Calculate local density for each possible window position
        for i in range(rows - window_size + 1):
            for j in range(cols - window_size + 1):
                window = binary_raster[i:i + window_size, j:j + window_size]
                local_density = np.mean(window)
                local_densities.append(local_density)

        if len(local_densities) == 0:
            return 0.5  # Default to random

        local_densities = np.array(local_densities)

        # Calculate observed variance in local densities
        observed_variance = np.var(local_densities)

        # Calculate global proportion
        global_proportion = np.mean(binary_raster)

        # Calculate maximum possible variance (occurs with maximum segregation)
        # This happens when windows are either all 0s or all 1s
        max_variance = global_proportion * (1 - global_proportion)

        if max_variance <= 0:
            return 1.0  # Uniform (all same value)

        # Hansen's Uniformity Index
        uniformity = 1 - (observed_variance / max_variance)

        # Ensure bounds [0, 1]
        uniformity = max(0, min(1, uniformity))

        return uniformity

    except Exception as e:
        warnings.warn(f"Could not calculate Hansen's Uniformity: {e}")
        return 0.5


def calculate_geary_c_binary(binary_raster):
    """
    Calculate Geary's C for binary data (alternative to Moran's I).

    Geary's C is more appropriate for binary data than Moran's I because
    it focuses on differences between neighboring values rather than covariance.

    Returns:
    --------
    float : Geary's C statistic
        - C < 1 = positive spatial autocorrelation (clustering)
        - C = 1 = no spatial autocorrelation (random)
        - C > 1 = negative spatial autocorrelation (dispersion)
    """
    try:
        rows, cols = binary_raster.shape
        n = rows * cols

        # Calculate mean
        mean_val = np.mean(binary_raster)

        # Calculate numerator: sum of squared differences between neighbors
        numerator = 0
        w_sum = 0  # Sum of weights

        # 4-connectivity neighbors
        for i in range(rows):
            for j in range(cols):
                # Check right neighbor
                if j < cols - 1:
                    diff = (binary_raster[i, j] - binary_raster[i, j + 1]) ** 2
                    numerator += diff
                    w_sum += 1

                # Check bottom neighbor
                if i < rows - 1:
                    diff = (binary_raster[i, j] - binary_raster[i + 1, j]) ** 2
                    numerator += diff
                    w_sum += 1

        # Calculate denominator: sum of squared deviations from mean
        denominator = 0
        for i in range(rows):
            for j in range(cols):
                denominator += (binary_raster[i, j] - mean_val) ** 2

        if denominator == 0 or w_sum == 0:
            return 1.0  # No variation or no neighbors

        # Geary's C formula
        geary_c = ((n - 1) / (2 * w_sum)) * (numerator / denominator)

        return geary_c

    except Exception as e:
        warnings.warn(f"Could not calculate Geary's C: {e}")
        return 1.0

#endregion