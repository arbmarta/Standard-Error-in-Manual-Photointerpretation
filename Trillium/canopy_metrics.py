import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from tqdm import tqdm
from multiprocessing import Pool
from shapely.geometry import box
import glob
import os

USE_TEST_SETTINGS = True

# HPC-optimized configuration
CHUNK_SIZE = 1  # Process 1 cell at a time for better progress visibility

# Configure paths based on test settings
if USE_TEST_SETTINGS:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw_test"
    GRID_PATH = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_100km.gpkg"
    OUTPUT_SUFFIX = "_test"
else:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw"
    # Production mode uses multiple grid files
    GRID_PATHS = [
        "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_3km.gpkg",
        "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_24km.gpkg",
        "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_54km.gpkg"
    ]
    OUTPUT_SUFFIX = ""

# Binary threshold: values < 2 → 0, values >= 2 → 1
CANOPY_THRESHOLD = 2.0

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for worker processes
_RASTER_CACHE = None


def build_raster_cache_once(raster_folder):
    """
    Build raster metadata cache ONCE in main process.

    OPTIMIZATION: Instead of each worker opening all files (32 × 156 = 4,992 opens),
    we open each file once (156 opens) and share the cache with workers.

    Speedup: ~30x faster initialization (1-2 min vs 40+ min)
    """
    print("\nCaching raster metadata (one-time operation)...")
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    if len(raster_files) == 0:
        print(f"WARNING: No raster files found in {raster_folder}")
        return []

    print(f"Found {len(raster_files)} raster files")

    cache = []
    for i, raster_path in enumerate(raster_files):
        if (i + 1) % 50 == 0 or (i + 1) == len(raster_files):
            print(f"  Caching metadata: {i + 1}/{len(raster_files)}", end='\r')

        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                cache.append({
                    'path': raster_path,
                    'bounds': bounds,
                    'box': box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                })
        except Exception as e:
            logger.warning(f"Could not cache {raster_path}: {e}")
            continue

    print(f"\n✓ Cached {len(cache)} raster metadata entries")
    return cache


def init_worker_with_cache(cache_data):
    """
    Initialize worker with pre-built cache.

    OPTIMIZATION: Workers receive cache instead of building it themselves.
    This is nearly instantaneous vs opening all files.
    """
    global _RASTER_CACHE
    _RASTER_CACHE = cache_data
    # No print statements to avoid multiprocessing issues


def get_intersecting_rasters_cached(grid_cell_geometry):
    """
    Find all raster files that intersect with a given grid cell.

    Uses pre-cached bounds - NO file I/O here!
    """
    global _RASTER_CACHE

    if _RASTER_CACHE is None:
        logger.error("Raster cache not initialized!")
        return []

    intersecting_rasters = []

    for raster_info in _RASTER_CACHE:
        try:
            if grid_cell_geometry.intersects(raster_info['box']):
                intersecting_rasters.append(raster_info['path'])
        except Exception as e:
            logger.warning(f"Error checking intersection: {e}")
            continue

    return intersecting_rasters


def apply_binary_threshold(raster_array, threshold=CANOPY_THRESHOLD):
    """Apply binary threshold to raw CHM data."""
    valid_mask = ~np.isnan(raster_array)
    binary_raster = np.zeros(raster_array.shape, dtype=np.uint8)
    binary_raster[valid_mask & (raster_array >= threshold)] = 1
    return binary_raster


def get_optimal_gdal_cache(n_workers=24):
    """Calculate optimal GDAL cache for Trillium."""
    total_ram_gb = 768
    safety_margin = 0.85
    usable_ram_gb = total_ram_gb * safety_margin
    ram_per_worker_gb = usable_ram_gb / n_workers
    gdal_cache_gb = ram_per_worker_gb * 0.5
    gdal_cache_mb = int(gdal_cache_gb * 1024)
    gdal_cache_mb = min(gdal_cache_mb, 65536)
    gdal_cache_mb = max(gdal_cache_mb, 4096)
    return gdal_cache_mb


GDAL_CACHEMAX = get_optimal_gdal_cache()


def mosaic_rasters(raster_paths, target_geometry, target_resolution=None, apply_threshold=True):
    """Mosaic multiple rasters that intersect a target geometry."""
    if not raster_paths:
        return None, None

    if len(raster_paths) == 1:
        with rasterio.open(raster_paths[0]) as src:
            out_image, out_transform = mask(
                src, [target_geometry], crop=True, all_touched=False
            )
            if apply_threshold:
                out_image[0] = apply_binary_threshold(out_image[0])
            return out_image[0], out_transform

    try:
        with rasterio.Env(
                GDAL_CACHEMAX=GDAL_CACHEMAX,
                GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                GDAL_NUM_THREADS='ALL_CPUS',
                CPL_VSIL_CURL_CHUNK_SIZE=52428800,
                VSI_CACHE=True,
                VSI_CACHE_SIZE=500000000
        ):
            src_files = [rasterio.open(path) for path in raster_paths]

            if target_resolution is None:
                target_resolution = min([abs(src.transform.a) for src in src_files])

            mosaic_array, out_trans = merge(
                src_files,
                res=target_resolution,
                method='max'
            )

            for src in src_files:
                src.close()

            from rasterio.io import MemoryFile
            with MemoryFile() as memfile:
                with memfile.open(
                        driver='GTiff',
                        height=mosaic_array.shape[1],
                        width=mosaic_array.shape[2],
                        count=1,
                        dtype=mosaic_array.dtype,
                        crs=src_files[0].crs,
                        transform=out_trans
                ) as dataset:
                    dataset.write(mosaic_array[0], 1)
                    out_image, out_transform = mask(
                        dataset, [target_geometry], crop=True, all_touched=False
                    )
                    if apply_threshold:
                        out_image[0] = apply_binary_threshold(out_image[0])
                    return out_image[0], out_transform

    except Exception as e:
        logger.error(f"Error in mosaicking: {e}")
        return None, None


def process_single_cell(args):
    """Process a single grid cell - designed for parallel execution."""
    cell_id, geometry = args

    try:
        intersecting_rasters = get_intersecting_rasters_cached(geometry)

        if not intersecting_rasters:
            return {
                'cell_id': cell_id,
                'canopy_cover': 0,
                'total_pixels': 0,
                'canopy_pixels': 0,
                'num_intersecting_rasters': 0,
                'error': 'no_rasters_found'
            }

        binary_raster, transform = mosaic_rasters(
            intersecting_rasters,
            geometry,
            apply_threshold=True
        )

        if binary_raster is None:
            return {
                'cell_id': cell_id,
                'canopy_cover': 0,
                'total_pixels': 0,
                'canopy_pixels': 0,
                'num_intersecting_rasters': len(intersecting_rasters),
                'error': 'mosaic_failed'
            }

        metrics = calculate_all_metrics(binary_raster, transform)
        metrics['cell_id'] = cell_id
        metrics['num_intersecting_rasters'] = len(intersecting_rasters)

        return metrics

    except Exception as e:
        logger.error(f"Error processing cell {cell_id}: {e}")
        return {
            'cell_id': cell_id,
            'canopy_cover': 0,
            'total_pixels': 0,
            'canopy_pixels': 0,
            'num_intersecting_rasters': 0,
            'error': str(e)
        }


def calculate_all_metrics(binary_raster, transform):
    """Calculate all canopy metrics from binary raster."""
    if binary_raster is None or binary_raster.size == 0:
        return get_empty_metrics()

    total_pixels = binary_raster.size
    canopy_pixels = np.sum(binary_raster == 1)
    cell_size = abs(transform.a)
    area_multiplier = cell_size ** 2
    canopy_cover = (canopy_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    metrics = {
        'canopy_cover': canopy_cover,
        'total_pixels': int(total_pixels),
        'canopy_pixels': int(canopy_pixels),
        'cell_size_m': cell_size,
    }

    if canopy_pixels > 0 and canopy_cover > 1.0:
        try:
            metrics['morans_i'] = calculate_morans_i_fast(binary_raster)
            metrics['gearys_c'] = calculate_geary_c_fast(binary_raster)
            patch_metrics = calculate_patch_metrics_raster(
                binary_raster, area_multiplier, cell_size
            )
            metrics.update(patch_metrics)
            metrics['hansens_uniformity'] = calculate_hansens_uniformity(binary_raster)
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
            metrics.update(get_empty_advanced_metrics())
    else:
        metrics.update(get_empty_advanced_metrics())

    return metrics


def get_empty_metrics():
    """Return empty metrics for cells with no data."""
    return {
        'canopy_cover': 0,
        'total_pixels': 0,
        'canopy_pixels': 0,
        'cell_size_m': 0,
        **get_empty_advanced_metrics()
    }


def get_empty_advanced_metrics():
    """Return empty advanced metrics."""
    return {
        'morans_i': 0,
        'gearys_c': 1.0,
        'number_of_patches': 0,
        'avg_patch_size': 0,
        'patch_size_std': 0,
        'patch_size_median': 0,
        'patch_size_min': 0,
        'patch_size_max': 0,
        'normalized_lsi': 0,
        'hansens_uniformity': 0.5
    }


def calculate_morans_i_fast(binary_raster):
    """Calculate Moran's I for spatial autocorrelation."""
    try:
        coords_y, coords_x = np.where(binary_raster >= 0)
        if len(coords_y) < 2:
            return 0

        max_sample = 500 if binary_raster.size > 1000000 else 1000
        sample_size = min(max_sample, len(coords_y))
        indices = np.random.choice(len(coords_y), sample_size, replace=False)

        coordinates = np.column_stack([coords_x[indices], coords_y[indices]])
        values = binary_raster[coords_y[indices], coords_x[indices]].astype(np.float32)

        n = len(values)
        distances = pdist(coordinates, metric='chebyshev')
        weights = (distances <= 1.5).astype(np.float32)

        from scipy.spatial.distance import squareform
        W = squareform(weights)
        np.fill_diagonal(W, 0)

        W_sum = np.sum(W)
        if W_sum == 0:
            return 0

        mean_val = np.mean(values)
        deviations = values - mean_val

        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations ** 2)

        if denominator == 0:
            return 0

        return (n / W_sum) * (numerator / denominator)

    except Exception as e:
        logger.warning(f"Could not calculate Moran's I: {e}")
        return 0


def calculate_geary_c_fast(binary_raster):
    """Calculate Geary's C for spatial autocorrelation."""
    try:
        mean_val = np.mean(binary_raster)

        h_diffs = np.sum((binary_raster[:, :-1] - binary_raster[:, 1:]) ** 2)
        v_diffs = np.sum((binary_raster[:-1, :] - binary_raster[1:, :]) ** 2)

        numerator = h_diffs + v_diffs
        w_sum = binary_raster.shape[0] * (binary_raster.shape[1] - 1) + \
                binary_raster.shape[1] * (binary_raster.shape[0] - 1)

        denominator = np.sum((binary_raster - mean_val) ** 2)

        if denominator == 0 or w_sum == 0:
            return 1.0

        n = binary_raster.size
        return ((n - 1) / (2 * w_sum)) * (numerator / denominator)

    except Exception as e:
        logger.warning(f"Could not calculate Geary's C: {e}")
        return 1.0


def calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size):
    """Calculate patch-based landscape metrics."""
    labeled_patches, num_patches = label(binary_raster, connectivity=1, return_num=True)

    results = {'number_of_patches': num_patches}

    if num_patches > 0:
        patch_props = regionprops(labeled_patches, extra_properties=None)
        patch_areas = np.array([prop.area * area_multiplier for prop in patch_props])

        results.update({
            'avg_patch_size': np.mean(patch_areas),
            'patch_size_std': np.std(patch_areas, ddof=1) if len(patch_areas) > 1 else 0,
            'patch_size_median': np.median(patch_areas),
            'patch_size_min': np.min(patch_areas),
            'patch_size_max': np.max(patch_areas)
        })

        if num_patches < 1000:
            results['normalized_lsi'] = calculate_normalized_lsi_raster(
                labeled_patches, patch_props, cell_size
            )
        else:
            results['normalized_lsi'] = 0
    else:
        results.update({
            'avg_patch_size': 0, 'patch_size_std': 0, 'patch_size_median': 0,
            'patch_size_min': 0, 'patch_size_max': 0, 'normalized_lsi': 0
        })

    return results


def calculate_normalized_lsi_raster(labeled_patches, patch_props, cell_size):
    """Calculate Normalized Landscape Shape Index."""
    try:
        total_area = 0
        total_perimeter = 0

        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)

        for prop in patch_props:
            area = prop.area * (cell_size ** 2)
            patch_mask = (labeled_patches == prop.label)

            edges = ndimage.convolve(patch_mask.astype(np.int8), kernel, mode='constant', cval=0)
            edge_cells = np.sum(patch_mask & (edges < 4))
            perimeter = edge_cells * cell_size

            total_area += area
            total_perimeter += perimeter

        if total_area > 0:
            return total_perimeter / (2 * np.sqrt(np.pi * total_area))
        else:
            return 0

    except Exception as e:
        logger.warning(f"Could not calculate normalized LSI: {e}")
        return 0


def calculate_hansens_uniformity(binary_raster):
    """Calculate Hansen's Uniformity Index."""
    try:
        rows, cols = binary_raster.shape

        if rows * cols > 1000000:
            window_size = 3
        else:
            window_size = min(5, min(rows, cols) // 3)
            if window_size < 3:
                window_size = 3

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(binary_raster, (window_size, window_size))
        local_densities = np.mean(windows, axis=(2, 3)).flatten()

        observed_variance = np.var(local_densities)
        global_proportion = np.mean(binary_raster)
        max_variance = global_proportion * (1 - global_proportion)

        if max_variance <= 0:
            return 1.0

        uniformity = 1 - (observed_variance / max_variance)
        return max(0, min(1, uniformity))

    except Exception as e:
        logger.warning(f"Could not calculate Hansen's Uniformity: {e}")
        return 0.5


def process_grid_cells_parallel(grid, raster_cache, n_workers=24):
    """
    Process all grid cells in parallel.

    OPTIMIZATION: Workers receive pre-built cache for instant initialization.
    """
    print(f"\nProcessing {len(grid)} grid cells with {n_workers} workers...")
    print(f"Using GDAL_CACHEMAX: {GDAL_CACHEMAX} MB ({GDAL_CACHEMAX / 1024:.1f} GB per worker)")

    args_list = [
        (row['cell_id'] if 'cell_id' in row else idx, row.geometry)
        for idx, row in grid.iterrows()
    ]

    print(f"Starting workers (with pre-cached metadata)...")

    with Pool(n_workers, initializer=init_worker_with_cache, initargs=(raster_cache,)) as pool:
        results = list(tqdm(
            pool.imap(process_single_cell, args_list, chunksize=CHUNK_SIZE),
            total=len(args_list),
            desc="Processing cells"
        ))

    return pd.DataFrame(results)


# Main execution
if __name__ == "__main__":
    import geopandas as gpd

    print("=" * 70)
    print("CANOPY ANALYSIS WITH BINARY THRESHOLDING (OPTIMIZED v2)")
    print(f"Threshold: Values < {CANOPY_THRESHOLD}m → 0, Values >= {CANOPY_THRESHOLD}m → 1")
    print("=" * 70)

    # Build raster cache ONCE in main process (fast!)
    raster_cache = build_raster_cache_once(RASTER_FOLDER)

    if len(raster_cache) == 0:
        print("\nERROR: No raster metadata cached. Cannot proceed.")
        exit(1)

    # Determine which grids to process
    if USE_TEST_SETTINGS:
        grid_files = [GRID_PATH]
        print(f"\nTest mode: Processing single grid file")
    else:
        grid_files = GRID_PATHS
        print(f"\nProduction mode: Processing {len(grid_files)} grid files")

    # Process each grid file
    all_results = []

    for grid_idx, grid_path in enumerate(grid_files, 1):
        print(f"\n{'=' * 70}")
        print(f"Processing Grid {grid_idx}/{len(grid_files)}")
        print(f"{'=' * 70}")

        # Load grid
        print(f"Loading grid from: {grid_path}")
        grid = gpd.read_file(grid_path).to_crs(epsg=3857)
        print(f"Grid CRS: {grid.crs}")
        print(f"Grid cells: {len(grid)}")

        # Extract grid resolution from filename for output
        grid_name = os.path.basename(grid_path).replace('.gpkg', '').replace('grid_', '')

        # Process grid cells with pre-built cache
        n_workers = min(os.cpu_count(), 24)  # 24 workers for Trillium
        results_df = process_grid_cells_parallel(grid, raster_cache, n_workers)

        # Add grid identifier to results
        results_df['grid_resolution'] = grid_name

        # Save individual results
        output_path = f"/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/canopy_metrics_{grid_name}{OUTPUT_SUFFIX}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")

        all_results.append(results_df)

        # Summary statistics for this grid
        print(f"\nSummary for {grid_name}:")
        print(f"  Total cells processed: {len(results_df)}")
        print(f"  Cells with canopy: {(results_df['canopy_cover'] > 0).sum()}")
        print(f"  Mean canopy cover: {results_df['canopy_cover'].mean():.2f}%")
        print(f"  Median canopy cover: {results_df['canopy_cover'].median():.2f}%")

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save combined results
    combined_output = f"/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/canopy_metrics_combined{OUTPUT_SUFFIX}.csv"
    combined_results.to_csv(combined_output, index=False)
    print(f"\n{'=' * 70}")
    print(f"✓ Combined results saved to: {combined_output}")
    print(f"{'=' * 70}")

    # Overall summary statistics
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total cells processed: {len(combined_results)}")
    print(f"Cells with canopy: {(combined_results['canopy_cover'] > 0).sum()}")
    print(f"Mean canopy cover: {combined_results['canopy_cover'].mean():.2f}%")
    print(f"Median canopy cover: {combined_results['canopy_cover'].median():.2f}%")
    print(f"Cells with intersecting rasters: {(combined_results['num_intersecting_rasters'] > 0).sum()}")

    # Summary by grid resolution
    if not USE_TEST_SETTINGS and len(grid_files) > 1:
        print("\nSummary by grid resolution:")
        for grid_res in combined_results['grid_resolution'].unique():
            grid_data = combined_results[combined_results['grid_resolution'] == grid_res]
            print(f"\n  {grid_res}:")
            print(f"    Cells: {len(grid_data)}")
            print(f"    Mean canopy cover: {grid_data['canopy_cover'].mean():.2f}%")
            print(f"    Median canopy cover: {grid_data['canopy_cover'].median():.2f}%")

    if 'error' in combined_results.columns:
        errors = combined_results['error'].notna().sum()
        if errors > 0:
            print(f"\n⚠️  Cells with errors: {errors}")
            print(combined_results['error'].value_counts())