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

USE_TEST_SETTINGS = True

# HPC-optimized configuration
CHUNK_SIZE = 50

# Configure paths based on test settings
if USE_TEST_SETTINGS:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw_test"
    OUTPUT_SUFFIX = "_test"
else:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw"
    OUTPUT_SUFFIX = ""

# GPKG file path
GRID_PATH = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_100km.gpkg"

# Binary threshold: values < 2 → 0, values >= 2 → 1
CANOPY_THRESHOLD = 2.0

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_SPATIAL_INDEX = None
_RASTER_INFO = None


def initialize_spatial_index():
    """Initialize spatial index once and cache it globally."""
    global _SPATIAL_INDEX, _RASTER_INFO
    if _SPATIAL_INDEX is not None:
        return _SPATIAL_INDEX, _RASTER_INFO
    print("Building spatial index (one-time initialization)...")
    _SPATIAL_INDEX, _RASTER_INFO = build_raster_spatial_index(RASTER_FOLDER)
    return _SPATIAL_INDEX, _RASTER_INFO


def build_raster_spatial_index(raster_folder):
    """Build a spatial index of all raster files."""
    try:
        from rtree import index
    except ImportError:
        print("Warning: rtree not available. Install with: pip install rtree")
        return None, None

    idx = index.Index()
    raster_info = {}

    import glob
    import os
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"Indexing {len(raster_files)} raw CHM raster files...")

    indexed_count = 0
    for i, raster_path in enumerate(raster_files):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds

                if bounds.left >= bounds.right or bounds.bottom >= bounds.top:
                    logger.warning(f"Invalid bounds for {raster_path}: {bounds}")
                    continue

                crs = src.crs if src.crs is not None else rasterio.crs.CRS.from_epsg(3857)
                idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top))

                raster_info[i] = {
                    'path': raster_path,
                    'bounds': bounds,
                    'crs': crs,
                }
                indexed_count += 1

        except Exception as e:
            logger.warning(f"Could not index {raster_path}: {e}")
            continue

    print(f"Spatial index built with {indexed_count} rasters")

    if indexed_count == 0:
        logger.error("No rasters were successfully indexed!")
        return None, None

    return idx, raster_info


def apply_binary_threshold(raster_array, threshold=CANOPY_THRESHOLD):
    """
    Apply binary threshold to raw CHM data.

    Values < threshold → 0 (no canopy)
    Values >= threshold → 1 (canopy present)

    Args:
        raster_array: numpy array with raw CHM values
        threshold: height threshold (default: 2.0 meters)

    Returns:
        Binary numpy array (0s and 1s)
    """
    # Create binary: < 2 becomes 0, >= 2 becomes 1
    binary_raster = np.where(raster_array >= threshold, 1, 0).astype(np.uint8)

    # Handle NaN or nodata values - set them to 0
    if np.isnan(raster_array).any():
        binary_raster[np.isnan(raster_array)] = 0

    return binary_raster


def get_intersecting_rasters_indexed(grid_cell_geometry, spatial_index, raster_info):
    """Find all raster files that intersect with a given grid cell geometry."""
    intersecting_rasters = []
    minx, miny, maxx, maxy = grid_cell_geometry.bounds

    if minx >= maxx or miny >= maxy:
        logger.warning(f"Invalid geometry bounds: {grid_cell_geometry.bounds}")
        return []

    if spatial_index is not None:
        try:
            candidate_ids = list(spatial_index.intersection((minx, miny, maxx, maxy)))
        except Exception as e:
            logger.error(f"Error querying spatial index: {e}")
            candidate_ids = list(raster_info.keys())
    else:
        candidate_ids = list(raster_info.keys())

    if len(candidate_ids) == 0:
        logger.debug(f"No candidates from spatial index for bounds: {(minx, miny, maxx, maxy)}")

    for raster_id in candidate_ids:
        try:
            raster_bounds = raster_info[raster_id]['bounds']
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)

            if grid_cell_geometry.intersects(raster_geom):
                intersecting_rasters.append(raster_info[raster_id]['path'])

        except Exception as e:
            logger.warning(f"Error checking intersection for raster {raster_id}: {e}")
            continue

    return intersecting_rasters


def mosaic_rasters(raster_paths, target_geometry, target_resolution=None, apply_threshold=True):
    """
    Mosaic multiple rasters that intersect a target geometry.

    NEW: Applies binary threshold after mosaicking if apply_threshold=True.
    """
    if not raster_paths:
        return None, None

    if len(raster_paths) == 1:
        with rasterio.open(raster_paths[0]) as src:
            out_image, out_transform = mask(
                src, [target_geometry], crop=True, all_touched=False
            )
            # Apply binary threshold to raw CHM data
            if apply_threshold:
                out_image[0] = apply_binary_threshold(out_image[0])
            return out_image[0], out_transform

    try:
        # Use context manager for all files
        with rasterio.Env(GDAL_CACHEMAX=512):
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
                    # Apply binary threshold to raw CHM data
                    if apply_threshold:
                        out_image[0] = apply_binary_threshold(out_image[0])
                    return out_image[0], out_transform

    except Exception as e:
        logger.error(f"Error in mosaicking: {e}")
        return None, None


def process_single_cell(args):
    """Process a single grid cell - designed for parallel execution."""
    cell_id, geometry, spatial_index, raster_info = args

    try:
        intersecting_rasters = get_intersecting_rasters_indexed(
            geometry, spatial_index, raster_info
        )

        if not intersecting_rasters:
            return {
                'cell_id': cell_id,
                'canopy_cover': 0,
                'total_pixels': 0,
                'canopy_pixels': 0,
                'error': 'no_rasters_found'
            }

        # Mosaic raw rasters and apply binary threshold
        binary_raster, transform = mosaic_rasters(
            intersecting_rasters,
            geometry,
            apply_threshold=True  # Apply threshold: < 2 → 0, >= 2 → 1
        )

        if binary_raster is None:
            return {
                'cell_id': cell_id,
                'canopy_cover': 0,
                'total_pixels': 0,
                'canopy_pixels': 0,
                'error': 'mosaic_failed'
            }

        # Calculate metrics on binary raster
        metrics = calculate_all_metrics(binary_raster, transform)
        metrics['cell_id'] = cell_id

        return metrics

    except Exception as e:
        logger.error(f"Error processing cell {cell_id}: {e}")
        return {
            'cell_id': cell_id,
            'canopy_cover': 0,
            'total_pixels': 0,
            'canopy_pixels': 0,
            'error': str(e)
        }


def calculate_all_metrics(binary_raster, transform):
    """Calculate all canopy metrics from binary raster."""

    if binary_raster is None or binary_raster.size == 0:
        return get_empty_metrics()

    # Basic counts
    total_pixels = binary_raster.size
    canopy_pixels = np.sum(binary_raster == 1)

    # Calculate cell size from transform
    cell_size = abs(transform.a)
    area_multiplier = cell_size ** 2

    # Basic canopy cover
    canopy_cover = (canopy_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    metrics = {
        'canopy_cover': canopy_cover,
        'total_pixels': int(total_pixels),
        'canopy_pixels': int(canopy_pixels),
        'cell_size_m': cell_size,
    }

    # Only calculate advanced metrics if there's canopy present
    if canopy_pixels > 0:
        try:
            # Spatial autocorrelation metrics
            metrics['morans_i'] = calculate_morans_i_fast(binary_raster)
            metrics['gearys_c'] = calculate_geary_c_fast(binary_raster)

            # Patch-based metrics
            patch_metrics = calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size)
            metrics.update(patch_metrics)

            # Uniformity
            metrics['hansens_uniformity'] = calculate_hansens_uniformity(binary_raster)

        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
            metrics.update(get_empty_advanced_metrics())
    else:
        metrics.update(get_empty_advanced_metrics())

    return metrics


def get_empty_metrics():
    """Return empty metrics dictionary."""
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

        sample_size = min(1000, len(coords_y))
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
    labeled_patches, num_patches = label(binary_raster, connectivity=2, return_num=True)

    results = {'number_of_patches': num_patches}

    if num_patches > 0:
        patch_props = regionprops(labeled_patches)
        patch_areas = np.array([prop.area * area_multiplier for prop in patch_props])

        results.update({
            'avg_patch_size': np.mean(patch_areas),
            'patch_size_std': np.std(patch_areas, ddof=1) if len(patch_areas) > 1 else 0,
            'patch_size_median': np.median(patch_areas),
            'patch_size_min': np.min(patch_areas),
            'patch_size_max': np.max(patch_areas)
        })

        results['normalized_lsi'] = calculate_normalized_lsi_raster(
            labeled_patches, patch_props, cell_size
        )
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


def process_grid_cells_parallel(grid, spatial_index, raster_info, n_workers=4):
    """Process all grid cells in parallel."""
    print(f"\nProcessing {len(grid)} grid cells with {n_workers} workers...")

    # Prepare arguments for parallel processing
    args_list = [
        (row['cell_id'] if 'cell_id' in row else idx, row.geometry, spatial_index, raster_info)
        for idx, row in grid.iterrows()
    ]

    # Process in parallel with progress bar
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_cell, args_list, chunksize=CHUNK_SIZE),
            total=len(args_list),
            desc="Processing cells"
        ))

    return pd.DataFrame(results)


def validate_spatial_index(spatial_index, raster_info, grid_sample):
    """Validate that the spatial index is working correctly."""
    if spatial_index is None or raster_info is None:
        print("❌ Spatial index not initialized")
        return False

    print("\n" + "=" * 60)
    print("VALIDATING SPATIAL INDEX")
    print("=" * 60)

    test_cell = grid_sample.iloc[0]
    test_geom = test_cell.geometry
    minx, miny, maxx, maxy = test_geom.bounds

    print(f"Test cell bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

    candidates = list(spatial_index.intersection((minx, miny, maxx, maxy)))
    print(f"Spatial index returned: {len(candidates)} candidates")

    manual_count = 0
    for raster_id, info in raster_info.items():
        bounds = info['bounds']
        raster_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        if test_geom.intersects(raster_box):
            manual_count += 1

    print(f"Manual intersection check: {manual_count} rasters")

    if len(candidates) != manual_count:
        print(f"❌ MISMATCH: Index returned {len(candidates)} but manual check found {manual_count}")
        return False
    else:
        print(f"✓ Spatial index validation PASSED")
        return True


# Main execution
if __name__ == "__main__":
    import geopandas as gpd
    import os

    print("=" * 70)
    print("CANOPY ANALYSIS WITH BINARY THRESHOLDING")
    print(f"Threshold: Values < {CANOPY_THRESHOLD}m → 0, Values >= {CANOPY_THRESHOLD}m → 1")
    print("=" * 70)

    # Build spatial index for raw CHM rasters
    print("\nBuilding spatial index...")
    spatial_index, raster_info = build_raster_spatial_index(RASTER_FOLDER)

    if spatial_index is None:
        print("Failed to build spatial index!")
        exit(1)

    # Load grid
    print(f"\nLoading grid from: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH).to_crs(epsg=3857)
    print(f"Grid CRS: {grid.crs}")
    print(f"Grid cells: {len(grid)}")

    # Validate spatial index
    print("\nValidating spatial index...")
    if not validate_spatial_index(spatial_index, raster_info, grid):
        print("\n⚠️  SPATIAL INDEX VALIDATION FAILED!")
        print("Possible issues:")
        print("1. CRS mismatch between rasters and grid")
        print("2. Incorrect bounds format in index")
        print("3. Rasters don't actually overlap with grid cells")
        exit(1)
    else:
        print("\n✓ Spatial index is working correctly!")

    # Process grid cells
    n_workers = min(os.cpu_count(), 16)  # Limit to 16 workers max
    results_df = process_grid_cells_parallel(grid, spatial_index, raster_info, n_workers)

    # Save results
    output_path = f"/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/canopy_metrics{OUTPUT_SUFFIX}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total cells processed: {len(results_df)}")
    print(f"Cells with canopy: {(results_df['canopy_cover'] > 0).sum()}")
    print(f"Mean canopy cover: {results_df['canopy_cover'].mean():.2f}%")
    print(f"Median canopy cover: {results_df['canopy_cover'].median():.2f}%")

    if 'error' in results_df.columns:
        errors = results_df['error'].notna().sum()
        if errors > 0:
            print(f"\n⚠️  Cells with errors: {errors}")
            print(results_df['error'].value_counts())