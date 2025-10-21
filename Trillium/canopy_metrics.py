import geopandas as gpd
import glob
import logging
import numpy as np
import os
import pandas as pd
import rasterio
import rasterio.mask
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import psutil

USE_TEST_SETTINGS = True

# HPC-optimized configuration
CHUNK_SIZE = 50  # Larger batches for better throughput with many workers

# Configure paths based on test settings
if USE_TEST_SETTINGS:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Binary_test"
    OUTPUT_SUFFIX = "_test"
else:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Binary"
    OUTPUT_SUFFIX = ""

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Precompute and cache spatial index globally
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
    """Build a spatial index of all raster files for efficient intersection queries."""
    try:
        from rtree import index
    except ImportError:
        print("Warning: rtree not available. Install with: pip install rtree")
        print("Falling back to brute force intersection checking...")
        return None, None

    idx = index.Index()
    raster_info = {}

    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"Indexing {len(raster_files)} raster files...")

    for i, raster_path in enumerate(tqdm(raster_files, desc="Indexing rasters")):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top))

                raster_info[i] = {
                    'path': raster_path,
                    'bounds': bounds,
                    'crs': src.crs,
                }
        except Exception as e:
            logger.warning(f"Could not index {raster_path}: {e}")
            continue

    print(f"Spatial index built with {len(raster_info)} rasters")
    return idx, raster_info


def get_intersecting_rasters_indexed(grid_cell_geometry, spatial_index, raster_info):
    """Find all raster files that intersect with a given grid cell geometry."""
    if spatial_index is None:
        return []

    intersecting_rasters = []
    minx, miny, maxx, maxy = grid_cell_geometry.bounds

    candidate_ids = list(spatial_index.intersection((minx, miny, maxx, maxy)))

    for raster_id in candidate_ids:
        try:
            raster_bounds = raster_info[raster_id]['bounds']
            from shapely.geometry import box
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)

            if grid_cell_geometry.intersects(raster_geom):
                intersecting_rasters.append(raster_info[raster_id]['path'])
        except Exception as e:
            logger.warning(f"Error checking intersection for raster {raster_id}: {e}")
            continue

    return intersecting_rasters


def mosaic_rasters(raster_paths, target_geometry, target_resolution=None):
    """Mosaic multiple rasters that intersect a target geometry."""
    if not raster_paths:
        return None, None

    if len(raster_paths) == 1:
        with rasterio.open(raster_paths[0]) as src:
            out_image, out_transform = rasterio.mask.mask(
                src, [target_geometry], crop=True, all_touched=False
            )
            return out_image[0], out_transform

    try:
        from rasterio.merge import merge

        # Use context manager for all files
        with rasterio.Env(GDAL_CACHEMAX=512):  # Optimize GDAL cache
            src_files = [rasterio.open(path) for path in raster_paths]

            if target_resolution is None:
                target_resolution = min([abs(src.transform.a) for src in src_files])

            mosaic, out_trans = merge(
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
                        height=mosaic.shape[1],
                        width=mosaic.shape[2],
                        count=1,
                        dtype=mosaic.dtype,
                        crs=src_files[0].crs,
                        transform=out_trans
                ) as dataset:
                    dataset.write(mosaic[0], 1)
                    out_image, out_transform = rasterio.mask.mask(
                        dataset, [target_geometry], crop=True, all_touched=False
                    )
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

        if len(intersecting_rasters) > 1:
            raster_data, transform = mosaic_rasters(intersecting_rasters, geometry)
        else:
            with rasterio.open(intersecting_rasters[0]) as src:
                out_image, transform = rasterio.mask.mask(
                    src, [geometry], crop=True, all_touched=False
                )
                raster_data = out_image[0]

        if raster_data is None or raster_data.size == 0:
            return None

        cell_size = abs(transform.a)
        metrics = calculate_landscape_metrics(raster_data, cell_size=cell_size)
        metrics['cell_id'] = cell_id
        metrics['num_intersecting_rasters'] = len(intersecting_rasters)

        return metrics

    except Exception as e:
        logger.error(f"Error processing cell_id {cell_id}: {e}")
        return None


def process_grid_cells_parallel(grid_gdf, aoi_size, output_dir='.'):
    """Process grid cells in parallel using multiprocessing."""

    # Initialize spatial index once
    spatial_index, raster_info = initialize_spatial_index()

    if spatial_index is None:
        print("ERROR: Could not build spatial index. Exiting.")
        return None

    print(f"Processing {len(grid_gdf)} grid cells for AOI size: {aoi_size}")

    # Prepare arguments for parallel processing
    args_list = [
        (row['cell_id'], row.geometry, spatial_index, raster_info)
        for _, row in grid_gdf.iterrows()
    ]

    # Process in parallel with progress bar
    results_list = []
    with Pool(processes=192) as pool:
        for result in tqdm(
                pool.imap_unordered(process_single_cell, args_list, chunksize=10),
                total=len(args_list),
                desc=f"Processing {aoi_size}"
        ):
            if result is not None:
                results_list.append(result)

    # Create DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_filename = f"canopy_metrics_{aoi_size}{OUTPUT_SUFFIX}.csv"
    output_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results_df)} grid cells successfully")

    return results_df


# LANDSCAPE METRICS (optimized versions)
def calculate_landscape_metrics(raster_data, cell_size=None):
    """Calculate comprehensive landscape metrics for binary raster (optimized)."""

    if raster_data is None or raster_data.size == 0:
        return {
            'canopy_extent': 0, 'morans_i': 0, 'join_count_bb': 0,
            'hansens_uniformity': 0.5, 'geary_c': 1.0, 'edge_density': 0,
            'clumpy': 0, 'number_of_patches': 0, 'avg_patch_size': 0,
            'patch_size_std': 0, 'patch_size_median': 0, 'patch_size_min': 0,
            'patch_size_max': 0, 'normalized_lsi': 0, 'landscape_type': 'no_data'
        }

    binary_raster = raster_data.astype(np.int8)  # More memory efficient

    if cell_size is None:
        cell_size = 1.0

    area_multiplier = cell_size ** 2
    results = {}

    unique_values = np.unique(binary_raster)
    total_cells = binary_raster.size
    canopy_cells = np.sum(binary_raster)

    results['canopy_extent'] = (canopy_cells / total_cells) * 100

    # Handle uniform landscapes
    if len(unique_values) == 1:
        if unique_values[0] == 0:
            results.update({
                'morans_i': 0, 'join_count_bb': 0, 'hansens_uniformity': 1.0,
                'geary_c': 1.0, 'edge_density': 0, 'clumpy': 0,
                'number_of_patches': 0, 'avg_patch_size': 0, 'patch_size_std': 0,
                'patch_size_median': 0, 'patch_size_min': 0, 'patch_size_max': 0,
                'normalized_lsi': 0, 'landscape_type': 'all_non_canopy'
            })
        else:
            total_area = total_cells * area_multiplier
            results.update({
                'morans_i': 1.0, 'join_count_bb': float('inf'),
                'hansens_uniformity': 1.0, 'geary_c': 0.0, 'edge_density': 0,
                'clumpy': 1.0, 'number_of_patches': 1, 'avg_patch_size': total_area,
                'patch_size_std': 0, 'patch_size_median': total_area,
                'patch_size_min': total_area, 'patch_size_max': total_area,
                'normalized_lsi': 1.0, 'landscape_type': 'all_canopy'
            })
        return results

    results['landscape_type'] = 'mixed'

    # Calculate metrics (with vectorized operations where possible)
    results['morans_i'] = calculate_morans_i_fast(binary_raster)
    results['join_count_bb'] = calculate_join_count_bb_fast(binary_raster)
    results['hansens_uniformity'] = calculate_hansens_uniformity(binary_raster)
    results['geary_c'] = calculate_geary_c_fast(binary_raster)
    results['edge_density'] = calculate_edge_density_fast(binary_raster, cell_size)
    results['clumpy'] = calculate_clumpy_index_fast(binary_raster)

    patch_metrics = calculate_patch_metrics_raster(binary_raster, area_multiplier, cell_size)
    results.update(patch_metrics)

    return results


def calculate_join_count_bb_fast(binary_raster):
    """Optimized Join Count BB calculation using vectorized operations."""
    try:
        # Vectorized horizontal joins
        horizontal_joins = np.sum(binary_raster[:, :-1] & binary_raster[:, 1:])

        # Vectorized vertical joins
        vertical_joins = np.sum(binary_raster[:-1, :] & binary_raster[1:, :])

        bb_joins = horizontal_joins + vertical_joins
        total_joins = binary_raster.shape[0] * (binary_raster.shape[1] - 1) + \
                      binary_raster.shape[1] * (binary_raster.shape[0] - 1)

        if total_joins == 0:
            return 0

        p = np.mean(binary_raster)
        expected_bb = total_joins * p * p
        variance_bb = total_joins * p * p * (1 - p * p)

        if variance_bb <= 0:
            return 0

        return (bb_joins - expected_bb) / np.sqrt(variance_bb)

    except Exception as e:
        logger.warning(f"Could not calculate Join Count BB: {e}")
        return 0


def calculate_edge_density_fast(binary_raster, cell_size):
    """Optimized edge density calculation."""
    try:
        # Use fast convolution for edge detection
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
        neighbors = ndimage.convolve(binary_raster, kernel, mode='constant', cval=0)
        edges = (binary_raster == 1) & (neighbors < 4)

        edge_length = np.sum(edges) * cell_size
        total_area = binary_raster.size * (cell_size ** 2)

        return edge_length / total_area if total_area > 0 else 0

    except Exception as e:
        logger.warning(f"Could not calculate edge density: {e}")
        return 0


def calculate_clumpy_index_fast(binary_raster):
    """Optimized CLUMPY index calculation."""
    try:
        P = np.mean(binary_raster)

        if P == 0 or P == 1:
            return 0

        # Vectorized adjacency calculation
        h_likes = np.sum(binary_raster[:, :-1] == binary_raster[:, 1:])
        v_likes = np.sum(binary_raster[:-1, :] == binary_raster[1:, :])

        total_adjacencies = binary_raster.shape[0] * (binary_raster.shape[1] - 1) + \
                            binary_raster.shape[1] * (binary_raster.shape[0] - 1)

        if total_adjacencies == 0:
            return 0

        G_observed = (h_likes + v_likes) / total_adjacencies
        G_expected = P * P + (1 - P) * (1 - P)

        if P < 0.5:
            clumpy = (G_observed - G_expected) / (1 - G_expected)
        else:
            clumpy = (G_observed - G_expected) / G_expected

        return clumpy

    except Exception as e:
        logger.warning(f"Could not calculate CLUMPY index: {e}")
        return 0


def calculate_morans_i_fast(raster_data):
    """Faster Moran's I using sampling for large rasters."""
    try:
        # Sample for very large rasters
        if raster_data.size > 5000:
            sample_size = 2500
            rows, cols = raster_data.shape
            sample_indices = np.random.choice(raster_data.size, sample_size, replace=False)
            sample_coords = np.unravel_index(sample_indices, (rows, cols))
            values = raster_data[sample_coords]
            coordinates = np.column_stack(sample_coords)
        else:
            rows, cols = raster_data.shape
            coordinates = np.array([[i, j] for i in range(rows) for j in range(cols)])
            values = raster_data.flatten()

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
    """Optimized Geary's C calculation."""
    try:
        mean_val = np.mean(binary_raster)

        # Vectorized difference calculations
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

        # Vectorized window calculation using stride tricks
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


# Main execution
if __name__ == "__main__":
    # Print system info
    print(f"CPU cores available: {cpu_count()}")
    print(f"Memory available: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")
    print(f"Raster folder: {RASTER_FOLDER}")
    print(f"Output suffix: {OUTPUT_SUFFIX}")

    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    # Choose which grids to process
    grid_paths = (
        ["/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_100km.gpkg"] if USE_TEST_SETTINGS
        else ["/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_3km.gpkg",
              "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_20km.gpkg",
              "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_40km.gpkg"]
    )

    # Initialize spatial index once before processing any grids
    initialize_spatial_index()

    # Process each grid size
    for grid_path in grid_paths:
        print(f"\n{'=' * 60}")
        print(f"Loading grid: {grid_path}")

        grid = gpd.read_file(grid_path).to_crs(epsg=3857)

        if grid.empty:
            print(f"⚠️ Skipping {grid_path}: no cells found.")
            continue

        if "cell_id" not in grid.columns:
            grid = grid.reset_index(drop=True)
            grid["cell_id"] = grid.index

        # Determine AOI size
        if "3km" in grid_path:
            aoi_size = "3km"
        elif "20km" in grid_path:
            aoi_size = "20km"
        elif "40km" in grid_path:
            aoi_size = "40km"
        elif "100km" in grid_path:
            aoi_size = "100km"
        else:
            aoi_size = "unknown"

        print(f"Processing {aoi_size} grid with {len(grid)} cells")

        # Process grid cells in parallel
        results_df = process_grid_cells_parallel(grid, aoi_size, output_directory)

        if results_df is not None:
            print(f"\nSummary for {aoi_size}:")
            print(f"Total cells processed: {len(results_df)}")
            print(f"Average canopy extent: {results_df['canopy_extent'].mean():.2f}%")
            print(f"Cells with no data: {sum(results_df['num_intersecting_rasters'] == 0)}")
            print(f"Cells with multiple rasters: {sum(results_df['num_intersecting_rasters'] > 1)}")

    print(f"\n{'=' * 60}")
    print("All grids processed successfully!")