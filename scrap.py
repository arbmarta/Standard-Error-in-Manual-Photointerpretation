import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import rasterio
import rasterio.mask
from shapely.geometry import Point, box
from tqdm import tqdm
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# HPC-OPTIMIZED PARALLEL PROCESSING FUNCTIONS
# ============================================

def process_grid_cell_chunk(chunk_data, relative_points, spatial_index_path=None, raster_info_path=None):
    """
    Process a chunk of grid cells in parallel.

    Parameters:
    -----------
    chunk_data : tuple
        (chunk_index, grid_cells_subset, crs)
    relative_points : list
        List of (x, y) relative positions for points
    spatial_index_path : str
        Path to serialized spatial index (if available)
    raster_info_path : str
        Path to serialized raster info (if available)

    Returns:
    --------
    dict : Results containing points, grid_ids, point_ids, has_raster
    """
    chunk_idx, grid_cells, crs = chunk_data

    # Load spatial index if provided
    spatial_index, raster_info = None, None
    if raster_info_path and os.path.exists(raster_info_path):
        with open(raster_info_path, 'r') as f:
            raster_info = json.load(f)
        # Note: R-tree index needs special handling for multiprocessing
        # For simplicity, we'll use the raster_info for brute force checking

    chunk_points = []
    chunk_grid_ids = []
    chunk_point_ids = []
    chunk_has_raster = []

    for idx, row in grid_cells.iterrows():
        grid_cell = row.geometry
        grid_id = row['grid_id']

        # Check raster coverage
        has_raster = False
        if raster_info:
            has_raster = check_raster_coverage_fast(grid_cell, raster_info)

        minx, miny, maxx, maxy = grid_cell.bounds
        cell_width = maxx - minx
        cell_height = maxy - miny

        for point_idx, (rel_x, rel_y) in enumerate(relative_points):
            abs_x = minx + (rel_x * cell_width)
            abs_y = miny + (rel_y * cell_height)
            point = Point(abs_x, abs_y)

            if grid_cell.contains(point) or grid_cell.intersects(point):
                chunk_points.append(point)
                chunk_grid_ids.append(grid_id)
                chunk_point_ids.append(f"{grid_id}_p{point_idx}")
                chunk_has_raster.append(has_raster)

    return {
        'points': chunk_points,
        'grid_ids': chunk_grid_ids,
        'point_ids': chunk_point_ids,
        'has_raster': chunk_has_raster,
        'chunk_idx': chunk_idx
    }


def check_raster_coverage_fast(grid_cell, raster_info):
    """
    Fast check if grid cell intersects any raster using serialized bounds.

    Parameters:
    -----------
    grid_cell : shapely.geometry
        Grid cell geometry
    raster_info : dict
        Dictionary of raster information with bounds

    Returns:
    --------
    bool : True if grid cell intersects any raster
    """
    cell_bounds = grid_cell.bounds
    minx, miny, maxx, maxy = cell_bounds

    for raster_data in raster_info.values():
        bounds = raster_data['bounds']
        # Quick bounding box intersection check
        if not (bounds['right'] < minx or bounds['left'] > maxx or
                bounds['top'] < miny or bounds['bottom'] > maxy):
            return True

    return False


def generate_systematic_sample_points_parallel(grid_gdf, points_per_cell,
                                               raster_folder=None,
                                               n_workers=None,
                                               chunk_size=100):
    """
    Generate systematic sample points using parallel processing.

    Parameters:
    -----------
    grid_gdf : GeoDataFrame
        Grid cells to sample
    points_per_cell : int
        Number of sample points per grid cell
    raster_folder : str, optional
        Path to raster folder for coverage validation
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    chunk_size : int
        Number of grid cells per chunk for parallel processing

    Returns:
    --------
    GeoDataFrame : Sample points with metadata
    """
    print(f"🚀 HPC MODE: GENERATING {points_per_cell:,} SYSTEMATIC SAMPLE POINTS PER GRID CELL...")
    print(f"   Processing {len(grid_gdf):,} grid cells...")

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    print(f"   Using {n_workers} parallel workers")

    # Calculate grid layout for points
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

    print(f"   Using {len(relative_points)} points per cell")

    # Build and serialize spatial index
    raster_info_path = None
    if raster_folder:
        print(f"\n   Building spatial index for raster validation...")
        raster_info = build_raster_info_dict(raster_folder)

        # Save to temporary file for worker processes
        raster_info_path = "temp_raster_info.json"
        with open(raster_info_path, 'w') as f:
            json.dump(raster_info, f)
        print(f"   Saved raster info to {raster_info_path}")

    # Split grid into chunks
    n_chunks = math.ceil(len(grid_gdf) / chunk_size)
    chunks = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(grid_gdf))
        chunk_gdf = grid_gdf.iloc[start_idx:end_idx].copy()
        chunks.append((i, chunk_gdf, grid_gdf.crs))

    print(f"   Split into {n_chunks} chunks of ~{chunk_size} cells each")

    # Process chunks in parallel
    print(f"\n   Processing chunks in parallel...")

    process_func = partial(
        process_grid_cell_chunk,
        relative_points=relative_points,
        raster_info_path=raster_info_path
    )

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, chunks),
            total=n_chunks,
            desc="Processing chunks"
        ))

    # Combine results
    print(f"\n   Combining results from {len(results)} chunks...")
    all_points = []
    all_grid_ids = []
    all_point_ids = []
    all_has_raster = []

    for result in results:
        all_points.extend(result['points'])
        all_grid_ids.extend(result['grid_ids'])
        all_point_ids.extend(result['point_ids'])
        all_has_raster.extend(result['has_raster'])

    # Clean up temporary files
    if raster_info_path and os.path.exists(raster_info_path):
        os.remove(raster_info_path)

    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame({
        'point_id': all_point_ids,
        'grid_id': all_grid_ids,
        'sample_type': 'systematic',
        'points_per_cell': points_per_cell,
        'has_raster_data': all_has_raster
    }, geometry=all_points, crs=grid_gdf.crs)

    print(f"\n✅ Generated {len(points_gdf):,} total sample points")
    print(f"   Points per cell: {len(points_gdf) / len(grid_gdf):.1f} (target: {points_per_cell})")
    if raster_folder:
        points_with_data = sum(all_has_raster)
        print(f"   Points with raster data: {points_with_data:,} ({100 * points_with_data / len(points_gdf):.1f}%)")

    return points_gdf


def build_raster_info_dict(raster_folder):
    """
    Build a serializable dictionary of raster information.

    Parameters:
    -----------
    raster_folder : str
        Path to folder containing raster files

    Returns:
    --------
    dict : Raster information with serializable bounds
    """
    raster_info = {}
    raster_pattern = os.path.join(raster_folder, "*.tif*")
    raster_files = glob.glob(raster_pattern)

    print(f"   Building raster info for {len(raster_files)} files...")

    for i, raster_path in enumerate(tqdm(raster_files, desc="Indexing rasters")):
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds

                raster_info[str(i)] = {
                    'path': raster_path,
                    'bounds': {
                        'left': bounds.left,
                        'bottom': bounds.bottom,
                        'right': bounds.right,
                        'top': bounds.top
                    }
                }
        except Exception as e:
            logger.warning(f"Could not index {raster_path}: {e}")
            continue

    print(f"   Indexed {len(raster_info)} rasters")
    return raster_info


# ============================================
# ARRAY JOB SUPPORT FOR HPC CLUSTERS
# ============================================

def save_grid_chunks_for_array_jobs(grid_gdf, output_dir, chunks_per_job=50):
    """
    Save grid chunks for SLURM/PBS array jobs.

    Parameters:
    -----------
    grid_gdf : GeoDataFrame
        Grid to split into chunks
    output_dir : str
        Directory to save chunks
    chunks_per_job : int
        Number of grid cells per job

    Returns:
    --------
    int : Number of array jobs needed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_jobs = math.ceil(len(grid_gdf) / chunks_per_job)

    for job_id in range(n_jobs):
        start_idx = job_id * chunks_per_job
        end_idx = min((job_id + 1) * chunks_per_job, len(grid_gdf))

        chunk_gdf = grid_gdf.iloc[start_idx:end_idx].copy()
        chunk_file = output_path / f"grid_chunk_{job_id:04d}.gpkg"
        chunk_gdf.to_file(chunk_file, driver="GPKG")

    print(f"✅ Saved {n_jobs} grid chunks to {output_dir}")
    print(f"   Each chunk contains ~{chunks_per_job} grid cells")
    print(f"\nTo run as array job:")
    print(f"   #SBATCH --array=0-{n_jobs - 1}")
    print(f"   python process_chunk.py --chunk-id $SLURM_ARRAY_TASK_ID")

    return n_jobs


def process_single_chunk_for_array_job(chunk_id, chunks_dir, output_dir,
                                       points_per_cell, raster_folder=None):
    """
    Process a single chunk (for use in array jobs).

    Parameters:
    -----------
    chunk_id : int
        Array job task ID
    chunks_dir : str
        Directory containing grid chunks
    output_dir : str
        Directory to save results
    points_per_cell : int
        Number of points per grid cell
    raster_folder : str, optional
        Path to raster folder
    """
    chunk_file = Path(chunks_dir) / f"grid_chunk_{chunk_id:04d}.gpkg"

    if not chunk_file.exists():
        logger.error(f"Chunk file not found: {chunk_file}")
        return

    print(f"Processing chunk {chunk_id}: {chunk_file}")
    grid_chunk = gpd.read_file(chunk_file)

    # Process this chunk (single-threaded, as parallelism is at job level)
    points = generate_systematic_sample_points_parallel(
        grid_chunk,
        points_per_cell=points_per_cell,
        raster_folder=raster_folder,
        n_workers=1  # Single worker per array job
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"points_chunk_{chunk_id:04d}.gpkg"
    points.to_file(output_file, driver="GPKG")

    print(f"✅ Saved {len(points)} points to {output_file}")


def merge_chunk_results(chunks_dir, output_file):
    """
    Merge results from array jobs.

    Parameters:
    -----------
    chunks_dir : str
        Directory containing chunk results
    output_file : str
        Output file path for merged results
    """
    chunk_pattern = os.path.join(chunks_dir, "points_chunk_*.gpkg")
    chunk_files = sorted(glob.glob(chunk_pattern))

    print(f"Merging {len(chunk_files)} chunk results...")

    all_points = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunk_gdf = gpd.read_file(chunk_file)
        all_points.append(chunk_gdf)

    merged = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))
    merged.to_file(output_file, driver="GPKG")

    print(f"✅ Merged {len(merged)} points to {output_file}")
    return merged


# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HPC-optimized sample point generation")
    parser.add_argument("--mode", choices=["parallel", "array-prep", "array-process", "array-merge"],
                        default="parallel", help="Processing mode")
    parser.add_argument("--grid-file", default="AOI/grid_10km.gpkg")
    parser.add_argument("--raster-folder", default="Meta CHM Binary")
    parser.add_argument("--points-per-cell", type=int, default=10)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--chunks-dir", default="grid_chunks")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-file", default="sample_points_10km.gpkg")

    args = parser.parse_args()

    if args.mode == "parallel":
        # Standard parallel processing on single node
        grid_gdf = gpd.read_file(args.grid_file)
        sample_points = generate_systematic_sample_points_parallel(
            grid_gdf,
            points_per_cell=args.points_per_cell,
            raster_folder=args.raster_folder,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size
        )
        sample_points.to_file(args.output_file, driver="GPKG")
        print(f"\n✅ Saved {len(sample_points)} points to {args.output_file}")

    elif args.mode == "array-prep":
        # Prepare chunks for array jobs
        grid_gdf = gpd.read_file(args.grid_file)
        save_grid_chunks_for_array_jobs(
            grid_gdf,
            output_dir=args.chunks_dir,
            chunks_per_job=args.chunk_size
        )

    elif args.mode == "array-process":
        # Process single chunk (called by array job)
        if args.chunk_id is None:
            raise ValueError("--chunk-id required for array-process mode")
        process_single_chunk_for_array_job(
            args.chunk_id,
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
            points_per_cell=args.points_per_cell,
            raster_folder=args.raster_folder
        )

    elif args.mode == "array-merge":
        # Merge results from array jobs
        merge_chunk_results(args.output_dir, args.output_file)