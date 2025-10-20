from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import logging
import numpy as np
import os
from pathlib import Path
import queue
import rasterio
import rasterio.mask
import time
from tqdm import tqdm

USE_TEST_SETTINGS = False  # Test this code using a low complexity, fast run by setting this value to True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Bucket = 'dataforgood-fb-data'
Prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

# Use delimiter='/' to get folder-like structure
response = s3.list_objects_v2(Bucket='dataforgood-fb-data', Prefix='forests/v1/alsgedi_global_v6_float/chm/', Delimiter='/')



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------- Functions for download_meta.py -------------------------------------------
#region

def main_download_workflow(quadkeys,
                       raw_dir="Meta_CHM_Raw",
                       binary_dir="Meta_CHM_Binary",
                       reprojected_dir="Meta_CHM_Reprojected_5070",
                       output_dir="reports"):
    """
    Main workflow to download, convert, reproject, and verify raster data with robust tracking.

    This function orchestrates a multi-threaded pipeline:
    1. Validates which files exist on S3.
    2. Checks for already processed files to avoid re-work.
    3. Downloads raw files (EPSG:3857).
    4. Converts raw files to binary (canopy/no canopy).
    5. Reprojects binary files to the analysis CRS (EPSG:5070).
    6. Verifies the integrity of the final reprojected files.
    7. Generates a detailed report of any failures.
    """
    logger.info("Starting robust download and processing workflow.")

    # --- Step 1: Setup S3 Client and Validate Remote Files ---
    logger.info("Step 1: Validating file existence in S3...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))
    bucket_name = 'dataforgood-fb-data'
    base_prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

    valid_paths, invalid_keys, file_info = validate_quadkey_paths(
        quadkeys, bucket_name, base_prefix, s3
    )

    logger.info(f"Found {len(valid_paths)} valid files on S3.")
    if invalid_keys:
        logger.warning(f"{len(invalid_keys)} quadkeys correspond to files not found on S3.")

    if not valid_paths:
        logger.critical("No valid files to download. Exiting workflow.")
        return

    # --- Step 2: Check Local Files and Determine Work to Be Done ---
    logger.info("Step 2: Checking for already processed local files...")
    files_to_process = []
    already_done = 0
    for s3_path in valid_paths:
        key = s3_path.replace(f"s3://{bucket_name}/", "")
        quadkey = file_info[key]['quadkey']
        final_reprojected_path = Path(reprojected_dir) / f"{quadkey}.tif"

        is_valid, _ = verify_raster(final_reprojected_path)
        if is_valid:
            already_done += 1
        else:
            files_to_process.append((s3_path, key))

    logger.info(f"{already_done} files are already fully processed and verified.")
    logger.info(f"{len(files_to_process)} files need to be downloaded and processed.")

    if not files_to_process:
        logger.info("Workflow complete. All required files are already processed.")
        return

    # --- Step 3: Setup Directories and Parallel Processing ---
    logger.info("Step 3: Preparing for parallel download and conversion...")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(binary_dir).mkdir(parents=True, exist_ok=True)
    Path(reprojected_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    conversion_queue = queue.Queue()
    HEIGHT_THRESHOLD = 5  # meters

    failed_downloads = []
    failed_conversions = []

    num_download_workers = 10
    num_conversion_workers = max(1, os.cpu_count() - 1)

    with ThreadPoolExecutor(max_workers=num_conversion_workers, thread_name_prefix='Converter') as conversion_executor, \
         ThreadPoolExecutor(max_workers=num_download_workers, thread_name_prefix='Downloader') as download_executor:

        for _ in range(num_conversion_workers):
            conversion_executor.submit(conversion_worker, conversion_queue, binary_dir,
                                       reprojected_dir, HEIGHT_THRESHOLD, failed_conversions)

        future_to_path = {}
        for s3_path, key in files_to_process:
            local_path = Path(raw_dir) / f"{Path(key).stem}.tif"
            future = download_executor.submit(download_tile_boto3, s3, bucket_name, key, str(local_path))
            future_to_path[future] = str(local_path)

        with tqdm(total=len(files_to_process), desc="Downloading & Processing") as pbar:
            for future in as_completed(future_to_path):
                # Retrieve the clean path associated with the completed future.
                local_path_str = future_to_path[future]
                quadkey = Path(local_path_str).stem

                try:
                    success, message = future.result()
                    if success:
                        # On successful download, put the verified local path on the queue.
                        conversion_queue.put(local_path_str)
                    else:
                        failed_downloads.append((quadkey, "Download failed", message))
                except Exception as exc:
                    failed_downloads.append((quadkey, "Download exception", str(exc)))
                finally:
                    pbar.update(1)

        logger.info("All downloads complete. Waiting for conversions to finish...")
        conversion_queue.join()

        for _ in range(num_conversion_workers):
            conversion_queue.put(None)

    # --- Step 4: Collate and Report Failures ---
    logger.info("Step 4: Generating failure report...")
    all_failures = failed_downloads + failed_conversions
    if all_failures:
        report_path = Path(output_dir) / "failure_report.csv"
        logger.warning(f"{len(all_failures)} tasks failed. See {report_path} for details.")
        with open(report_path, 'w') as f:
            f.write("Quadkey,FailureStage,Reason\n")
            for quadkey, stage, reason in all_failures:
                f.write(f'"{quadkey}","{stage}","{reason.strip()}"\n')
    else:
        logger.info("All tasks completed successfully.")

    logger.info("Workflow finished.")


#endregion

# -------------------------------------------------- Helper Functions --------------------------------------------------

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

def conversion_worker(q, binary_dir, reprojected_dir, threshold, failed_conversions_list):
    """
    Worker pulling from a queue to convert, reproject, verify, and clean up files.
    This function is designed to run in a separate thread.
    """
    while True:
        raw_path_str = q.get()
        if raw_path_str is None:
            break  # Sentinel value to stop the worker.

        raw_path = Path(raw_path_str)
        quadkey = raw_path.stem

        # Define paths for all stages
        binary_path = Path(binary_dir) / f"{quadkey}.tif"
        final_reprojected_path = Path(reprojected_dir) / f"{quadkey}.tif"
        temp_reprojected_path = Path(reprojected_dir) / f"{quadkey}_temp.tif"

        try:
            # Stage 1: Create Binary Raster
            success, msg = create_binary_raster(str(raw_path), str(binary_path), threshold)
            if not success:
                failed_conversions_list.append((quadkey, "Binary creation", msg))
                continue

            # Stage 2: Reproject to a temporary file
            success, msg = reproject_raster(str(binary_path), str(temp_reprojected_path))
            if not success:
                failed_conversions_list.append((quadkey, "Reprojection", msg))
                continue

            # Stage 3: Verify the reprojected temporary file
            is_valid, msg = verify_raster(temp_reprojected_path)
            if not is_valid:
                failed_conversions_list.append((quadkey, "Verification", msg))
                continue

            # Stage 4: If all checks pass, rename the temp file to its final name
            os.rename(temp_reprojected_path, final_reprojected_path)
            logger.info(f"Successfully processed and verified: {quadkey}.tif")

        except Exception as e:
            failed_conversions_list.append((quadkey, "Unhandled worker exception", str(e)))
        finally:
            # Clean up intermediate files regardless of outcome
            if binary_path.exists():
                os.remove(binary_path)
            if raw_path.exists():
                os.remove(raw_path)
            if temp_reprojected_path.exists():
                os.remove(temp_reprojected_path)
            q.task_done()

def verify_raster(file_path, expected_crs='EPSG:5070'):
    """
    Verifies a raster for readability, correct CRS, and valid data content.
    Returns: (is_valid: bool, message: str)
    """
    try:
        p_file_path = Path(file_path)
        if not p_file_path.exists() or p_file_path.stat().st_size < 1024:
            return False, "File does not exist or is too small."

        with rasterio.open(file_path) as src:
            if src.crs != rasterio.crs.CRS.from_string(expected_crs):
                return False, f"CRS mismatch. Is {src.crs}, expected {expected_crs}."

            data = src.read(1, masked=True)
            if np.ma.count(data) == 0:
                return False, "Raster contains no valid data (all NoData)."

    except Exception as e:
        return False, f"File is corrupt or unreadable: {str(e)}"
    return True, "File is valid."

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
                'compress': 'lzw',
                'nbits': 1  # Ensures output is optimized for 1-bit data
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


#endregion

## ------------------------------ IDENTIFY META CHM TILES IN AOI ------------------------------
# region

# Identify which AOI to use
if USE_TEST_SETTINGS is False:
    with open('AOI/tiles_in_aoi.txt', "r") as f:
        aoi = [line.strip().split()[-1] for line in f]

else:
    with open('AOI/tiles_in_aoi_test.txt', "r") as f:
        aoi = [line.strip().split()[-1] for line in f]


print(f"Total number of tiles: {len(aoi)}")

# endregion

## ---------------------------------------------------- RUN SCRIPT -----------------------------------------------------
# Run the enhanced download workflow
if __name__ == "__main__":
    # Use the quadkeys identified from the AOI analysis
    main_download_workflow(aoi)