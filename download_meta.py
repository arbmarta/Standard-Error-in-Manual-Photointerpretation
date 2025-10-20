# For use on Trillium HPC datamover nodes

from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import logging
from pathlib import Path
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test mode toggle
USE_TEST_SETTINGS = True

# Download a single file from S3 with retry logic and integrity checking.
def download_file_from_s3(s3_client, bucket, key, local_path, expected_size, max_retries=3):

    local_path = Path(local_path)
    filename = local_path.name

    # Check if file already exists with correct size
    if local_path.exists():
        local_size = local_path.stat().st_size
        if local_size == expected_size:
            return True, f"✓ {filename} (already exists)"
        else:
            logger.warning(f"Removing incomplete file: {filename}")
            local_path.unlink()

    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with retries
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket, key, str(local_path))

            # Verify file size
            actual_size = local_path.stat().st_size
            if actual_size == expected_size:
                size_mb = actual_size / (1024 * 1024)
                return True, f"✓ {filename} ({size_mb:.1f} MB)"
            else:
                local_path.unlink()
                return False, f"✗ {filename}: Size mismatch"

        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"✗ {filename}: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

    return False, f"✗ {filename}: Max retries exceeded"

# Check which files exist in S3 and get their metadata.
def check_s3_files_exist(s3_client, bucket, keys, max_workers=10):

    def check_single_file(key):
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            return key, (True, response['ContentLength'], response['LastModified'])
        except s3_client.exceptions.NoSuchKey:
            return key, (False, 0, None)
        except Exception as e:
            logger.warning(f"Error checking {key}: {e}")
            return key, (False, 0, None)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_single_file, key) for key in keys]

        for future in tqdm(as_completed(futures), total=len(keys), desc="Checking S3 files"):
            key, result = future.result()
            results[key] = result

    return results

# Download raster files from S3 for given quadkeys.
def download_rasters(quadkeys, output_dir="Meta_CHM_Raw", max_workers=10):

    logger.info(f"Starting download of {len(quadkeys)} rasters")

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED, max_pool_connections=50)
    )

    # Step 1: Build expected S3 keys
    expected_keys = [f"forests/v1/alsgedi_global_v6_float/chm/{qk}.tif" for qk in quadkeys]

    # Step 2: Check which files exist on S3
    logger.info("Validating files on S3...")
    file_info = check_s3_files_exist(s3_client, 'dataforgood-fb-data', expected_keys)

    # Step 3: Separate valid from invalid files
    valid_files = []
    invalid_files = []

    for key, (exists, size, modified) in file_info.items():
        if exists:
            quadkey = Path(key).stem
            local_path = Path(output_dir) / f"{quadkey}.tif"
            valid_files.append((key, local_path, size))
        else:
            invalid_files.append(key)

    logger.info(f"Found {len(valid_files)} valid files on S3")
    if invalid_files:
        logger.warning(f"{len(invalid_files)} files not found on S3")

    if not valid_files:
        logger.error("No valid files to download")
        return

    # Step 4: Download files in parallel
    logger.info(f"Downloading with {max_workers} workers...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_file_from_s3,
                s3_client,
                'dataforgood-fb-data',
                key,
                local_path,
                size
            ): (key, local_path)
            for key, local_path, size in valid_files
        }

        with tqdm(total=len(valid_files), desc="Downloading") as pbar:
            for future in as_completed(futures):
                key, local_path = futures[future]
                quadkey = Path(key).stem

                try:
                    success, message = future.result()
                    if success:
                        successful += 1
                    else:
                        failed.append((quadkey, message))
                        logger.error(message)
                except Exception as e:
                    failed.append((quadkey, str(e)))
                    logger.error(f"Exception downloading {quadkey}: {e}")
                finally:
                    pbar.update(1)

    # Step 5: Report results
    logger.info(f"\nDownload complete: {successful} successful, {len(failed)} failed")

    if failed:
        report_path = Path("download_failures.txt")
        with open(report_path, 'w') as f:
            f.write("Quadkey\tReason\n")
            for quadkey, reason in failed:
                f.write(f"{quadkey}\t{reason}\n")
        logger.warning(f"Failure details saved to {report_path}")


if __name__ == "__main__":
    # Load quadkeys from AOI file
    if USE_TEST_SETTINGS:
        aoi_file = 'AOI/tiles_in_aoi_test.txt'
    else:
        aoi_file = 'AOI/tiles_in_aoi.txt'

    with open(aoi_file, "r") as f:
        quadkeys = [line.strip().split()[-1] for line in f]

    logger.info(f"Loaded {len(quadkeys)} quadkeys from {aoi_file}")

    # Download rasters
    download_rasters(quadkeys, output_dir="Meta_CHM_Raw", max_workers=10)