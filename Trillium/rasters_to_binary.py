# For use on Trillium HPC - Convert CHM rasters to 1-bit binary

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test mode toggle
USE_TEST_SETTINGS = False


def convert_to_binary(input_path, output_path):
    """
    Convert CHM raster to 1-bit binary:
    - Values 0 or 1 become 0
    - Values >= 2 become 1

    Returns: (success: bool, message: str)
    """
    try:
        with rasterio.open(input_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            source_nodata = src.nodata

            # Create binary array: values >= 2 become 1, otherwise 0
            binary_data = (data >= 2).astype(np.uint8)

            # Preserve NoData values if they exist
            if source_nodata is not None:
                binary_data[data == source_nodata] = source_nodata

            # Update metadata for 1-bit binary output
            meta.update({
                'dtype': 'uint8',
                'count': 1,
                'compress': 'lzw',
                'nbits': 1  # Optimizes storage for 1-bit data
            })

            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Write binary raster
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(binary_data, 1)

        filename = Path(output_path).name
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        return True, f"✓ {filename} ({size_mb:.2f} MB)"

    except Exception as e:
        filename = Path(input_path).name
        logger.error(f"Failed to convert {filename}: {e}")
        return False, f"✗ {filename}: {str(e)}"


def convert_all_rasters(input_dir, output_dir, max_workers=192):
    """
    Convert all .tif files in input_dir to binary format in output_dir
    """
    logger.info(f"Starting conversion from {input_dir} to {output_dir}")

    # Find all .tif files in input directory
    input_path = Path(input_dir)
    tif_files = list(input_path.glob("*.tif"))

    if not tif_files:
        logger.error(f"No .tif files found in {input_dir}")
        return

    logger.info(f"Found {len(tif_files)} raster files to convert")

    # Check which files already exist and are valid
    files_to_process = []
    already_done = 0

    for input_file in tif_files:
        output_file = Path(output_dir) / input_file.name

        if output_file.exists() and output_file.stat().st_size > 1024:
            already_done += 1
        else:
            files_to_process.append((input_file, output_file))

    logger.info(f"{already_done} files already converted")
    logger.info(f"{len(files_to_process)} files need conversion")

    if not files_to_process:
        logger.info("All files already converted!")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert files in parallel
    successful = 0
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_to_binary, str(input_file), str(output_file)): (input_file, output_file)
            for input_file, output_file in files_to_process
        }

        with tqdm(total=len(files_to_process), desc="Converting to binary") as pbar:
            for future in as_completed(futures):
                input_file, output_file = futures[future]

                try:
                    success, message = future.result()
                    if success:
                        successful += 1
                    else:
                        failed.append((input_file.stem, message))
                        logger.error(message)
                except Exception as e:
                    failed.append((input_file.stem, str(e)))
                    logger.error(f"Exception converting {input_file.name}: {e}")
                finally:
                    pbar.update(1)

    # Report results
    logger.info(f"\nConversion complete: {successful} successful, {len(failed)} failed")

    if failed:
        report_path = Path("conversion_failures.txt")
        with open(report_path, 'w') as f:
            f.write("Filename\tReason\n")
            for filename, reason in failed:
                f.write(f"{filename}\t{reason}\n")
        logger.warning(f"Failure details saved to {report_path}")


if __name__ == "__main__":
    # Set directories based on test mode
    if USE_TEST_SETTINGS:
        input_dir = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw_test"
        output_dir = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Binary_test"
    else:
        input_dir = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw"
        output_dir = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Binary"

    logger.info(f"USE_TEST_SETTINGS = {USE_TEST_SETTINGS}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Run conversion
    convert_all_rasters(input_dir, output_dir, max_workers=192)
