import os
import glob
from pathlib import Path
import rasterio
from tqdm import tqdm


# ==============================================================================
# SECTION 1: RASTER REPROJECTION HELPER FUNCTION
# ==============================================================================

os.chdir("I:\Martin & Olson 2025")


def reproject_raster(input_path, output_path, target_crs='EPSG:5070'):
    """
    Reprojects a raster to the target CRS using Nearest Neighbor resampling.
    This is ideal for binary (categorical) data as it doesn't invent new pixel values.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    try:
        with rasterio.open(input_path) as src:
            # Calculate the transformation parameters for the new CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            # Copy the metadata from the source and update it for the new file
            kwargs = src.meta.copy()
            kwargs.update({
                'crs':       target_crs,
                'transform': transform,
                'width':     width,
                'height':    height,
                'compress':  'lzw'  # Add compression to save space
            })

            # Open the destination file and write the reprojected data
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
        return True, f"Reprojected {Path(input_path).name}"
    except Exception as e:
        return False, f"ERROR reprojecting {Path(input_path).name}: {e}"


# ==============================================================================
# SECTION 2: MAIN BATCH REPROJECTION SCRIPT
# ==============================================================================

def batch_reproject_from_binary():
    """
    Scans a directory of binary CHM files and reprojects them into the
    final analysis-ready projection.
    """
    # --- Configuration ---
    # **IMPORTANT**: Make sure this points to your folder of binary rasters
    binary_source_dir = "Meta CHM Binary"
    reprojected_dir = "Meta_CHM_Reprojected_5070"

    print("--- Starting Batch Reprojection from Binary Folder ---")
    print(f"Source Directory:      {Path(binary_source_dir).resolve()}")
    print(f"Destination Directory: {Path(reprojected_dir).resolve()}")

    # --- Setup ---
    Path(reprojected_dir).mkdir(parents=True, exist_ok=True)

    # Find all .tif files in the source binary directory
    binary_files = glob.glob(str(Path(binary_source_dir) / "*.tif"))

    if not binary_files:
        print(f"\nNo .tif files found in '{binary_source_dir}'. Exiting.")
        return

    print(f"\nFound {len(binary_files)} binary files to reproject.")

    successful_files = 0
    failed_files = 0

    # --- Processing Loop ---
    for binary_path_str in tqdm(binary_files, desc="Reprojecting files"):
        binary_path = Path(binary_path_str)
        quadkey = binary_path.stem

        # Define path for the final reprojected file
        reprojected_path = Path(reprojected_dir) / f"{quadkey}.tif"

        # Skip if the final file already exists to avoid re-work
        if reprojected_path.exists():
            continue

        try:
            # Reproject the binary raster
            success, message = reproject_raster(binary_path, reprojected_path)
            if not success:
                print(f"\n{message}")  # Print error message on failure
                failed_files += 1
            else:
                successful_files += 1
        except Exception as e:
            print(f"\nUnhandled error for {quadkey}.tif: {e}")
            failed_files += 1

    # --- Final Summary ---
    print("\n--- Batch Reprojection Complete ---")
    print(f"✅ Successfully processed: {successful_files}")
    print(f"❌ Failed to process:    {failed_files}")
    print(f"Total files in destination: {len(glob.glob(str(Path(reprojected_dir) / '*.tif')))}")
    print("-----------------------------------")


if __name__ == "__main__":
    batch_reproject_from_binary()
