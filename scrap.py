import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from typing import List, Tuple, Set


# Check for the test tile values in the s3 bucket
import boto3; from botocore import UNSIGNED; from botocore.config import Config; tiles = [line.strip().split()[-1] for line in open('AOI/tiles_in_aoi_test.txt')]; s3_tiles = {p['Prefix'].rstrip('/').split('/')[-1] for page in boto3.client('s3', config=Config(signature_version=UNSIGNED)).get_paginator('list_objects_v2').paginate(Bucket='dataforgood-fb-data', Prefix='forests/v1/alsgedi_global_v6_float/chm/', Delimiter='/') for p in page.get('CommonPrefixes', [])}; print(f"AOI tiles: {len(tiles)}, S3 tiles: {len(s3_tiles)}, Found: {len(set(tiles) & s3_tiles)}, Missing: {sorted(set(tiles) - s3_tiles)}")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USE_TEST_SETTINGS = True  # Test this code using a low complexity, fast run

# S3 Configuration
BUCKET = 'dataforgood-fb-data'
PREFIX = 'forests/v1/alsgedi_global_v6_float/chm/'

# Initialize S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))


def load_aoi_tiles(test_mode: bool = True) -> List[str]:
    """Load tiles from AOI file"""
    filename = 'AOI/tiles_in_aoi_test.txt' if test_mode else 'AOI/tiles_in_aoi.txt'

    try:
        with open(filename, "r") as f:
            aoi = [line.strip().split()[-1] for line in f]
        logger.info(f"Loaded {len(aoi)} tiles from {filename}")
        return aoi
    except FileNotFoundError:
        logger.error(f"AOI file not found: {filename}")
        return []


def get_s3_tiles(bucket: str, prefix: str) -> Set[str]:
    """
    Retrieve all available tiles from S3 bucket

    Returns:
        Set of tile identifiers (quadkeys or tile names)
    """
    logger.info(f"Fetching available tiles from s3://{bucket}/{prefix}")

    tiles = set()
    paginator = s3.get_paginator('list_objects_v2')

    try:
        # Use paginator to handle large listings
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            # Get subdirectories (CommonPrefixes)
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    # Extract tile identifier from path
                    tile_path = prefix_obj['Prefix']
                    tile_id = tile_path.rstrip('/').split('/')[-1]
                    tiles.add(tile_id)

            # Also check Contents for direct files (in case tiles are files not folders)
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract potential tile identifier from filename
                    parts = key.replace(prefix, '').split('/')
                    if len(parts) > 0 and parts[0]:
                        tiles.add(parts[0])

        logger.info(f"Found {len(tiles)} tiles in S3")
        return tiles

    except Exception as e:
        logger.error(f"Error fetching S3 tiles: {e}")
        return set()


def verify_tiles_exist(aoi_tiles: List[str], s3_tiles: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Verify which AOI tiles exist in S3

    Returns:
        Tuple of (existing_tiles, missing_tiles)
    """
    aoi_set = set(aoi_tiles)

    existing = list(aoi_set & s3_tiles)
    missing = list(aoi_set - s3_tiles)

    return existing, missing


def print_verification_report(aoi_tiles: List[str], existing: List[str], missing: List[str]):
    """Print detailed verification report"""

    total = len(aoi_tiles)
    found = len(existing)
    not_found = len(missing)

    print("\n" + "=" * 70)
    print("TILE VERIFICATION REPORT")
    print("=" * 70)
    print(f"Total AOI tiles:     {total}")
    print(f"Found in S3:         {found} ({100 * found / total:.1f}%)")
    print(f"Missing from S3:     {not_found} ({100 * not_found / total:.1f}%)")
    print("=" * 70)

    if missing:
        print("\nMISSING TILES:")
        print("-" * 70)
        for tile in sorted(missing)[:20]:  # Show first 20
            print(f"  • {tile}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("\n✓ All AOI tiles found in S3!")

    if existing:
        print("\nSAMPLE OF EXISTING TILES:")
        print("-" * 70)
        for tile in sorted(existing)[:10]:  # Show first 10
            print(f"  • {tile}")
        if len(existing) > 10:
            print(f"  ... and {len(existing) - 10} more")

    print("=" * 70 + "\n")


def main():
    """Main verification workflow"""

    # Load AOI tiles
    aoi_tiles = load_aoi_tiles(test_mode=USE_TEST_SETTINGS)

    if not aoi_tiles:
        logger.error("No tiles loaded from AOI file. Exiting.")
        return

    # Get available tiles from S3
    s3_tiles = get_s3_tiles(BUCKET, PREFIX)

    if not s3_tiles:
        logger.error("No tiles found in S3. Check bucket/prefix configuration.")
        return

    # Verify tile existence
    existing, missing = verify_tiles_exist(aoi_tiles, s3_tiles)

    # Print report
    print_verification_report(aoi_tiles, existing, missing)

    # Save results
    if missing:
        output_file = 'missing_tiles.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(sorted(missing)))
        logger.info(f"Missing tiles saved to {output_file}")

    return existing, missing


if __name__ == "__main__":
    existing_tiles, missing_tiles = main()

    # Only proceed with download if all tiles exist
    if not missing_tiles:
        logger.info("All tiles verified. Ready to proceed with download.")
        # Uncomment to run download workflow:
        # from functions import main_download_workflow
        # main_download_workflow(existing_tiles)
    else:
        logger.warning(f"{len(missing_tiles)} tiles not found. Review before proceeding.")