# Enhanced Meta CHM downloader with robust file existence checking

import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from functions import main_download_workflow

USE_TEST_SETTINGS = True  # Test this code using a low complexity, fast run by setting this value to True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Bucket = 'dataforgood-fb-data'
Prefix = 'forests/v1/alsgedi_global_v6_float/chm/'

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=50))

# Use delimiter='/' to get folder-like structure
response = s3.list_objects_v2(Bucket=Bucket, Prefix=Prefix, Delimiter='/')

## ------------------------------ IDENTIFY META CHM TILES IN AOI ------------------------------
# region

# Identify which AOI to use
if USE_TEST_SETTINGS is False:
    with open('AOI/tiles_in_aoi.txt', "r") as f:
        aoi = [line.strip() for line in f]

else:
    with open('AOI/tiles_in_aoi_test.txt', "r") as f:
        aoi = [line.strip() for line in f]

print(f"Total number of tiles: {len(aoi)}")

# endregion

## ---------------------------------------------------- RUN SCRIPT -----------------------------------------------------
# Run the enhanced download workflow
if __name__ == "__main__":
    # Use the quadkeys identified from the AOI analysis
    main_download_workflow(aoi)
