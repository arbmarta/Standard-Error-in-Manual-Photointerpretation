# Change Log for master branch

October 21:
1. rasters_to_binary.py updated for use on Trillium HPC with reduced I/O demand. Decreased max_workers from 64 to 16.
2. rasters_to_binary.py changed to remove compression. Total size of project on scratch ~2.2 TB

October 20: 
1. AOI scripts updated to use EPSG 3857, matching the Meta tiles.
2. download_meta.py and rasters_to_binary.py optimized for running on Trillium HPC
3. Meta CHMs successfully downloaded using 

# Files
## Folders
### AOI
This folder contains the scripts and datasets for generating grids over the contiguous United States (CONUS), which is our area of interest.

**.py Files:**
1. **CONUS.py:** Loads the US states shapefile from TIGER (US Census Bureau), removes states and territories outside the contiguous United States (AK, HI, PR, GU, VI, MP, AS), merges the remaining states, and exports conus.gpkg. This file provides the AOI polygon used for further processing.
2. **grid_generator.py:** Uses conus.gpkg to generate square grid datasets across the contiguous United States with cell size lengths of 3, 24, and 54 km. Each grid includes a unique cell_id and centroid coordinates in both projected (EPSG:5070) and geographic (EPSG:4326) coordinates. Only cells fully contained within the AOI are retained.

**Datasets:**
1. **conus.gpkg:** Geopackage of the contiguous United States AOI.
2. **tiles_in_aoi.geojson:** AOI-filtered Meta CHM tiles used to define grid extent.
3. **grid_3km.gpkg:** Generated grid cells with length and width of 3 km (must be created locally).
4. **grid_24km.gpkg:** Generated grid cells with length and width of 24 km.
5. **grid_54km.gpkg:** Generated grid cells with length and width of 54 km.
6. **tiles_in_aoi.txt:** List of tile IDs contained within the AOI.
7. **Tiles in AOI.pdf:** Map showing tiles and AOI boundaries.

### Trillium
This folder contains the scripts (.py and .sh) used for processing on SciNet's Trillium HPC (University of Toronto).

**.py Files**
1. **download Meta.py:** Automates the download and processing of the Meta CHM tiles from the S3 bucket, 
first validating that each file exists in the cloud. If the tiles are not already locally stored, it downloads them in parallel.
2. **rasters_to_binary.py:** Converts downloaded Meta tiles in Meta_CHM_Raw to binary 1-bit binary rasters (0/1) based on a height threshold - heights above 2 m take the value of 1, else 0.

**Other Files**
1. **rasters_to_binary.sh:** The bash file associated with rasters_to_binary.py.
2. **helpers:** Commands used on Trillium HPC.

## Scripts
