import rasterio
import geopandas as gpd
import glob
import os
from shapely.geometry import box
from rasterio.mask import mask

# Configuration
USE_TEST_SETTINGS = True

if USE_TEST_SETTINGS:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw_test"
else:
    RASTER_FOLDER = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw"

GRID_PATH = "/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_100km.gpkg"

print("=" * 70)
print("DIAGNOSTIC SCRIPT - Finding the Issue")
print("=" * 70)

# Step 1: Check if raster folder exists and has files
print(f"\n1. Checking raster folder: {RASTER_FOLDER}")
if os.path.exists(RASTER_FOLDER):
    print("   ✓ Folder exists")
    raster_files = glob.glob(os.path.join(RASTER_FOLDER, "*.tif*"))
    print(f"   ✓ Found {len(raster_files)} raster files")
    if len(raster_files) > 0:
        print(f"   First 3 files: {[os.path.basename(f) for f in raster_files[:3]]}")
else:
    print("   ❌ Folder does not exist!")
    exit(1)

if len(raster_files) == 0:
    print("   ❌ No raster files found!")
    exit(1)

# Step 2: Check first raster properties
print(f"\n2. Examining first raster: {os.path.basename(raster_files[0])}")
with rasterio.open(raster_files[0]) as src:
    print(f"   CRS: {src.crs}")
    print(f"   Bounds: {src.bounds}")
    print(f"   Shape: {src.shape}")
    print(f"   Resolution: {src.res}")
    print(f"   Data type: {src.dtypes[0]}")

    # Read a sample of data
    sample_data = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
    print(f"   Sample data min: {sample_data.min()}")
    print(f"   Sample data max: {sample_data.max()}")
    print(f"   Sample data mean: {sample_data.mean():.2f}")
    print(f"   Non-zero values in sample: {(sample_data > 0).sum()}")

# Step 3: Check grid properties
print(f"\n3. Checking grid file: {GRID_PATH}")
if os.path.exists(GRID_PATH):
    print("   ✓ Grid file exists")
    grid = gpd.read_file(GRID_PATH)
    print(f"   Original CRS: {grid.crs}")
    print(f"   Number of cells: {len(grid)}")
    print(f"   Original bounds: {grid.total_bounds}")

    # Convert to same CRS as rasters
    grid_transformed = grid.to_crs(src.crs)
    print(f"   Transformed CRS: {grid_transformed.crs}")
    print(f"   Transformed bounds: {grid_transformed.total_bounds}")
else:
    print("   ❌ Grid file does not exist!")
    exit(1)

# Step 4: Check for spatial overlap
print("\n4. Checking for spatial overlap")
print(f"   Raster bounds: {src.bounds}")
print(f"   Grid bounds:   {grid_transformed.total_bounds}")

raster_box = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
grid_box = box(*grid_transformed.total_bounds)

if raster_box.intersects(grid_box):
    print("   ✓ Raster and grid DO overlap!")
else:
    print("   ❌ Raster and grid DO NOT overlap!")
    print("\n   This is the problem! Check:")
    print("   - Are rasters and grid in the same geographic area?")
    print("   - Are CRS transformations working correctly?")

# Step 5: Check individual cell intersections
print("\n5. Checking individual cell intersections with all rasters")
cell_intersection_counts = []

for idx, cell_row in grid_transformed.iterrows():
    cell_geom = cell_row.geometry
    intersecting_count = 0

    for raster_file in raster_files:
        with rasterio.open(raster_file) as rsrc:
            raster_bounds = rsrc.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)
            if cell_geom.intersects(raster_geom):
                intersecting_count += 1

    cell_intersection_counts.append(intersecting_count)

    if idx < 3:  # Show first 3 cells
        print(f"   Cell {idx}: {intersecting_count} intersecting rasters")

print(f"\n   Cells with 0 intersections: {cell_intersection_counts.count(0)}")
print(f"   Cells with >0 intersections: {sum(1 for c in cell_intersection_counts if c > 0)}")
print(f"   Max intersections per cell: {max(cell_intersection_counts)}")

# Step 6: Test actual data extraction
print("\n6. Testing data extraction from first intersecting cell")
for idx, cell_row in grid_transformed.iterrows():
    cell_geom = cell_row.geometry
    found_data = False

    for raster_file in raster_files[:5]:  # Test first 5 rasters
        with rasterio.open(raster_file) as rsrc:
            raster_bounds = rsrc.bounds
            raster_geom = box(raster_bounds.left, raster_bounds.bottom,
                              raster_bounds.right, raster_bounds.top)

            if cell_geom.intersects(raster_geom):
                try:
                    out_image, out_transform = mask(
                        rsrc, [cell_geom], crop=True, all_touched=False
                    )

                    if out_image[0].size > 0:
                        print(f"   ✓ Successfully extracted data from cell {idx}")
                        print(f"     Raster: {os.path.basename(raster_file)}")
                        print(f"     Extracted shape: {out_image[0].shape}")
                        print(f"     Min value: {out_image[0].min()}")
                        print(f"     Max value: {out_image[0].max()}")
                        print(f"     Mean value: {out_image[0].mean():.2f}")
                        print(f"     Non-zero pixels: {(out_image[0] > 0).sum()}")
                        print(f"     Values >= 2: {(out_image[0] >= 2).sum()}")
                        found_data = True
                        break
                except Exception as e:
                    print(f"     ❌ Error extracting data: {e}")

    if found_data:
        break

    if idx >= 10:  # Only check first 10 cells
        print("   ❌ No data found in first 10 cells")
        break

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)