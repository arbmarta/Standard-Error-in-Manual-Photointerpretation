import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box

# Set paths
grid_path = '/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/AOI/grid_100km.gpkg'
tif_folder = '/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/Meta_CHM_Raw_test'
output_pdf = '/scratch/arbmarta/Standard-Error-in-Manual-Photointerpretation/output/100km_grid_and_tifs.pdf'

# Read the grid
grid = gpd.read_file(grid_path)
print(f"Grid CRS: {grid.crs}")
print("-" * 50)

# Find all TIF files in the folder
tif_files = list(Path(tif_folder).glob("*.tif"))
print(f"Found {len(tif_files)} TIF files")
print("-" * 50)

# Store CRS information
crs_dict = {}
bounding_boxes = []

# Check CRS of all TIF files and get bounding boxes
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        crs_str = str(src.crs)

        if crs_str not in crs_dict:
            crs_dict[crs_str] = []
        crs_dict[crs_str].append(tif_file.name)

        # Get bounding box
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bounding_boxes.append({'geometry': bbox, 'name': tif_file.name, 'crs': src.crs})

# Print CRS summary
print(f"Number of unique CRS: {len(crs_dict)}")

# Check if any CRS are different
if len(crs_dict) > 1:
    print("⚠️  WARNING: Multiple different CRS found!")
    for crs, files in crs_dict.items():
        print(f"\nCRS: {crs} - {len(files)} files")
else:
    print("✓ All TIF files have the same CRS")

print("-" * 50)

# Create GeoDataFrame of bounding boxes
bboxes_gdf = gpd.GeoDataFrame(bounding_boxes, crs=bounding_boxes[0]['crs'])

# Reproject bounding boxes to grid CRS if needed
if bboxes_gdf.crs != grid.crs:
    print(f"Reprojecting bounding boxes from {bboxes_gdf.crs} to {grid.crs}")
    bboxes_gdf = bboxes_gdf.to_crs(grid.crs)

# Create map with grid and bounding boxes
fig, ax = plt.subplots(figsize=(15, 12))

# Plot the grid
grid.boundary.plot(ax=ax, color='blue', linewidth=2, label='Grid (100km)', zorder=2)

# Plot bounding boxes
bboxes_gdf.boundary.plot(ax=ax, color='red', linewidth=1, alpha=0.7, label='TIF Bounding Boxes', zorder=1)
bboxes_gdf.plot(ax=ax, color='red', alpha=0.2, zorder=0)

ax.set_title(f'Grid and TIF Bounding Boxes (n={len(tif_files)})', fontsize=14, fontweight='bold')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()

# Save figure as PDF
plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nMap saved to: {output_pdf}")
plt.close()