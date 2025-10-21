import geopandas as gpd
grid = gpd.read_file('AOI/grid_100km.gpkg')
print(f"Grid CRS type: {type(grid.crs)}")
print(f"Grid CRS: {grid.crs}")
print(f"Grid CRS string: '{str(grid.crs)}'")
