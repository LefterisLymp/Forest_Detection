# script that creates {m} km square grid of Greece

import os
import sys
import geopandas
from geojson import Polygon, Feature

DATA_DIR = '../data'

if len(sys.argv) < 2 or not 1 < int(sys.argv[1]) <= 850:
    print('Usage: python grid_create.py m\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()

# tile size in km
m = int(sys.argv[1])

# longitude, latidude limits in km
x_limits = [5100, 6100]
y_limits = [1400, 2250]

# create grid as FeatureCollection
features = []
for i in range((x_limits[1] - x_limits[0])//m):
	for j in range((y_limits[1] - y_limits[0])//m):
		x = x_limits[0] + m * i
		y = y_limits[0] + m * j
		x_ = x_limits[0] + m * (i+1)
		y_ = y_limits[0] + m * (j+1)
		cell_id = str(x) + '_' + str(y)
		coords = [[(1000.0*x, 1000.0*y), (1000.0*x, 1000.0*y_), (1000.0*x_, 1000.0*y_), (1000.0*x_, 1000.0*y), (1000.0*x, 1000.0*y)]]
		features.append(Feature(properties={"cell_id": cell_id}, geometry=Polygon(coords)))
grid = geopandas.GeoDataFrame.from_features(features, crs='EPSG:3035')

# save file
grid_dir = os.path.join(DATA_DIR, str(m) + 'k', 'grid')
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)
grid.to_file(os.path.join(grid_dir, 'grid_full.GeoJSON'), driver='GeoJSON')

print('Grid created with', len(grid), 'tiles of size', m, 'x', m, 'km.')
