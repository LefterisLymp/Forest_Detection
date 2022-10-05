import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
import collections

collections.Callable = collections.abc.Callable

import GEE_download

if (len(sys.argv) < 2 or int(sys.argv[1]) not in range(2,851)):
    print('Usage: python download_tiles.py m gridfile\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

if len(sys.argv) == 3:
    gridfile = sys.argv[2]
else:
    gridfile = os.path.join('train_data', 'grid_greece', 'grid_' + m + 'k.GeoJSON')

# read grid polygons from .GeoJSON file and convert crs
grid_polygons = geopandas.read_file(gridfile)
grid_pols = geopandas.GeoDataFrame.from_features(grid_polygons, crs='epsg:3035').to_crs('epsg:4326')

outputDirName = os.path.join('train_data', m + 'k', 'data_tiles')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

# function that calculates bounding box
def bbox(tile):
    x, y = tile.geometry.exterior.coords.xy
    return [min(x), min(y), max(x), max(y)]

# download images using GEE_download functions
for index, tile in grid_pols.iterrows():
    
    if os.path.exists(os.path.join(outputDirName, tile.cell_id, tile.cell_id + '.tif')):
        continue
    
    print('----------------------------------------------------')
    print('Tile no', index, ':', tile.cell_id)
    print('----------------------------------------------------')
        
    polygon = bbox(tile)

    # date range
    dates = ['2018-06-01', '2018-09-01']

    # name of the tile
    sitename = tile.cell_id

    # directory where the data will be stored
    filepath = outputDirName

    # put all the inputs into a dictionnary
    inputs = {'polygon': polygon, 'dates': dates, 'sitename': sitename, 'datapath': filepath, 'm': m}

    while True:
        try:
            GEE_download.retrieve_images(inputs)
            break
        except Exception as e:
            print(e)
            continue
