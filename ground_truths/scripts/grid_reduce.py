# script that reads square grid of Greece and filters only those regions containing forests (from .GeoJSON file)

import os
import sys
import geopandas
import pandas as pd

DATA_DIR = '/data/data1/users/lefterislymp/ground_truths/data'

if len(sys.argv) < 2 or not 1 < int(sys.argv[1]) <= 850:
    print('Usage: python grid_reduce.py m [beach_file]\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()

# tile size in km
m = int(sys.argv[1])

# file paths
beach_files = os.listdir(DATA_DIR + '/osm')

grid_file = os.path.join(DATA_DIR, str(m) + 'k', 'grid', 'grid_full.GeoJSON')
dest_file = os.path.join(DATA_DIR, str(m) + 'k', 'grid', 'grid.GeoJSON')

#load from files
grid_pols = geopandas.read_file(grid_file)
geodf_list = []

for beach_file in beach_files:
    beach_file = os.path.join(DATA_DIR, 'osm', beach_file)
    beach_pols = geopandas.read_file(beach_file)
    beach_pols = geopandas.GeoDataFrame.from_features(beach_pols, crs='epsg:4326').to_crs('epsg:3035')
    geodf_list.append(beach_pols)

#merge all of the dataframes into one
beach_pols = geopandas.GeoDataFrame(pd.concat( geodf_list, ignore_index=True))

def to_cell_id(x, y):
    return str(x) + '_' + str(y)

# chose which tiles to keep
hash_dict = {}
for beach in beach_pols.geometry:
    x, y = beach.exterior.coords.xy
    x = [int(coord/1000) for coord in x]
    y = [int(coord/1000) for coord in y]
    
    for i in range(min(x), max(x)+1):
        for j in range(min(y), max(y)+1):
            hash_dict[to_cell_id(m*(i//m), m*(j//m))] = True

# chose which tiles to drop
dropped = []
for index, tile in grid_pols.iterrows():
    if (hash_dict.get(tile.cell_id) is None):
        dropped.append(index)

# drop tiles
print('Number of tiles we must keep:', len(hash_dict))
print('Number of tiles in the beginning:', len(grid_pols))
grid_pols.drop(dropped, inplace=True)
print('Number of tiles in the end:', len(grid_pols))

# save reduced grid
grid_pols.to_file(dest_file, driver='GeoJSON')
