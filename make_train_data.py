import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pyproj
import cv2
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union, transform
import rasterio
import rasterio.mask
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

DATA_DIR = '/data/data1/users/lefterislymp/coast_dir/train_data/data_osm'

if (len(sys.argv) < 3 or int(sys.argv[1]) not in range(2,851) or sys.argv[2] not in ['tif','png','all']):
    print('Usage: python make_train_data.py m [tif|png|all] \nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

# create folder structure
dataDirName = 'train_data'
inputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles')
outputDirName = os.path.join(dataDirName, m + 'k', 'data_labels')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

polyOutputDirName = os.path.join(dataDirName, m + 'k', 'data_labels_poly')
if (os.path.exists(polyOutputDirName) == False):
    os.makedirs(polyOutputDirName)

tif_out = (sys.argv[2] in ['tif','all'])
png_out = (sys.argv[2] in ['png','all'])

tifOutputDirName = os.path.join(outputDirName, 'tif')
if (tif_out and os.path.exists(tifOutputDirName) == False):
    os.makedirs(tifOutputDirName)

# function that generates a binary mask from a polygon
def generate_mask(tile_id, shapes):
    
    # load raster
    with rasterio.open(tile_file(tile_id), 'r') as src:
        polys = []
        pixel_polys = []
        cnt = 0
        for (i, shape) in enumerate(shapes):
            # reproject polygon
            beach = shape.geometry
            #Get class id
            class_id = shape['surface']
            project = pyproj.Transformer.from_crs(pyproj.CRS('epsg:3035'), src.crs, always_xy=True).transform
            beach = transform(project, beach)

            # find polygon pixels
            pixels = [src.index(x,y) for (x,y) in np.array(beach.exterior.coords)]
            # chose only in-image pixels
            in_pixels = [(x,y) for (x,y) in pixels if x in range(0,200) and y in range(0,200)]
            if in_pixels != [] and {'geometry': in_pixels} not in pixel_polys:
                cnt += 1
                # transform polygon using raster metadata
                poly_pts = [~src.meta['transform'] * tuple(i) for i in np.array(beach.exterior.coords)]
                poly = Polygon(poly_pts)
                polys.append((poly, cnt))
                pixel_polys.append({'geometry': in_pixels, 'class_id': class_id})

        if cnt == 0:
            return None

        polyOutputFile = os.path.join(polyOutputDirName, tile_id + '.json')
        with open(polyOutputFile, 'w') as output_json_file:
            json.dump(pixel_polys, output_json_file)
        
        # generate the mask
        mask = rasterize(shapes=polys, out_shape=(src.meta['height'], src.meta['width']), all_touched=True)
        
        # save .tif file
        mask = mask.astype('uint16')

        meta = src.meta.copy()
        meta['count'] = 1

        if tif_out:
            tifOutputFile = os.path.join(tifOutputDirName, tile_id + '.tif')
            with rasterio.open(tifOutputFile, 'w', **meta) as dst:
                dst.write(mask, 1)

        if png_out:
            meta['driver'] = 'PNG'
            pngOutputFile = os.path.join(outputDirName, tile_id + '.png')
            with rasterio.open(pngOutputFile, 'w', **meta) as dst:
                dst.write(mask, 1)
            if os.path.exists(os.path.join(outputDirName, tile_id + '.png.aux.xml')):
                os.remove(os.path.join(outputDirName, tile_id + '.png.aux.xml'))


# function that returns tile's id
def to_cell_id(x, y):
    return str(x) + '_' + str(y)


# function that returns all tiles that a shape is in
def tile_list(shape):
    x, y = shape.geometry.exterior.coords.xy
    x = [int(coord/1000) for coord in x]
    y = [int(coord/1000) for coord in y]
    tile_ids = []
    for i in range(min(x), max(x)+1):
        for j in range(min(y), max(y)+1):
            m_ = int(m)
            ci = m_*(i//m_)
            cj = m_*(j//m_)
            tile_ids.append(to_cell_id(ci-m_, cj-m_))
            tile_ids.append(to_cell_id(ci-m_, cj   ))
            tile_ids.append(to_cell_id(ci-m_, cj+m_))
            tile_ids.append(to_cell_id(ci   , cj-m_))
            tile_ids.append(to_cell_id(ci   , cj   ))
            tile_ids.append(to_cell_id(ci   , cj+m_))
            tile_ids.append(to_cell_id(ci+m_, cj-m_))
            tile_ids.append(to_cell_id(ci+m_, cj   ))
            tile_ids.append(to_cell_id(ci+m_, cj+m_))
    return tile_ids


# function that returns tile file path
def tile_file(tile_id):
    tif_file = os.path.join(inputDirName, tile_id, tile_id + '.tif')
    if os.path.exists(tif_file):
        return tif_file
    print('ERROR: no .tif file found')


# read shapefiles
grid_polys = gpd.read_file(os.path.join('train_data', 'grid_greece', 'grid_' + m + 'k.GeoJSON'))
#beach_polys_ = gpd.read_file(os.path.join('train_data', 'data_osm', 'osm', 'beaches.GeoJSON'))
#beach_polys = gpd.GeoDataFrame.from_features(beach_polys_, crs='epsg:4326').to_crs('epsg:3035')

beach_files = os.listdir(DATA_DIR + '/osm')

geodf_list = []

for beach_file in beach_files:
    beach_file = os.path.join(DATA_DIR, 'osm', beach_file)
    beach_pols = gpd.read_file(beach_file)
    beach_pols = gpd.GeoDataFrame.from_features(beach_pols, crs='epsg:4326').to_crs('epsg:3035')
    geodf_list.append(beach_pols)

#merge all of the dataframes into one
beach_polys = gpd.GeoDataFrame(pd.concat(geodf_list, ignore_index=True))

# create beaches per tile dict
json_dict = {i: [] for i in grid_polys.cell_id}
for index, beach in beach_polys.iterrows():
    for tile_id in tile_list(beach):
        if json_dict.get(tile_id) is not None:
            json_dict[tile_id].append(beach)

# create label files
empty_tiles_cnt = 0
missing_tiles_cnt = 0
already_tiles_cnt = 0
for index, tile in grid_polys.iterrows():

    print('----------------------------------------------------')
    print('Tile no', index, ':', tile.cell_id)
    print('----------------------------------------------------')

    tile_id = tile.cell_id
    if (os.path.exists(os.path.join(inputDirName, tile_id)) == False):
        missing_tiles_cnt += 1
        print('ERROR: tile does not exist')
        continue
    if (json_dict[tile_id] == []):
        empty_tiles_cnt += 1
        print('ERROR: tile does not include any polygons')
        continue

    if os.path.exists(os.path.join(outputDirName, tile_id + '.png')) == True:
        already_tiles_cnt += 1
        continue
    
    generate_mask(tile_id, json_dict[tile_id])

print('There are', already_tiles_cnt, 'tiles already.')
print('There are', empty_tiles_cnt, 'empty and', missing_tiles_cnt, 'missing tiles.')
