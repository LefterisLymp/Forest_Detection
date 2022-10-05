import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pyproj
import cv2
from shapely.geometry import mapping, Point, Polygon, LineString
from shapely.ops import cascaded_union, transform
import rasterio
import rasterio.mask
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

if (len(sys.argv) < 3 or int(sys.argv[1]) not in range(2, 851) or sys.argv[2] not in ['tif', 'png', 'all']):
    print('Usage: python make_expanded_train_data.py m [tif|png|all] sf \nwhere m is the tile size in km, with 1 < m <= 850 and sf is the scale factor')
    sys.exit()
m = sys.argv[1]

#Scale factor is 4 by default
sf = 4

if len(sys.argv) == 4:
    sf = int(sys.argv[3])

# create folder structure
dataDirName = 'train_data'
inputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles')
outputDirName = os.path.join(dataDirName, m + 'k', 'data_labels_sr')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

polyOutputDirName = os.path.join(dataDirName, m + 'k', 'data_labels_poly_sr')
if (os.path.exists(polyOutputDirName) == False):
    os.makedirs(polyOutputDirName)

tif_out = (sys.argv[2] in ['tif', 'all'])
png_out = (sys.argv[2] in ['png', 'all'])

tifOutputDirName = os.path.join(outputDirName, 'tif')
if (tif_out and os.path.exists(tifOutputDirName) == False):
    os.makedirs(tifOutputDirName)


# function that generates a binary mask from a polygon
def generate_mask(tile_id, shapes):
    # load raster
    with rasterio.open(tile_file(tile_id), 'r') as src:
        pixel_polys_dict = dict()

        for i in range(sf*sf):
            pixel_polys_dict[i] = []

        polys = []
        #pixel_polys = []
        cnt = 0
        for (i, shape) in enumerate(shapes):
            # reproject polygon
            beach = shape.geometry
            # Get class id
            class_id = shape['surface']
            project = pyproj.Transformer.from_crs(pyproj.CRS('epsg:3035'), src.crs, always_xy=True).transform
            beach = transform(project, beach)

            # find polygon pixels
            pixels = [src.index(x, y) for (x, y) in np.array(beach.exterior.coords)]

            # chose only in-image pixels and scale them
            pixels = [(sf*x, sf*y) for (x, y) in pixels]
            in_pixels = [(x, y) for (x, y) in pixels if x in range(0, 800) and y in range(0, 800)]

            if in_pixels != [] and {'geometry': in_pixels} not in pixel_polys_dict.values():
                cnt += 1
                # transform polygon using raster metadata
                poly_pts = [~src.meta['transform'] * tuple(i) for i in np.array(beach.exterior.coords)]
                poly_pts = [(sf * x, sf * y) for (x, y) in poly_pts]
                poly = Polygon(poly_pts)
                polys.append((poly, cnt))
                try:
                    polygon = Polygon(in_pixels)
                except:
                    continue
                for i in range(sf):
                    for j in range(sf):
                        #tr: top right, tl: top left, br: bottom right, bl: bottom left
                        y1 = 200*i
                        x1 = 200*j
                        y2 = y1 + 199
                        x2 = x1 + 199
                        image_limits = Polygon([[y1, x1], [y1, x2], [y2, x2], [y2, x1], [y1, x1]])
                        seg = Polygon(polygon).simplify(0).buffer(1, resolution=1).intersection(image_limits)
                        if seg.is_empty:
                            continue
                        else:
                            #coords_list = [list(x.exterior.coords) for x in seg._geom]
                            if hasattr(seg, 'exterior'):
                                geom_pixels = [(round(x) - y1, round(y) - x1) for (x, y) in np.array(seg.exterior.coords)]
                                pixel_polys_dict[sf*i + j].append({'geometry': geom_pixels, 'class_id': class_id})
                            else:
                                coords_list = []
                                #print(seg.geoms.items())
                                if isinstance(seg, LineString):
                                    coords_list.append(list(seg.coords))
                                elif isinstance(seg, Point):
                                    continue
                                else:
                                    for x in seg.geoms:
                                        if isinstance(x, LineString):
                                            coords_list.append(list(x.coords))
                                        elif isinstance(x, Polygon):
                                            coords_list.append(list(x.exterior.coords))
                                        else: continue
                                for l in coords_list:
                                    geom_pixels = [(round(x) - y1, round(y) - x1) for (x, y) in l]
                                    pixel_polys_dict[sf * i + j].append({'geometry': geom_pixels, 'class_id': class_id})


        if cnt == 0:
            return None

        for i in range(sf*sf):
            if not pixel_polys_dict[i] == []:
                polyOutputFile = os.path.join(polyOutputDirName, tile_id + '_' + str(i) + '.json')

                with open(polyOutputFile, 'w') as output_json_file:
                    json.dump(pixel_polys_dict[i], output_json_file)

        # generate the mask
        mask = rasterize(shapes=polys, out_shape=(sf*src.meta['height'], sf*src.meta['width']), all_touched=True)

        # save .tif file
        mask = mask.astype('uint16')

        meta = src.meta.copy()
        meta['count'] = 1

        for i in range(sf*sf):
            if pixel_polys_dict[i] == []:
                continue

            row = i // sf
            col = i % sf

            p_mask = mask[row*200:(row+1)*200, col*200:(col+1)*200]

            if tif_out:
                tifOutputFile = os.path.join(tifOutputDirName, tile_id + '_' + str(i) +'.tif')
                with rasterio.open(tifOutputFile, 'w', **meta) as dst:
                    dst.write(p_mask, 1)

            if png_out:
                meta['driver'] = 'PNG'
                pngOutputFile = os.path.join(outputDirName, tile_id + '_' + str(i) + '.png')
                with rasterio.open(pngOutputFile, 'w', **meta) as dst:
                    dst.write(p_mask, 1)
                if os.path.exists(os.path.join(outputDirName, tile_id + '_' + str(i) + '.png.aux.xml')):
                    os.remove(os.path.join(outputDirName, tile_id +  '_' + str(i) + '.png.aux.xml'))



# function that returns tile's id
def to_cell_id(x, y):
    return str(x) + '_' + str(y)


# function that returns all tiles that a shape is in
def tile_list(shape):
    x, y = shape.geometry.exterior.coords.xy
    x = [int(coord / 1000) for coord in x]
    y = [int(coord / 1000) for coord in y]
    tile_ids = []
    for i in range(min(x), max(x) + 1):
        for j in range(min(y), max(y) + 1):
            m_ = int(m)
            ci = m_ * (i // m_)
            cj = m_ * (j // m_)
            tile_ids.append(to_cell_id(ci - m_, cj - m_))
            tile_ids.append(to_cell_id(ci - m_, cj))
            tile_ids.append(to_cell_id(ci - m_, cj + m_))
            tile_ids.append(to_cell_id(ci, cj - m_))
            tile_ids.append(to_cell_id(ci, cj))
            tile_ids.append(to_cell_id(ci, cj + m_))
            tile_ids.append(to_cell_id(ci + m_, cj - m_))
            tile_ids.append(to_cell_id(ci + m_, cj))
            tile_ids.append(to_cell_id(ci + m_, cj + m_))
    return tile_ids


# function that returns tile file path
def tile_file(tile_id):
    tif_file = os.path.join(inputDirName, tile_id, tile_id + '.tif')
    if os.path.exists(tif_file):
        return tif_file
    print('ERROR: no .tif file found')


# read shapefiles
grid_polys = gpd.read_file(os.path.join(dataDirName, 'grid_greece', 'grid_' + m + 'k.GeoJSON'))
# beach_polys_ = gpd.read_file(os.path.join('train_data', 'data_osm', 'osm', 'beaches.GeoJSON'))
# beach_polys = gpd.GeoDataFrame.from_features(beach_polys_, crs='epsg:4326').to_crs('epsg:3035')

beach_files = os.listdir(dataDirName + '/data_osm/osm')

geodf_list = []

for beach_file in beach_files:
    beach_file = os.path.join(dataDirName, 'data_osm', 'osm', beach_file)
    beach_pols = gpd.read_file(beach_file)
    beach_pols = gpd.GeoDataFrame.from_features(beach_pols, crs='epsg:4326').to_crs('epsg:3035')
    geodf_list.append(beach_pols)

# merge all of the dataframes into one
beach_polys = gpd.GeoDataFrame(pd.concat(geodf_list, ignore_index=True))

# create forests per tile dict
json_dict = {i: [] for i in grid_polys.cell_id}
for index, beach in beach_polys.iterrows():
    for tile_id in tile_list(beach):
        if json_dict.get(tile_id) is not None:
            json_dict[tile_id].append(beach)

#find all existing json files
files = os.listdir(polyOutputDirName)
file_set = set()
N = len(files)
for i in range(N):
    name = files[i].split('.')[0]
    names = name.split('_')

    name = names[0] + '_' + names[1]
    file_set.add(name)

# create label files
empty_tiles_cnt = 0
missing_tiles_cnt = 0
already_tiles_cnt = 0
found_previous = False
previous_tile_id = ''
for index, tile in grid_polys.iterrows():

    tile_id = tile.cell_id

    # Compatibility of file_set to be done
    if tile_id in file_set:
        previous_tile_id = tile_id
        continue
    elif index == 0:
        found_previous = True
        previous_tile_id = tile_id

    elif not found_previous:  # Checking if it is the first tile
        found_previous = True
        print('----------------------------------------------------')
        print('Tile no', index - 1, ':', previous_tile_id)
        print('----------------------------------------------------')

        if (os.path.exists(os.path.join(inputDirName, tile_id)) == True) and (json_dict[tile_id] != []):
            generate_mask(previous_tile_id, json_dict[previous_tile_id])

    print('----------------------------------------------------')
    print('Tile no', index, ':', tile_id)
    print('----------------------------------------------------')

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