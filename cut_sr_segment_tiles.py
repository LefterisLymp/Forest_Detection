import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib import cm

#script that cuts the super-resolved images into the multiple images of the original size
if (len(sys.argv) < 2 or int(sys.argv[1]) not in range(2, 851)):
    print('Usage: python cut_sr_segment_tiles.py m [sf]\nwhere m is the tile size in km, with 1 < m <= 850 and sf is the scale factor')
    sys.exit()
m = sys.argv[1]
if len(sys.argv) == 3:
    sf = int(sys.argv[2])
else:
    sf = 4 #scale factor is 4 by default

# create folder structure
dataDirName = 'train_data'
inputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles_forest_sr')
polyDirName = os.path.join(dataDirName, m + 'k', 'data_labels_poly_sr')
outputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles_forest_segments_sr')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

tile_dict = dict()
tiles = os.listdir(polyDirName)

for t in tiles:
    names = t.split('.')[0]
    names = names.split('_')
    tile_id = names[0] + '_' + names[1]
    segment = int(names[2])
    if tile_id in tile_dict.keys():
        tile_dict[tile_id].append(segment)
    else:
        tile_dict[tile_id] = [segment]



for i, file in enumerate(os.listdir(inputDirName)):
    image_path = os.path.join(inputDirName, file)
    with rasterio.open(image_path) as src:
        im = src.read()
        meta = src.meta.copy()
        meta['height'] //= sf
        meta['width'] //= sf
        height = meta['height']

        filename = file.split('.')[0]
        names = filename.split('_')

        tile_id = names[0] + '_' + names[1]

        print('----------------------------------------------------')
        print('Tile no', i, ':', tile_id)
        print('----------------------------------------------------')

        if tile_id not in tile_dict.keys():
            print('Tile is missing!')
            continue

        for segment in tile_dict[tile_id]:
            row = segment // sf
            column = segment % sf

            final_image = im[:, row*height:(row+1)*height, column*height:(column+1)*height]

            outputFileName = tile_id + '_' + str(segment) + '_forest.tif'
            outputFile = os.path.join(outputDirName, outputFileName)

            with rasterio.open(outputFile, 'w', **meta) as dst:
                dst.write(final_image)
