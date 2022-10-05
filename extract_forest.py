import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
from matplotlib import cm

if (len(sys.argv) < 3 or int(sys.argv[1]) not in range(2,851)):
    print('Usage: python extract_forest.py m [3|5]\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]
ch = sys.argv[2]

# create folder structure
dataDirName = 'train_data'
inputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles')
outputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles_forest')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

for i, (path, subdirs, files) in enumerate(os.walk(inputDirName)):
    for file in files:
        outputFileName = file.split('.')[0] + '_forest.tif'
        outputFile = os.path.join(outputDirName, outputFileName)
        if os.path.exists(outputFile):
            break
        if file.endswith('.tif'):
            # ignore the 20m-resolution images
            if file.endswith('20m.tif'):
                continue
            # open .tif file and read image
            image_path = os.path.join(path, file)
            with rasterio.open(image_path) as src:
                im = src.read()
                im = im / 10000
                meta = src.meta.copy()
        
        print('----------------------------------------------------')
        print('Tile no', i, ':', file.split('.')[0])
        print('----------------------------------------------------')

        final_image = im
        if (final_image.max() == 0.0):
            break

        outputFileName = file.split('.')[0] + '_forest.tif'
        outputFile = os.path.join(outputDirName, outputFileName)

        meta['dtype'] = 'uint8'
        final_image = 255.0*final_image
        final_image = final_image.astype(np.uint8)
        final_image = final_image[[2, 1, 0, 3, 4]]

        if ch == '3':
            meta['count'] = 3
            final_image = final_image[:3]

        with rasterio.open(outputFile, 'w', **meta) as dst:
            dst.write(final_image)
