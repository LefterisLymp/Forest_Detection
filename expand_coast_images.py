import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
from matplotlib import cm
from ISR.models import RRDN
from PIL import Image

#script that applies super resolution to the images
if (len(sys.argv) < 2 or int(sys.argv[1]) not in range(2, 851)):
    print('Usage: python expand_images.py m \nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

rrdn = RRDN(weights='gans')

# create folder structure
dataDirName = 'train_data'
inputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles_forest')
outputDirName = os.path.join(dataDirName, m + 'k', 'data_tiles_forest_sr')
if (os.path.exists(outputDirName) == False):
    os.makedirs(outputDirName)

count = 0
for i, (path, subdirs, files) in enumerate(os.walk(inputDirName)):
    for file in files:
        outputFileName = file.split('.')[0] + '_sr.tif'
        outputFile = os.path.join(outputDirName, outputFileName)
        if os.path.exists(outputFile):
            break
        if file.endswith('.tif'):
            # ignore the 20m-resolution images
            if file.endswith('20m.tif'):
                continue
            # open .tif file and read image
            image_path = os.path.join(path, file)

            print('----------------------------------------------------')
            print('Tile no', count, ':', file.split('.')[0])
            print('----------------------------------------------------')

            count += 1
            img = Image.open(image_path)
            lr_img = np.array(img)
            sr_img = rrdn.predict(lr_img)
            sr = Image.fromarray(sr_img)
            im = sr.save(outputFile)