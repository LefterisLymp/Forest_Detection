import os
import sys
import json
import datetime
import re
import fnmatch
import numpy as np
import cv2
import logging

from PIL import Image
import tifffile
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union, cascaded_union, polygonize

train_data_dir = '../all_train_data/2k/train/data_tiles_coast_segments_sr'
train_labels_dir = '../all_train_data/2k/train/data_labels_poly_sr'

val_data_dir = '../all_train_data/2k/val/data_tiles_coast_segments_sr'
val_labels_dir = '../all_train_data/2k/val/data_labels_poly_sr'

test_data_dir = '../all_train_data/2k/test/data_tiles_coast_segments_sr'
test_labels_dir = '../all_train_data/2k/test/data_labels_poly_sr'


class_map = {
    "agriculture": 1,
    "beach": 2,
    "forest": 3,
    "residential": 4
}

"""
class_map = {
    "residential": 1
}
"""

logging.basicConfig(filename="beach_info.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


split_index = 0 #0 is train, 1 is val, 2 is test
split_map = {0: 'train', 1: 'val', 2:'test'}

for imageDir, polyDir in zip([train_data_dir, val_data_dir, test_data_dir], [train_labels_dir, val_labels_dir, test_labels_dir]):
    image_files = [os.path.join(imageDir, name) for name in os.listdir(imageDir)]

    image_arrays = []
    label_arrays = []

    image_outfile = split_map[split_index] + '_images_final.npy'
    label_outfile = split_map[split_index] + '_labels_final.npy'

    print('Split:', split_map[split_index])

    logger.info('Split: ' + split_map[split_index])
    # go through each image
    for i, image_filename in enumerate(image_files):
        #if i > 10 or split_index > 0: break
        #if split_index == 2 and i <= 600: continue

        basename = os.path.splitext(os.path.basename(image_filename))[0]
        names = basename.split('_')

        tile_id = names[0] + '_' + names[1] + '_' + names[2]

        print('----------------------------------------------------')
        print('Tile no', i, ':', tile_id)
        print('----------------------------------------------------')

        #image = tifffile.imread(image_filename)
        image = Image.open(image_filename)
        image_ar = np.array(image)

        image_arrays.append(image_ar)

        #size = image.shape[:2]

        labelDir = ''
        polyDir_names = polyDir.split('/')
        for it in range(len(polyDir_names)-1):
            labelDir += polyDir_names[it] + '/'
        labelDir += 'data_labels_sr'

        label_filename = os.path.join(labelDir, tile_id + '.png')
        label_img = Image.open(label_filename)
        label = np.array(label_img)

        poly_filename = os.path.join(polyDir, tile_id + '.json')

        label_array = np.zeros((200, 200))
        polygons = []

        with open(poly_filename) as json_file:
            polys = json.load(json_file)
            for poly in polys:
                class_id = poly['class_id']
                #if class_id != "residential":
                #    continue

                class_num = class_map[class_id]

                poly = [[y, x] for [x, y] in poly['geometry']]

                bbdim = cv2.minAreaRect(np.array(poly))[1]
                if (bbdim[0] == 0) or (bbdim[1] == 0):
                    continue

                polygon = Polygon(poly)

                polygons.append((class_num, polygon))

        for x in range(200):
            for y in range(200):
                if label[x, y] == 0: continue
                else:
                    for (num, poly) in polygons:
                        if poly.contains(Point(x, y)):
                            label_array[x, y] = num
                            break

        label_arrays.append(label_array)

        if i % 100 == 0:
            print("Saving...")
            logger.info(str(i))

            with open(image_outfile, 'wb') as f:
                np.save(f, np.array(image_arrays))

            with open(label_outfile, 'wb') as f:
                np.save(f, np.array(label_arrays))


    image_arrays = np.array(image_arrays)
    label_arrays = np.array(label_arrays)



    with open(image_outfile, 'wb') as f:
        np.save(f, np.array(image_arrays))

    with open(label_outfile, 'wb') as f:
        np.save(f, np.array(label_arrays))

    split_index += 1