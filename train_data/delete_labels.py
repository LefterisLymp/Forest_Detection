import os
import sys
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt

#deletes tiles with no corresponding ground truths
if (len(sys.argv) < 2):
    print('Usage: python delete_labels.py m\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

rootDir = m + 'k'

# input directories
polysDir = os.path.join(rootDir, 'data_labels_poly')
labelsDir = os.path.join(rootDir, 'data_labels')

for tile in os.listdir(labelsDir):
    tile_id, ext = tile.split('.')
    
    # read image
    with open(os.path.join(polysDir, tile_id + '.json')) as file:
        polys = json.load(file)
    
    # read mask
    mask = imageio.imread(os.path.join(labelsDir, tile))
    masks = [(mask == i) for i in range(1, mask.max()+1)]

    for p, m in zip(polys, masks):
        if m.sum() == 0:
            print('ERROR:')
            print(p['geometry'])
            print(m.sum())
