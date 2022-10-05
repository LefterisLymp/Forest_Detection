import os
import sys
import shutil
from sklearn.model_selection import train_test_split
import random

if (len(sys.argv) < 2):
    print('Usage: python split_files.py m\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

seed = 42
random.seed(seed)

rootDir = m + 'k'

# subdirectories
tiles = 'data_tiles'
coastTiles = 'data_tiles_forest'
labels = 'data_labels'
polys = 'data_labels_poly'

# output directories
trainTilesDir  = os.path.join(rootDir, 'train', tiles)
valTilesDir    = os.path.join(rootDir, 'val',   tiles)
testTilesDir   = os.path.join(rootDir, 'test',  tiles)

trainCtilesDir = os.path.join(rootDir, 'train', coastTiles)
valCtilesDir   = os.path.join(rootDir, 'val',   coastTiles)
testCtilesDir  = os.path.join(rootDir, 'test',  coastTiles)

trainLabelsDir = os.path.join(rootDir, 'train', labels)
valLabelsDir   = os.path.join(rootDir, 'val',   labels)
testLabelsDir  = os.path.join(rootDir, 'test',  labels)

trainPolysDir = os.path.join(rootDir, 'train', polys)
valPolysDir   = os.path.join(rootDir, 'val',   polys)
testPolysDir  = os.path.join(rootDir, 'test',  polys)

for d in [trainTilesDir, valTilesDir, testTilesDir, trainCtilesDir, valCtilesDir, testCtilesDir, trainLabelsDir, valLabelsDir, testLabelsDir, trainPolysDir, valPolysDir, testPolysDir]:
    if not os.path.exists(d):
        os.makedirs(d)

# input directories
tilesDir = os.path.join(rootDir, tiles)
coastTilesDir = os.path.join(rootDir, coastTiles)
labelsDir = os.path.join(rootDir, labels)
polysDir = os.path.join(rootDir, polys)

TEST_RATIO = 0.15
VAL_RATIO = 0.15

tile_ids = os.listdir(tilesDir)
train_val_tiles, test_tiles = train_test_split(tile_ids, test_size=TEST_RATIO, random_state=seed)
train_tiles, val_tiles = train_test_split(train_val_tiles, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=seed)


for tile_id in train_tiles:

    src = os.path.join(tilesDir, tile_id)
    #shutil.move(src, trainTilesDir)

    src = os.path.join(coastTilesDir, tile_id + '_forest.tif')
    shutil.move(src, trainCtilesDir)

    src = os.path.join(labelsDir, tile_id + '.png')
    shutil.move(src, trainLabelsDir)

    src = os.path.join(polysDir, tile_id + '.json')
    shutil.move(src, trainPolysDir)

for tile_id in val_tiles:

    src = os.path.join(tilesDir, tile_id)
    #shutil.move(src, valTilesDir)

    src = os.path.join(coastTilesDir, tile_id + '_forest.tif')
    shutil.move(src, valCtilesDir)

    src = os.path.join(labelsDir, tile_id + '.png')
    shutil.move(src, valLabelsDir)

    src = os.path.join(polysDir, tile_id + '.json')
    shutil.move(src, valPolysDir)

for tile_id in test_tiles:

    src = os.path.join(tilesDir, tile_id)
    #shutil.move(src, testTilesDir)

    src = os.path.join(coastTilesDir, tile_id + '_forest.tif')
    shutil.move(src, testCtilesDir)

    src = os.path.join(labelsDir, tile_id + '.png')
    shutil.move(src, testLabelsDir)

    src = os.path.join(polysDir, tile_id + '.json')
    shutil.move(src, testPolysDir)

for d in [tilesDir, coastTilesDir, labelsDir, polysDir]:
    if os.listdir(d) == []:
        shutil.rmtree(d)
