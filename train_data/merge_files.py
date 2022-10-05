import os
import sys
import shutil

if (len(sys.argv) < 2):
    print('Usage: python merge_files.py m\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

rootDir = m + 'k'

# subdirectories
tiles = 'data_tiles'
coastTiles = 'data_tiles_forest'
labels = 'data_labels'
polys = 'data_labels_poly'

# output directories
trainDir = os.path.join(rootDir, 'train')
valDir   = os.path.join(rootDir, 'val')
testDir  = os.path.join(rootDir, 'test')

trainTilesDir  = os.path.join(trainDir, tiles)
valTilesDir    = os.path.join(valDir,   tiles)
testTilesDir   = os.path.join(testDir,  tiles)

trainCtilesDir = os.path.join(trainDir, coastTiles)
valCtilesDir   = os.path.join(valDir,   coastTiles)
testCtilesDir  = os.path.join(testDir,  coastTiles)

trainLabelsDir = os.path.join(trainDir, labels)
valLabelsDir   = os.path.join(valDir,   labels)
testLabelsDir  = os.path.join(testDir,  labels)

trainPolysDir = os.path.join(trainDir, polys)
valPolysDir   = os.path.join(valDir,   polys)
testPolysDir  = os.path.join(testDir,  polys)

# input directories
tilesDir = os.path.join(rootDir, tiles)
coastTilesDir = os.path.join(rootDir, coastTiles)
labelsDir = os.path.join(rootDir, labels)
polysDir = os.path.join(rootDir, polys)

for d in [tilesDir, coastTilesDir, labelsDir, polysDir]:
    if not os.path.exists(d):
        os.makedirs(d)


for tile_id in os.listdir(trainTilesDir):

    src = os.path.join(trainTilesDir, tile_id)
    shutil.move(src, tilesDir)

    src = os.path.join(trainCtilesDir, tile_id + '_forest.tif')
    shutil.move(src, coastTilesDir)

    src = os.path.join(trainLabelsDir, tile_id + '.png')
    shutil.move(src, labelsDir)

    src = os.path.join(trainPolysDir, tile_id + '.json')
    shutil.move(src, polysDir)

for tile_id in os.listdir(valTilesDir):

    src = os.path.join(valTilesDir, tile_id)
    shutil.move(src, tilesDir)

    src = os.path.join(valCtilesDir, tile_id + '_forest.tif')
    shutil.move(src, coastTilesDir)

    src = os.path.join(valLabelsDir, tile_id + '.png')
    shutil.move(src, labelsDir)

    src = os.path.join(valPolysDir, tile_id + '.json')
    shutil.move(src, polysDir)

for tile_id in os.listdir(testTilesDir):

    src = os.path.join(testTilesDir, tile_id)
    shutil.move(src, tilesDir)

    src = os.path.join(testCtilesDir, tile_id + '_forest.tif')
    shutil.move(src, coastTilesDir)

    src = os.path.join(testLabelsDir, tile_id + '.png')
    shutil.move(src, labelsDir)

    src = os.path.join(testPolysDir, tile_id + '.json')
    shutil.move(src, polysDir)

for d in [trainTilesDir, valTilesDir, testTilesDir, trainCtilesDir, valCtilesDir, testCtilesDir, trainLabelsDir, valLabelsDir, testLabelsDir, trainPolysDir, valPolysDir, testPolysDir, trainDir, valDir, testDir]:
    if os.listdir(d) == []:
        shutil.rmtree(d)
