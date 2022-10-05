import os
import sys
import shutil
import matplotlib.pyplot as plt

#deletes tiles with no corresponding ground truths
if (len(sys.argv) < 2):
    print('Usage: python delete_files.py m\nwhere m is the tile size in km, with 1 < m <= 850')
    sys.exit()
m = sys.argv[1]

rootDir = m + 'k'

# subdirectories
tiles = 'data_tiles'
coastTiles = 'data_tiles_coast'
labels = 'data_labels'
polys = 'data_labels_poly'

# input directories
tilesDir = os.path.join(rootDir, tiles)
coastTilesDir = os.path.join(rootDir, coastTiles)
labelsDir = os.path.join(rootDir, labels)
polysDir = os.path.join(rootDir, polys)

def remove_file(filename):
    if os.path.exists(filename):
        print('Removing', filename, '...')
        os.remove(filename)

def remove_dir(dirname):
    if os.path.exists(dirname):
        print('Removing', dirname, '...')
        shutil.rmtree(dirname)

def remove_tile(tile_id):
    labelPath = os.path.join(labelsDir, tile_id + '.png')
    remove_file(labelPath)

    polyPath = os.path.join(polysDir, tile_id + '.json')
    remove_file(polyPath)

    tilePath = os.path.join(tilesDir, tile_id)
    remove_dir(tilePath)

    coastTilePath = os.path.join(coastTilesDir, tile_id + '_coast.tif')
    remove_file(coastTilePath)

    print()

for tile_id in os.listdir(tilesDir):
    if not os.path.exists(os.path.join(coastTilesDir, tile_id + '_coast.tif')):
        remove_tile(tile_id)
    if not os.path.exists(os.path.join(labelsDir, tile_id + '.png')):
        remove_tile(tile_id)
    else:
        im = plt.imread(os.path.join(labelsDir, tile_id + '.png'))
        if im.max() == 0:
            remove_tile(tile_id)
