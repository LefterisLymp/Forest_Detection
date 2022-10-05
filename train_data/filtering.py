from numpy import load, shape, concatenate, save, count_nonzero, dot, rot90
import numpy as np
from random import sample, shuffle

#script that filters the dataset and can be used to create train, val and test sets in a straightforward way

data = load('train_images_sr.npy')
labels = load('train_labels_sr.npy')

class_indexes = []
pixel_count = 0

#filtering
for i in range(labels.shape[0]):
    x = labels[i, :, :]
    pxls = count_nonzero(x == 1)

    if pxls > 4000: #>10% of image
        class_indexes.append(i)

data = data[class_indexes]
data_labels = labels[class_indexes]

pixel_count = np.sum(data_labels)

print(data.shape[0])
print("Pixel count:", pixel_count)
print("Total pixels:", data.shape[0]*data.shape[1]*data.shape[2])

with open('train_images_forest.npy', 'wb') as f:
    save(f, balanced_data)

with open('train_labels_forest.npy', 'wb') as g:
    save(g, balanced_labels)

