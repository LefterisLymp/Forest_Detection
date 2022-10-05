from numpy import load, shape, concatenate, save, count_nonzero, dot, rot90
import numpy as np
from random import sample, shuffle

data_org = load('test_images_org_final.npy')
labels_org = load('test_labels_org_final.npy')

data = load('test_images_final.npy')
labels = load('test_labels_final.npy')

class_map = {
    1: "agriculture",
    2: "beach",
    3: "forest",
    4: "residential"
}

class_indexes = []
pixel_count = 0

def eligible_label(arr):
    return np.count_nonzero(arr == 3) > 16000 #>40% of image

for i in range(labels.shape[0]):
    x = labels[i, :, :]
    pxls = count_nonzero(x == 3)

    if pxls > 4000:
        class_indexes.append(i)

class_indexes = sample(class_indexes, 250)

data_3 = data[class_indexes]
data_labels_3 = labels[class_indexes]

balanced_data = data_3
balanced_labels = data_labels_3

balanced_labels[balanced_labels != 3] = 0
balanced_labels[balanced_labels == 3] = 1

pixel_count += np.sum(balanced_labels)

class_indexes = []
for i in range(labels_org.shape[0]):
    x = labels_org[i, :, :]
    pxls = count_nonzero(x == 3)

    if pxls > 0:
        class_indexes.append(i)


class_indexes = sample(class_indexes, 250)

data_3_org = data_org[class_indexes]
data_labels_3_org = labels_org[class_indexes]

balanced_data_org = data_3_org
balanced_labels_org = data_labels_3_org

balanced_labels_org[balanced_labels_org != 3] = 0
balanced_labels_org[balanced_labels_org == 3] = 1

pixel_count += np.sum(balanced_labels_org)

balanced_data = concatenate((balanced_data, balanced_data_org), axis=0)
balanced_labels = concatenate((balanced_labels, balanced_labels_org), axis=0)

indexes = list(range(balanced_data.shape[0]))
shuffle(indexes)

balanced_data = balanced_data[indexes]
balanced_labels = balanced_labels[indexes]

print(balanced_data.shape[0])
print("Pixel count:", pixel_count)
print("Total pixels:", balanced_data.shape[0]*balanced_data.shape[1]*balanced_data.shape[2])

with open('test_images_mix_forest.npy', 'wb') as f:
    save(f, balanced_data)

with open('test_labels_mix_forest.npy', 'wb') as g:
    save(g, balanced_labels)

"""
with open('test_images_org_forest.npy', 'wb') as f:
    save(f, balanced_data)

with open('test_labels_org_forest.npy', 'wb') as g:
    save(g, balanced_labels)
"""

