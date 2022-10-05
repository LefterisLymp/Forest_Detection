from numpy import load, shape, concatenate, save, count_nonzero, dot
from random import sample, shuffle

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
    return np.count_nonzero(arr == 3) > 16000 #>0.5% of image

for i in range(labels.shape[0]):
    x = labels[i, :, :]
    pxls = count_nonzero(x == 4)

    if pxls > 0:
        class_indexes.append(i)
        pixel_count += pxls


data_4 = data[class_indexes]
data_labels_4 = labels[class_indexes]

"""
beach_data = beach_data[indexes_2]
beach_labels = dot(2, beach_labels[indexes_2]) #2* for the labels to match

data_2 = concatenate((data_2, beach_data), axis = 0)
data_labels_2 = concatenate((data_labels_2, beach_labels), axis = 0)
"""



balanced_data = data_4
balanced_labels = data_labels_4

balanced_labels[balanced_labels != 4] = 0
balanced_labels[balanced_labels == 4] = 1


print(balanced_data.shape[0])
print("Pixel count:", pixel_count)
print("Total pixels:", balanced_data.shape[0]*balanced_data.shape[1]*balanced_data.shape[2])


indexes = list(range(balanced_data.shape[0]))
shuffle(indexes)

balanced_data = balanced_data[indexes]
balanced_labels = balanced_labels[indexes]

print(balanced_data.shape)

with open('test_images_residential.npy', 'wb') as f:
    save(f, balanced_data)

with open('test_labels_residential.npy', 'wb') as g:
    save(g, balanced_labels)
