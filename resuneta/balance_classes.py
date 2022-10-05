from numpy import load, shape, concatenate, save, count_nonzero, dot
from random import sample, shuffle

data = load('train_images_final.npy')
labels = load('train_labels_final.npy')

class_map = {
    1: "agriculture",
    2: "beach",
    3: "forest",
    4: "residential"
}

class_count = {1: 0, 2: 0, 3: 0, 4: 0}
class_indexes = {1: [], 2: [], 3: [], 4: []}

for i in range(labels.shape[0]):
    x = labels[i, :, :]
    count_instances = []
    for ind in range(1, 5):
        count_instances.append(count_nonzero(x == ind))
    count_instances.append(count_nonzero)
    max_ind = 0
    counter = -1
    for ind in range(len(count_instances)):
        if count_instances[ind] > counter:
            max_ind = ind + 1
            counter = count_instances[ind]
    class_count[max_ind] += 1
    class_indexes[max_ind].append(i)

beach_data = load('train_images_beach.npy')
beach_labels = load('train_labels_beach.npy')
print(class_count)

#{1: 7853, 2: 286, 3: 10428, 4: 2702}
class_indexes[1] = sample(class_indexes[1], 2500)
class_indexes[3] = sample(class_indexes[3], 2500)
class_indexes[4] = sample(class_indexes[4], 2500)

"""
indexes_2 = sample(list(range(beach_data.shape[0])), 2314)
data_2 = data[class_indexes[2]]
data_labels_2 = labels[class_indexes[2]]
"""

data_1 = data[class_indexes[1]]
data_labels_1 = labels[class_indexes[1]]

data_3 = data[class_indexes[3]]
data_labels_3 = labels[class_indexes[3]]

data_4 = data[class_indexes[4]]
data_labels_4 = labels[class_indexes[4]]

"""
beach_data = beach_data[indexes_2]
beach_labels = dot(2, beach_labels[indexes_2]) #2* for the labels to match

data_2 = concatenate((data_2, beach_data), axis = 0)
data_labels_2 = concatenate((data_labels_2, beach_labels), axis = 0)
"""

balanced_data = concatenate((data_1, data_3, data_4), axis = 0)
balanced_labels = concatenate((data_labels_1, data_labels_3, data_labels_4), axis = 0)

balanced_labels[balanced_labels == 2] = 0
balanced_labels[balanced_labels == 3] = 2
balanced_labels[balanced_labels == 4] = 3

print(balanced_data.shape[0])
indexes = list(range(balanced_data.shape[0]))
shuffle(indexes)

balanced_data = balanced_data[indexes]
balanced_labels = balanced_labels[indexes]

print(balanced_data.shape)

with open('train_images_balanced_3c.npy', 'wb') as f:
    save(f, balanced_data)

with open('train_labels_balanced_3c.npy', 'wb') as g:
    save(g, balanced_labels)