from numpy import load, shape, concatenate, save

data = load('test_images_3.npy')
data_1 = load('test_images_4.npy')


data_labels = load('test_labels_3.npy')
data_labels_1 = load('test_labels_4.npy')


merged_images = concatenate((data, data_1), axis=0)
merged_labels = concatenate((data_labels, data_labels_1), axis=0)

with open('test_images_final.npy', 'wb') as f:
    print(merged_images.shape)
    save(f, merged_images)

with open('test_labels_final.npy', 'wb') as g:
    print(merged_labels.shape)
    save(g, merged_labels)