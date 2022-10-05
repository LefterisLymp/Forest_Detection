from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import cv2

data = np.load('test_images_mix_forest.npy')
labels = np.load('test_labels_mix_forest.npy')

#img = Image.fromarray(data, 'RGB')

dim = (224, 224)

print("Resizing test data...")

res_img = []
res_lbl = []

for ind in tqdm(range(data.shape[0])):
    resized = cv2.resize(data[ind, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
    resized_target = cv2.resize(labels[ind, :, :], dim, interpolation=cv2.INTER_NEAREST)
    res_img.append(resized)
    res_lbl.append(resized_target)

data = np.array(res_img)
labels = np.array(res_lbl)


for i in tqdm(range(data.shape[0])):
    #im = data[i].transpose(2, 0, 1)
    im = data[i]
    img = Image.fromarray(im, 'RGB')
    lbl = Image.fromarray(labels[i])


    img.save('data_mix/test_'+ str(i) + '.png')
    lbl.convert('L').save('annotations_mix/test_'+ str(i) + '_labelTrainIds.png')


""""
img.save('forest_ex.png')


img_r = Image.fromarray(data[:, :, 0])
img_g = Image.fromarray(data[:, :, 1])
img_b = Image.fromarray(data[:, :, 2])

img_r.convert('L').save('forest_ex_r.png')
img_g.convert('L').save('forest_ex_g.png')
img_b.convert('L').save('forest_ex_b.png')


"""
"""
for i in range(3):
    img = Image.fromarray(data[i], 'RGB')
    l = labels[i]
    l[l == 1] = 2
    l[l == 0] = 1
    l[l == 2] = 0
    lbl = Image.fromarray(l*255)

    img.save('data'+ str(i+1) + '.png')
    lbl.convert('L').save('label' + str(i+1) + '.jpg')
    data[i][l == 0, 1] = 255
    data[i][l == 0, 0] = 0
    data[i][l == 0, 2] = 0

    img = Image.fromarray(data[i], 'RGB')
    img.save('data_labels' + str(i + 1) + '.png')
    
"""
