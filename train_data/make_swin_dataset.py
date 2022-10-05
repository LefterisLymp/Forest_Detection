from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import cv2
#script that converts the dataset into mmsegmentation format
#change the directories in the following 2 lines whether it is train, val or test data

data = np.load('test_images_forest.npy')
labels = np.load('test_labels_forest.npy')

#img = Image.fromarray(data, 'RGB')

#resize to fit the mmsegmentation configuration
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
