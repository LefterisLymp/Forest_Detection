import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from math import sqrt, ceil

import skimage.io
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray

from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

import cv2

from loss import *
#from utils import *
#from model import *

from keras_unet_collection.models import resunet_a_2d
from focal_loss import BinaryFocalLoss

np.random.seed(123)

physical_devices = tf.config.list_physical_devices('GPU')

# Disable first GPU
tf.config.set_visible_devices(physical_devices[1:], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
# Logical device was not created for first GPU
assert len(logical_devices) == len(physical_devices) - 1

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

print("Loading test data...")
test_data = np.load('test_images_mix_forest.npy')
test_target = np.load('test_labels_mix_forest.npy')

test_data_resized = []
test_target_resized = []

dim = (256, 256)

def eligible_label(arr):
    return np.count_nonzero(arr) > 58 #>0.5% of image

print("Resizing test data...")

for ind in tqdm(range(test_data.shape[0])):
    resized = cv2.resize(test_data[ind, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
    resized_target = cv2.resize(test_target[ind, :, :], dim, interpolation=cv2.INTER_NEAREST)
    test_data_resized.append(resized)
    test_target_resized.append(resized_target)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

test_data_resized = np.array(test_data_resized)
test_target_resized = np.array(test_target_resized)

test_data = test_data_resized
test_target = test_target_resized

#test_target = keras.utils.to_categorical(test_target, num_classes=2)

model_save_dir = 'checkpoints_weighted_forest_tanimoto/'

class_weights = [1.0, 1.0]

bn_loss = BinaryFocalLoss(gamma=0)

def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
    loss_iou = iou_seg(y_true, y_pred)

    return loss_focal+loss_iou

loss_fn = multiclass_weighted_tanimoto_loss(class_weights)

model = resunet_a_2d((256, 256, 3), [16, 32, 64, 128, 256],
                            dilation_num=[1, 3, 15, 31],
                            n_labels=2, aspp_num_down=256, aspp_num_up=128,
                            activation='ReLU', output_activation='Sigmoid',
                            batch_norm=True, unpool='nearest', name='resunet-a')

#Load model from checkpoint
#
model.load_weights(model_save_dir + 'model_weights_4')

print("loaded the model")
model.compile(optimizer=keras.optimizers.Adam(lr = 1e-4), loss=dice_loss, metrics=['accuracy'])

X_test = test_data
Y_test = test_target

test_length = X_test.shape[0]
test_ind = 0
Y_pred = np.zeros(shape=Y_test.shape)
printed = False
while test_length - test_ind >= 1:
    X_test_batch = []
    for counter in range(0, 1):
        X_test_batch.append(X_test[test_ind + counter])
    X_test_batch = np.array(X_test_batch)

    y_pred = model.predict(X_test_batch)
    for counter in range(0, 1):
        Y_pred[test_ind + counter] = y_pred[counter, :, :, 1]
        if not printed:
            printed = True
            #print(y_pred[counter][0, 0, :])

    test_ind += 1

Y_pred = (Y_pred >= 0.5).astype(int)

def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    #intersection = np.sum(intersection)
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (intersection + 1e-15) / (union + 1e-15),tp/(tp+fp),tp/(tp+fn)

res = mean_dice_coef(Y_test,Y_pred)
print("dice coef on test set",res)


res = compute_iou(Y_pred,Y_test)
print('iou on test set is ',res[0]," precision is ",res[1]," recall is ",res[2])

print(Y_pred.shape)

for i in range(5):
    img = Image.fromarray(X_test[i], 'RGB')
    lbl = Image.fromarray(Y_test[i]*255)
    pred = Image.fromarray((Y_pred[i]*255).astype(np.uint8))
    img.save('test/data'+ str(i+1) + '.jpg')
    lbl.convert('L').save('test/label' + str(i+1) + '.jpg')
    pred.convert('L').save('test/pred'+ str(i+1) + '.jpg')