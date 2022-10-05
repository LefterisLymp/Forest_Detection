import os
from os import listdir
from os.path import isfile, join
#import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
import random
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import cv2
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate ,Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

from sklearn.utils import class_weight
from keras_unet_collection import losses

from focal_loss import BinaryFocalLoss

from loss import *
from model import *

from keras_unet_collection.models import resunet_a_2d

np.random.seed(123)
physical_devices = tf.config.list_physical_devices('GPU')

# Disable first GPU
tf.config.set_visible_devices(physical_devices[1:], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
# Logical device was not created for first GPU
assert len(logical_devices) == len(physical_devices) - 1

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

print("Loading training data...")
train_data = np.load('../resuneta/train_images_org_forest.npy')
train_target = np.load('../resuneta/train_labels_org_forest.npy')

print("Loading validation data...")
val_data = np.load('../resuneta/val_images_org_forest.npy')
val_target = np.load('../resuneta/val_labels_org_forest.npy')

train_data_resized = []
train_target_resized = []

dim = (256, 256)

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def eligible_label(arr):
    return np.count_nonzero(arr) > 5250 #>8% of image


print("Resizing training data...")
for ind in tqdm(range(train_data.shape[0])):
    resized = cv2.resize(train_data[ind, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
    resized_target = cv2.resize(train_target[ind, :, :], dim, interpolation=cv2.INTER_NEAREST)
    #if eligible_label(resized_target):
    train_data_resized.append(resized)
    train_target_resized.append(resized_target)
"""

    if eligible_label(resized_target[0:128, 0:128]):
        train_data_resized.append(resized[0:128, 0:128, :])
        train_target_resized.append(resized_target[0:128, 0:128])

    if eligible_label(resized_target[0:128, 128:256]):
        train_data_resized.append(resized[0:128, 128:256, :])
        train_target_resized.append(resized_target[0:128, 128:256])

    if eligible_label(resized_target[128:256, 0:128]):
        train_data_resized.append(resized[128:256, 0:128, :])
        train_target_resized.append(resized_target[128:256, 0:128])

    if eligible_label(resized_target[128:256, 128:256]):
        train_data_resized.append(resized[128:256, 128:256, :])
        train_target_resized.append(resized_target[128:256, 128:256])

"""
train_data = np.array(train_data_resized)
train_target = np.array(train_target_resized)

print("Train_data_shape:", train_data.shape)
print("Train labels shape:", train_target.shape)

val_data_resized = []
val_target_resized = []

logging.basicConfig(filename="deeplab-training_forest_org_tanimoto.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


#---------------------------

"""
print("Resizing training data...")
for ind in tqdm(range(train_data.shape[0])):
    resized = cv2.resize(train_data[ind, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
    resized_target = cv2.resize(train_target[ind, :, :], dim, interpolation=cv2.INTER_NEAREST)
    train_data_resized.append(resized)
    train_target_resized.append(resized_target)

train_data_resized = np.array(train_target_resized)
train_target_resized = np.array(train_target_resized)
"""

#---------------------------


print("Resizing validation data...")
for ind in tqdm(range(val_data.shape[0])):
    resized = cv2.resize(val_data[ind, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
    resized_target = cv2.resize(val_target[ind, :, :], dim, interpolation=cv2.INTER_NEAREST)

    val_data_resized.append(resized)
    val_target_resized.append(resized_target)

"""
    val_data_resized.append(resized[0:128, 0:128, :])
    val_target_resized.append(resized_target[0:128, 0:128])

    val_data_resized.append(resized[0:128, 128:256, :])
    val_target_resized.append(resized_target[0:128, 128:256])

    val_data_resized.append(resized[128:256, 0:128, :])
    val_target_resized.append(resized_target[128:256, 0:128])

    val_data_resized.append(resized[128:256, 128:256, :])
    val_target_resized.append(resized_target[128:256, 128:256])
"""

val_data_resized = np.array(val_data_resized)
val_target_resized = np.array(val_target_resized)

#train_data = train_data_resized
#train_target = train_target_resized

val_data = val_data_resized
val_target = val_target_resized


train_target = keras.utils.to_categorical(train_target, num_classes=2)
#val_target = keras.utils.to_categorical(val_target, num_classes=2)

#SIZE_1 = SIZE_2 = 200

train_data_length = train_data.shape[0]

class_weights = [1.0, 1.0]
size = (256, 256)

loss_fn = multiclass_weighted_tanimoto_loss(class_weights)
bn_loss = BinaryFocalLoss(gamma=0)

def hybrid_loss(y_true, y_pred):
    #loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
    loss_iou = iou_seg(y_true, y_pred)

    return loss_iou

def train(epochs, batch_size, model_save_dir):
    max_val_dice = -1
    batch_count = train_data_length // batch_size
    model = DeeplabV3Plus(image_size=size[0], num_classes=2)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Load model from checkpoint
    #
    #model.load_weights(model_save_dir + 'model_weights_29')

    model.compile(optimizer=keras.optimizers.Adam(lr = 1e-4), loss=loss_fn, metrics=['accuracy'])
    model.summary()
    for e in range(0, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, ' out of ', epochs, '-' * 15)
        # sp startpoint
        for sp in tqdm(range(0, batch_count, 1)): #batch_count
            if (sp + 1) * batch_size > train_data_length:
                batch_end = train_data_length
            else:
                batch_end = (sp + 1) * batch_size

            X_batch = []
            Y_batch = []

            for ind in range((sp * batch_size), batch_end):
                X_batch.append(train_data[ind, :, :, :])
                Y_batch.append(train_target[ind, :, :])

            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)
            #Y_batch = keras.utils.to_categorical(Y_batch, num_classes=2)

            model.train_on_batch(X_batch, Y_batch)


        X_val = val_data
        Y_val = val_target

        val_length = X_val.shape[0]
        val_ind = 0
        Y_pred = np.zeros(shape=Y_val.shape)
        while val_length - val_ind >= batch_size:
            X_val_batch = []
            for counter in range(0, batch_size):
                X_val_batch.append(X_val[val_ind + counter])
            X_val_batch = np.array(X_val_batch)

            y_pred = model.predict(X_val_batch)
            for counter in range(0, batch_size):
                Y_pred[val_ind + counter] = y_pred[counter, :, :, 1]
            val_ind += batch_size

        #Y_pred = np.argmax(Y_pred, axis=3)
        Y_pred = (Y_pred >= 0.5).astype(int)
        #Y_val = np.argmax(Y_val, axis=3)

        res = mean_dice_coef(Y_val, Y_pred)
        print('Mean Dice Coefficient on Validation:', res)
        #model.save(model_save_dir + 'model_' + str(e))
        #model.save_weights(model_save_dir + 'model_weights_' + str(e))

        if (res > max_val_dice):
            max_val_dice = res
            logger.info("Best epoch: " + str(e) + "with MDC: " + str(res))
            model.save(model_save_dir + 'model_' + str(e) )
            model.save_weights(model_save_dir + 'model_weights_' + str(e))
            print('New Val_Dice HighScore', res)


model_save_dir = 'checkpoints_forest_org_tanimoto/'
train(100, 2, model_save_dir)