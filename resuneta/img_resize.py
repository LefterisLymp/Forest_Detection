import numpy as np
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)


val_images = np.load('val_images_final.npy')
val_labels = np.load('val_labels_final.npy')

scale_percent = 128
dim = (256, 256)

resized = cv2.resize(val_images[1, :, :, :], dim, interpolation=cv2.INTER_NEAREST)
target_resized = cv2.resize(val_labels[1, :, :], dim, interpolation=cv2.INTER_NEAREST)

print(resized.shape)
print(target_resized.shape)

print(val_labels[1, : :])
print(target_resized)