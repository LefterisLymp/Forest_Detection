import numpy as np
from PIL import Image

img = Image.open('train_data/2k/data_tiles_coast/5300_1714_coast.tif')
lr_img = np.array(img)

from ISR.models import RDN

rdn = RDN(weights='psnr-small')
sr_img = rdn.predict(lr_img)
sr = Image.fromarray(sr_img)
im = sr.save('test.jpg')