# Forest_Detection
🌲 This is the code for my Thesis which focuses on Super Resolution and Semantic Segmentation of Satellite Images for Forest Detection.

---
### Abstract
With the ever-increasing amount of satellite missions in orbit, Earth observation and
remote sensing have advanced rapidly and are applied in a multitude of fields, such
as ecosystem monitoring and natural disaster prevention. At the same time, due to
the remarkable progress of Machine Learning in Computer Vision, combined with the
need for more efficient observations, more and more tools are developed, which rely on
Convolutional Neural Networks and attempt to map and detect pixel-scale changes on
the Earth’s surface. Such an application is forest monitoring, which is deemed nowadays
highly critical, considering the threat of deforestation on a global scale. However, the
spatial resolution of available open-source satellite imagery is limited. One way
to address this issue is to augment the spatial resolution using special Neural Network
architectures, the Image Super Resolution Networks.
This diploma thesis aims to the development of a Machine Learning system, which
processes satellite images in order to detect forest areas with the Semantic Segmentation
method, which identifies the image pixels belonging to a forest area. Initially, a dataset is
created which comprises of remote sensing images from the Sentinel-2 satellite mission along with open-source ground truth labels, which are collected from OpenStreetMap. After this, the ESRGAN architecture is used, which increases the resolution of the images and labels, resulting in having two datasets, one with super-resolved images and one with the original images. Then 3 Semantic Segmentation networks are trained on the two datasets and tested on a mutual test set, which comprises of both original and super-resolved images. The performance of the networks compared to one another, as well as the effect of the super-resolved dataset on the performance of the networks, are examined.

---
