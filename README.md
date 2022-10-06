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
created which comprises of remote sensing images from the Sentinel-2 satellite mission along with open-source ground truth labels, which are collected from OpenStreetMap. After this, the ESRGAN architecture is used, which increases the resolution of the images and labels, resulting in having two datasets, one with super-resolved images and one with the original images. Then 3 Semantic Segmentation networks, [ResUNet-a](https://arxiv.org/abs/1904.00592), [DeepLabv3+](https://arxiv.org/abs/1802.02611) and [Swin Transformer](https://arxiv.org/abs/2103.14030), are trained on the two datasets and tested on a mutual test set, which comprises of both original and super-resolved images. The performance of the networks compared to one another, as well as the effect of the super-resolved dataset on the performance of the networks, are examined.

---
### Dataset Creation
- At first, the ground truths have to be downloaded. In <b>ground truths/scripts</b>, the osm_data_downloader script downloads a .GeoJSON file which includes the coordinates of the forests. Then, we build a grid of the entire surface of Greece with $2km\times2km$ tiles with the grid_create script. Using the grid_reduce script, we keep only the tiles that contain the regions in the forest .GeoJSON file, resulting in a reduced grid file.
- Then, the Sentinel-2 images have to be downloaded. For this, a Google Earth Engine API key is required. Using the <b>GEE_download</b> script, the corresponding image tiles are downloaded.
- The ground truths must accompany the data tiles. To create a dataset with non-resolved images, run extract_forest.py -> make_train_data.py.<br> To create a dataset with super resolved images, run extract_forest.py -> expand_forest_images.py (super-resolution) -> make_expanded_train_data.py -> cut_sr_segment_tiles.py
- For each dataset, in the <b>train_data</b> directory, run split_files.py (or split_files_sr.py if you have SR images) in order to split the dataset into train, val and test dataset. Then run create_numpy_data.py to transform the dataset into NumPy arrays, in order to train ResUNet-a and DeepLabv3+. Run make_swin_dataset.py to make the dataset compatible with Swin Transformer. Run filtering.py if you wish to filter some of the data. Note that the images and labels are of size $200\times 200$ pixels.

<p align="middle">

  <img src="https://user-images.githubusercontent.com/64773191/194282048-1be65aed-7cbd-4fe7-85fe-825b798487c8.png" width="200" height="200"/>
  <img src="https://user-images.githubusercontent.com/64773191/194282073-97766e10-c63e-4ca4-8101-806466949ba5.png" width="200" height="200"/>

</p>

---
### Training and Testing
To train <b>ResUNet-a</b> and <b>DeepLabv3+</b>, run train.py on <b>resuneta</b> and <b>deeplabv3</b> directories respectively. To test them, run test.py. For the <b>Swin Transformer</b> [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) is used. My custom configuration file is [mmsegmentation/configs/swin/my_config.py](https://github.com/LefterisLymp/Forest_Detection/blob/main/mmsegmentation/configs/swin/my_config.py) and I provide custom implemetnations on Hybrid Loss Function (Focal Tversky Loss + IoU Loss) in [mmsegmentation/mmseg/models/losses/hybrid_loss.py](https://github.com/LefterisLymp/Forest_Detection/blob/main/mmsegmentation/mmseg/models/losses/hybrid_loss.py) and Tanimoto Loss in [mmsegmentation/mmseg/models/losses/tanimoto_loss.py](https://github.com/LefterisLymp/Forest_Detection/blob/main/mmsegmentation/mmseg/models/losses/tanimoto_loss.py)
