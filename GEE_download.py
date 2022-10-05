# functions needed to download satellite images from the Google Earth Engine server

# load basic modules
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback

# earth engine module
import ee

# modules to download, unzip and stack the images
from urllib.request import urlretrieve
import zipfile
import shutil
import rasterio

# additional modules
from datetime import datetime, timedelta
import pytz


# Downloads all images from Sentinel-2 covering the area of interest and acquired
# between the specified dates in .TIF format
def retrieve_images(inputs):
    """
    Arguments
    inputs: dict with the following keys
        'polygon': (list) polygon containing the lon/lat coordinates to be extracted,
        'dates': (list of str) dates in format 'yyyy-mm-dd' (ex: ['1987-01-01', '2018-01-01'])
        'sitename': (str) name of the site
        'datapath': (str) filepath to the directory where the images are downloaded
    """

    # initialise connection with GEE server
    service_account =  'lefterislymp@beach-recognition-lefterislymp.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
    ee.Initialize(credentials)

    # (server-side) check image availability and retrieve composite image of EE collection
    print('Filtering ImageCollection to get cloud-free composite image...')
    while True:
        try:
            # chose dataset and filter
            col = (ee.ImageCollection('COPERNICUS/S2_SR')
            # filter by selected region
            .filterBounds(ee.Geometry.Rectangle(inputs['polygon']))
            # filter by selected dates
            .filterDate(inputs['dates'][0], inputs['dates'][1])
            # filter by cloud percentage
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20)
            # use cloud mask (band QA60) to mask clouds
            .map(lambda im: im.updateMask(im.select(['QA60']).lt(1)))
            # select only the bands we need
            .select(['B2', 'B3', 'B4', 'B8', 'B11']))
            col_crs = col.getInfo()['features'][0]['bands'][0]['crs']
            image = col.median().uint16().reproject(col_crs, scale=10)
            break
        except:
            traceback.print_exc()
            continue
    
    # create a new directory for this site with the name of the site
    filepath = os.path.join(inputs['datapath'], inputs['sitename'])
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # chose bands: we need multispectral(B, G, R, NIR) and SWIR bands
    bands = ['B2', 'B3', 'B4', 'B8', 'B11']
    
    # filename
    filename = inputs['sitename'] + '.tif'
        
    # geometry
    geom = ee.Geometry.Rectangle(inputs['polygon'])

    # espg
    crs = image.select(0).projection()

    # download .tif from EE
    print('Downloading image...')
    while True:
        try:
            pansharpened_image = panSharpen(image, geom, crs)
            download_tif(pansharpened_image, inputs['polygon'], bands, filepath, filename, inputs['m'])
            # only for presentation purposes
            download_tif(image, inputs['polygon'], ['B11'], filepath, 'swir_20m.tif', inputs['m'])
            break
        except Exception as e:
            traceback.print_exc()
            #continue
            break


# Downloads a .TIF image from the ee server. The image is downloaded as a zip file then
# moved to the working directory, unzipped and stacked into a single .TIF file.
def download_tif(image, polygon, bandsId, filepath, filename, m):
    """
    Arguments
    image: (ee.Image) Image object to be downloaded
    polygon: (list) polygon containing the lon/lat coordinates to be extracted,
    bandsId: (list of dict) list of bands to be downloaded
    filepath: location where the temporary file should be saved
    """

    # calculate dimensions
    # crop image on the server and create url to download
    url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
        'image': image,
        'region': polygon,
        'bands': bandsId,
        'filePerBand': 'false',
        'dimensions' : [200, 200],
        'name': 'data',
        }))

    # download zipfile with the cropped bands
    local_zip, headers = urlretrieve(url)
    
    # move zipfile from temp folder to data folder
    dest_file = os.path.join(filepath, 'imagezip')
    shutil.move(local_zip, dest_file)
    
    # unzip file
    with zipfile.ZipFile(dest_file) as local_zipfile:
        for fn in local_zipfile.namelist():
            local_zipfile.extract(fn, filepath)
        fn_tifs = [os.path.join(filepath,_) for _ in local_zipfile.namelist()]
    
    # stack bands into single .tif
    stack_tiffs(fn_tifs, os.path.join(filepath, filename))
    
    # delete single-band files and zipfile
    for fn in fn_tifs: os.remove(fn)
    os.remove(dest_file)


# Stacks bands from different files to a single .TIF file
def stack_tiffs(fn_tifs, filepath):
    """
    Arguments
    fn_tifs: (list of str) list of files to be stacked
    filepath: location where the result file should be saved
    """
    # read metadata of first file
    with rasterio.open(fn_tifs[0]) as src0:
        meta = src0.meta

    # update meta to reflect the number of layers
    meta.update(count = len(fn_tifs))

    # read each layer and write it to stack
    with rasterio.open(filepath, 'w', **meta) as dst:
        for id, layer in enumerate(fn_tifs, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


# pansharpening function that merges high-resolution panchromatic and lower
# resolution multispectral imagery to create a single high-resolution color image)
# code implementation adapted to python from
# https://code.earthengine.google.com/86ccd4ff98f3806eb7290ef0cf926782
def panSharpen(image, geometry, crs):
    #image = image.clip(geometry)

    bands10m = ['B2', 'B3', 'B4', 'B8']
    bands20m = ['B11']

    panchromatic = image.select(bands10m).reduce(ee.Reducer.mean())
    image20m = image.select(bands20m)
    image20mResampled = image20m.resample('bilinear')

    stats20m = image20m.reduceRegion(
        reducer= ee.Reducer.stdDev().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry= geometry,
        scale= 20,
        crs= crs).toImage()

    mean20m = stats20m.select('.*_mean').regexpRename('(.*)_mean', '$1')
    stdDev20m = stats20m.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

    kernel = ee.Kernel.fixed(
        width= 5,
        height= 5,
        weights= [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
        x= -3,
        y= -3,
        normalize= False )

    highPassFilter = panchromatic.convolve(kernel).rename('highPassFilter')

    stdDevHighPassFilter = highPassFilter.reduceRegion(
        reducer= ee.Reducer.stdDev(),
        geometry= geometry,
        scale= 10,
        crs= crs ).getNumber('highPassFilter')

    def calculateOutput(bandName):
        bandName = ee.String(bandName)
        W = ee.Image().expression('stdDev20m / stdDevHighPassFilter * modulatingFactor', {
            'stdDev20m': stdDev20m.select(bandName),
            'stdDevHighPassFilter': stdDevHighPassFilter,
            'modulatingFactor': 0.25 })
        return ee.Image().expression('image20mResampled + (HPF * W)', {
            'image20mResampled': image20mResampled.select(bandName),
            'HPF': highPassFilter,
            'W': W }).uint16()

    output = ee.ImageCollection([calculateOutput(b) for b in bands20m]).toBands().regexpRename('.*_(.*)', '$1')

    statsOutput = output.reduceRegion(
        reducer= ee.Reducer.stdDev().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry= geometry,
        scale= 10,
        crs= crs ).toImage()

    meanOutput = statsOutput.select('.*_mean').regexpRename('(.*)_mean', '$1')
    stdDevOutput = statsOutput.select('.*_stdDev').regexpRename('(.*)_stdDev', '$1')

    sharpened = ee.Image().expression('(output - meanOutput) / stdDevOutput * stdDev20m + mean20m', {
        'output': output,
        'meanOutput': meanOutput,
        'stdDevOutput': stdDevOutput,
        'stdDev20m': stdDev20m,
        'mean20m': mean20m }).uint16() 

    return image.addBands(sharpened, overwrite=True).select(image.bandNames())
