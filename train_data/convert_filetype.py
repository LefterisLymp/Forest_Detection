import sys
import os
import geopandas
import fiona

fiona.supported_drivers['KML'] = 'rw'

if (len(sys.argv) != 3 or sys.argv[1] not in ['geojson', 'shp', 'kml']):
    print('Usage: python convert_filetype.py [geojson|shp|kml] filepath')
    sys.exit()

driver = {'geojson':'GeoJSON', 'shp':'ESRI Shapefile', 'kml':'KML'}[sys.argv[1]]
ext = {'geojson':'.GeoJSON', 'shp':'.shp', 'kml':'.kml'}[sys.argv[1]]

filepath = sys.argv[2]
(dirname, filename) = os.path.split(filepath)
filename_new = os.path.splitext(filename)[0] + ext
filepath_new = os.path.join(dirname, filename_new)
data = geopandas.read_file(filepath)
data.to_file(filepath_new, driver=driver)
