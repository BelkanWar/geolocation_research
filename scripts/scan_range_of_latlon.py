import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../src/")
import utils


for root, folder, files in os.walk("../splitted_data/"):
    for file in files:
        data = utils.data_parsing(utils.read_raw_data(os.path.join(root, file)))
        if 'lat' in dir():
            lat = [max(max(data['lat_first']), lat[0]), min(min(data['lat_first']), lat[1])]
            lon = [max(max(data['lon_first']), lon[0]), min(min(data['lon_first']), lon[1])]
        else:
            lat = [max(data['lat_first']), min(data['lat_first'])]
            lon = [max(data['lon_first']), min(data['lon_first'])]

print(lat, lon)

