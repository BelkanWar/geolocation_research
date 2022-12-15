import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
import csv
import pandas as pd

data = pd.read_csv("result/output.csv", index_col=False, usecols=['start_time','lat_first','lon_first','moving status'])
data['timestamp'] = [datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data['start_time'].tolist()]

latlon_count_dict = {}

for lat, lon, moving in zip(data['lat_first'], data['lon_first'], data['moving status']):
    if moving == 1:
        if (lat, lon) in latlon_count_dict:
            latlon_count_dict[(lat, lon)] += 1
        else:
            latlon_count_dict[(lat, lon)] = 1

output = [['lat','lon','count']] + [[key[0], key[1], value] for key, value in latlon_count_dict.items()]

with open("result/traffic_density.csv",'w', newline='') as f:
    w = csv.writer(f)
    w.writerows(output)