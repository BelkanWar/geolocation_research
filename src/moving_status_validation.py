'''
視覺化covmo所標記的moving標籤的合理性
輸出圖片：橫軸為時間，縱軸為lat/lon，每個時間點會有兩個點
紅色點為靜止(moving=0), 綠色為移動(moving=1)
'''

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt


for root, folder, files in os.walk("../img/"):
    for file in files:
        os.remove(os.path.join(root, file))

# parameters
START_TIME, END_TIME = datetime.datetime(2022, 9, 17), datetime.datetime(2022, 9, 20)
IMG_HIGHT = 300
IMG_WEIGHT = 1000


data = utils.data_parsing(utils.read_raw_data("../data/150men.csv"))
data = data.loc[data['start_time'] > START_TIME]
data = data.loc[data['start_time'] < END_TIME]

for imsi in set(data['imsi']):
    try:
        subset = data.loc[data['imsi'] == imsi]
        subset = subset.sort_values(by='start_time')

        subset['time_idx'] = [math.ceil(IMG_WEIGHT*(i-START_TIME).total_seconds()/(END_TIME-START_TIME).total_seconds()) for i in subset['start_time']]
        max_lat, min_lat = subset['lat_first'].max(), subset['lat_first'].min()
        max_lon, min_lon = subset['lon_first'].max(), subset['lon_first'].min()

        subset['lat'] = [round((i-min_lat)*IMG_HIGHT/(max_lat-min_lat)) for i in subset['lat_first']]
        subset['lon'] = [round((i-min_lon)*IMG_HIGHT/(max_lon-min_lon)) for i in subset['lon_first']]

        image = np.zeros((IMG_HIGHT+1, IMG_WEIGHT+1, 3))

        for lat, lon, time_idx, moving in zip(subset['lat'], subset['lon'], subset['time_idx'], subset['moving']):
            image[lat, time_idx, moving] = 1
            image[lon, time_idx, moving] = 1
        
        plt.imsave(f"../img/{imsi}.png", image)
    except Exception as error:
        pass

    
