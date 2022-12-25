import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("result/output.csv", index_col=False, usecols=['start_time','lat_first','lon_first','moving status'])
data['timestamp'] = [datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data['start_time'].tolist()]

lat_list = list(set(data['lat_first']))
lon_list = list(set(data['lon_first']))

lat_list.sort()
lon_list.sort()

lat_to_idx_dict = {lat_list[i]:i for i in range(len(lat_list))}
lon_to_idx_dict = {lon_list[i]:i for i in range(len(lon_list))}

TIME_FRAME_INTERVAL = 3600
START_TIME = min(data['timestamp']).replace(minute=0, second=0, microsecond=0)
END_TIME = max(data['timestamp'])


time_frame_to_idx_dict = {
    START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
        for i in range(math.ceil((END_TIME-START_TIME).total_seconds()/TIME_FRAME_INTERVAL)+1)}

for T_idx in range(math.ceil((END_TIME-START_TIME).total_seconds()/TIME_FRAME_INTERVAL)+1):
    subset = data.loc[data['timestamp'] >= START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*T_idx)]
    subset = subset.loc[subset['timestamp'] < START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*(1+T_idx))]
    time_str = (START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*T_idx)).strftime("%Y-%m-%d %H:%M:%S")

    contour = np.zeros((len(lat_list), len(lon_list)))

    for lat, lon, moving in zip(subset['lat_first'], subset['lon_first'], subset['moving status']):
        if moving == 1:
            contour[lat_to_idx_dict[lat], lon_to_idx_dict[lon]] += 1
    
    plt.plot()
    try:
        

        if 'colormap' not in dir():
            img = plt.contour(contour, norm='log')
            colormap = plt.get_cmap().colors.copy()
        else:
            img = plt.contour(contour, colors=colormap, norm='log')

        plt.savefig(f"timeframe_density/timeframe_density_{time_str}.png")
        
    except Exception:
        print(T_idx, np.sum(contour))
    plt.close()


    