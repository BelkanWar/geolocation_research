import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
import math
import shapefile
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


norm = 'log'
epsilon = sys.float_info.epsilon

shape = shapefile.Reader("resources/admin/idn_admbnda_adm1_bps_20200401.shp")

for shapeRecord in shape.shapeRecords():
    if shapeRecord.record[2] == 'Dki Jakarta':
        jakarta_shapes = shapeRecord.shape.__geo_interface__['coordinates']
    elif shapeRecord.record[2] == 'Jawa Barat':
        barat_shapes = shapeRecord.shape.__geo_interface__['coordinates']
    elif shapeRecord.record[2] == 'Banten':
        banten_shapes = shapeRecord.shape.__geo_interface__['coordinates']

jawa_coord = []

for shapes in [jakarta_shapes, barat_shapes, banten_shapes]:
    shapeCoords_length_list = [[shapeCoord[0], len(shapeCoord[0])] for shapeCoord in shapes]
    shapeCoords_length_list.sort(key=lambda x:x[1], reverse=True)
    shapeCoord = shapeCoords_length_list[0][0]
    jawa_coord.append([[i[0] for i in shapeCoord], [i[1] for i in shapeCoord]])


if os.path.exists("timeframe_density/") == False:
    os.mkdir("timeframe_density/")
else:
    for root, folder, files in os.walk("timeframe_density/"):
        for file in files:
            os.remove(os.path.join(root, file))


data = pd.read_csv("result/output.csv", index_col=False, usecols=['start_time','lat_first','lon_first','moving status'])
data['timestamp'] = [datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data['start_time'].tolist()]

lat_list = list(set(data['lat_first']))
lon_list = list(set(data['lon_first']))

lat_list.sort()
lon_list.sort()

lat_to_idx_dict = {lat_list[i]:i for i in range(len(lat_list))}
lon_to_idx_dict = {lon_list[i]:i for i in range(len(lon_list))}

contour = np.zeros((len(lat_list), len(lon_list)))

for lat, lon, moving in zip(data['lat_first'], data['lon_first'], data['moving status']):
    if moving == 1:
        contour[lat_to_idx_dict[lat], lon_to_idx_dict[lon]] += 1

img = plt.contour(lon_list, lat_list, contour, cmap='Reds', norm=norm)
levels = img.levels.copy()
xlim = plt.xlim()
ylim = plt.ylim()

TIME_FRAME_INTERVAL = 3600
START_TIME = min(data['timestamp']).replace(minute=0, second=0, microsecond=0)
END_TIME = max(data['timestamp'])


for T_idx in range(math.ceil((END_TIME-START_TIME).total_seconds()/TIME_FRAME_INTERVAL)+1):
    subset = data.loc[data['timestamp'] >= START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*T_idx)]
    subset = subset.loc[subset['timestamp'] < START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*(1+T_idx))]
    time_str = (START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*T_idx)).strftime("%Y-%m-%d %H:%M:%S")

    if subset.shape[0] > 0:
        contour = np.zeros((len(lat_list), len(lon_list)))

        for lat, lon, moving in zip(subset['lat_first'], subset['lon_first'], subset['moving status']):
            if moving == 1:
                contour[lat_to_idx_dict[lat], lon_to_idx_dict[lon]] += 1
        
        plt.plot()
        try:
            if 'levels' not in dir():
                img = plt.contour(lon_list, lat_list, contour, cmap='Reds', norm=norm)
                levels = img.levels.copy()
            else:
                if norm == 'log':
                    img = plt.contour(lon_list, lat_list, contour, cmap='Reds', levels=np.log10(levels+epsilon))
                else:
                    img = plt.contour(lon_list, lat_list, contour, cmap='Reds', levels=levels)

            for jawa_X, jawa_Y in jawa_coord:
                plt.plot(jawa_X, jawa_Y)

            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.text(
                xlim[0], 
                ylim[0], 
                time_str,
                fontdict={'size':20, 'color':'blue'})
            plt.savefig(f"timeframe_density/timeframe_density_{time_str}.png")
            
        except Exception as error:
            print(error)
        plt.close()

images = []

for root, folder, files in os.walk("timeframe_density/"):
    for file in files:
        images.append([
            datetime.datetime.strptime(file, 'timeframe_density_%Y-%m-%d %H:%M:%S.png'), 
            imageio.imread(os.path.join(root,file))])

images.sort()
images = [i[1] for i in images]

imageio.mimsave("timeframe_density/result.gif", images, duration=1)
