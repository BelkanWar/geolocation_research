import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../src/")
import csv
import datetime
import matplotlib.pyplot as plt
import random
import pandas as pd


# parameters
interval = 100
new_method = True
fix_map = True


with open("../result/output.csv") as f:
    data = {}
    for i in csv.reader(f):

        imsi = i[0]

        if imsi != 'imsi':

            if imsi not in data:
                data[imsi] = {'timestamp':[], 'lat':[], 'lon':[], 'moving':[]}

            data[imsi]['timestamp'].append(datetime.datetime.strptime(i[1],'%Y-%m-%d %H:%M:%S'))
            data[imsi]['lat'].append(float(i[3]))
            data[imsi]['lon'].append(float(i[4]))
            if new_method:
                data[imsi]['moving'].append(int(i[-1]))
            else:
                data[imsi]['moving'].append(int(i[5]))


global_lat_range = [
    min([min(data[imsi]['lat']) for imsi in data]),
    max([max(data[imsi]['lat']) for imsi in data])]
global_lon_range = [
    min([min(data[imsi]['lon']) for imsi in data]),
    max([max(data[imsi]['lon']) for imsi in data])]

imsi_list = list(data)
random.shuffle(imsi_list)
imsi_list = ['510018610002079']+imsi_list

for imsi in imsi_list:

    print(imsi)
    imsi_data = pd.DataFrame(data[imsi])
    imsi_data = imsi_data.sort_values(by='timestamp')

    lat_list = list(imsi_data['lat'])
    lon_list = list(imsi_data['lon'])
    moving_list = list(imsi_data['moving'])
    timestamp_list = list(imsi_data['timestamp'])

    start_time, end_time = timestamp_list[0], timestamp_list[-1]
    interval_seconds = int((end_time - start_time).total_seconds()+10)
    convert_data = {'timestamp':[], 'lat':[], 'lon':[], 'moving':[]}

    idx = 0

    for sec in range(interval_seconds):
        
        if start_time + datetime.timedelta(seconds=sec) >= timestamp_list[idx]:
            convert_data['timestamp'].append(start_time + datetime.timedelta(seconds=sec))
            convert_data['lat'].append(lat_list[idx])
            convert_data['lon'].append(lon_list[idx])
            convert_data['moving'].append('red' if moving_list[idx]==0 else 'green')
            idx += 1
        
        if idx >= len(timestamp_list):
            break
    
    plt.ion()
    plt.show()

    if fix_map:
        lat_range = global_lat_range
        lon_range = global_lon_range
    else:
        lat_range = [min(convert_data['lat']), max(convert_data['lat'])]
        lon_range = [min(convert_data['lon']), max(convert_data['lon'])]
    

    for idx in range(1, len(convert_data['timestamp']), 5):
        
        plt.cla()
        plt.scatter(
            convert_data['lon'][max(0, idx-interval):idx], 
            convert_data['lat'][max(0, idx-interval):idx], 
            c=convert_data['moving'][max(0, idx-interval):idx])
        plt.plot(
            convert_data['lon'][max(0, idx-interval):idx], 
            convert_data['lat'][max(0, idx-interval):idx])
        plt.xlim(lon_range)
        plt.ylim(lat_range)
        plt.text(
            lon_range[0], 
            lat_range[0], 
            convert_data['timestamp'][idx].strftime("%Y-%m-%d %H:%M:%S"),
            fontdict={'size':20, 'color':'red'})
        plt.pause(0.0001)
    
    plt.pause(2)


