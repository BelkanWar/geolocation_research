import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("result/output.csv", index_col=False, usecols=['start_time','lat_first','lon_first','moving status'])
data['timestamp'] = [datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in data['start_time'].tolist()]

lat_list = list(set(data['lat_first']))
lon_list = list(set(data['lon_first']))

lat_list.sort()
lon_list.sort()

lat_to_idx_dict = {lat_list[i]:i for i in range(len(lat_list))}
lon_to_idx_dict = {lon_list[i]:i for i in range(len(lon_list))}

# traffic density count

latlon_count_traffic_dict = {}
latlon_count_stationary_dict = {}
contour = np.zeros((len(lat_list), len(lon_list)))

for lat in lat_list:
    for lon in lon_list:
        latlon_count_traffic_dict[(lat, lon)] = 0
        latlon_count_stationary_dict[(lat, lon)] = 0

for lat, lon, moving in zip(data['lat_first'], data['lon_first'], data['moving status']):
    if moving == 1:
        latlon_count_traffic_dict[(lat, lon)] += 1
        contour[lat_to_idx_dict[lat], lon_to_idx_dict[lon]] += 1
    else:
        latlon_count_stationary_dict[(lat, lon)] += 1
    


# output_traffic = [['lat','lon','count']] + [[key[0], key[1], value] for key, value in latlon_count_traffic_dict.items()]
# output_stationary = [['lat','lon','count']] + [[key[0], key[1], value] for key, value in latlon_count_stationary_dict.items()]

# with open("result/traffic_density_full.csv",'w', newline='') as f:
#     w = csv.writer(f)
#     w.writerows(output_traffic)

# with open("result/stationary_density_full.csv",'w', newline='') as f:
#     w = csv.writer(f)
#     w.writerows(output_stationary)


# output_traffic = [['lat','lon','count']] + [[key[0], key[1], value] for key, value in latlon_count_traffic_dict.items() if value > 0]
# output_stationary = [['lat','lon','count']] + [[key[0], key[1], value] for key, value in latlon_count_stationary_dict.items() if value > 0]

# with open("result/traffic_density.csv",'w', newline='') as f:
#     w = csv.writer(f)
#     w.writerows(output_traffic)

# with open("result/stationary_density.csv",'w', newline='') as f:
#     w = csv.writer(f)
#     w.writerows(output_stationary)


plt.contour(contour, cmap='RdGy', norm='log')
plt.show()