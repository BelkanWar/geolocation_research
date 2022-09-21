import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
TIME_FRAME_INTERVAL = 120
TRAINING_IMEI = '510016935104469'
['510018335526102', '510016935104469']

for root, folder, files in os.walk("../img/"):
    for file in files:
        os.remove(os.path.join(root, file))


raw_data = utils.read_raw_data("../data/150men.csv")
data = utils.data_parsing(raw_data)
data = data.loc[data['start_time'] > datetime.datetime(2022, 9, 17)]
data = data.loc[data['start_time'] < datetime.datetime(2022, 9, 20)]

# T_start, T_end = data['start_time'].min(), data['start_time'].max()
# time_frame_to_idx_dict = {
#     T_start+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
#         for i in range(math.ceil((T_end-T_start).total_seconds()/TIME_FRAME_INTERVAL)+1)}


# training model
# training_data = utils.personal_data_processing(data.loc[data['imei']==TRAINING_IMEI], T_start, TIME_FRAME_INTERVAL)[0]
# model = utils.HMM_modeling(training_data)

for imei in [TRAINING_IMEI] + list(set(list(data['imei']))):
    subset = data.loc[data['imei']==imei]
    subset = subset.sort_values(by='start_time')
    subset['time_idx'] = [(subset['start_time'].tolist()[i] - subset['start_time'].tolist()[0]).total_seconds() for i in range(len(subset.index))]
    
    plt.plot()
    sns.scatterplot(data=subset, x='time_idx', y='lat_first')
    plt.savefig(f"../img/{imei}_lat.png")
    plt.close()
    plt.plot()
    sns.scatterplot(data=subset, x='time_idx', y='lon_first')
    plt.savefig(f"../img/{imei}_lon.png")
    plt.close()
    
    
