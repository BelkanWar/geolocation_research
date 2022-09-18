import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt

# parameters
TIME_FRAME_INTERVAL = 180


raw_data = utils.read_raw_data("../data/150_lte.csv")
data = utils.data_parsing(raw_data)
data = data.loc[data['start_time'] > datetime.datetime(2022, 9, 16, 5, 35)]
data = data.loc[data['start_time'] < datetime.datetime(2022, 9, 16, 7, 35)]

T_start, T_end = data['start_time'].min(), data['start_time'].max()
time_frame_to_idx_dict = {
    T_start+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
        for i in range(math.ceil((T_end-T_start).total_seconds()/TIME_FRAME_INTERVAL)+1)}


# training model
training_data = utils.personal_data_processing(data.loc[data['imei']=='3556201001264401'], T_start, TIME_FRAME_INTERVAL)[0]
model = utils.HMM_modeling(training_data)

for imei in set(list(data['imei'])):
    personal_data, enodeb_to_idx_dict = utils.personal_data_processing(data.loc[data['imei']==imei], T_start, TIME_FRAME_INTERVAL)
    
    try:
        personal_data['status'] = model.predict(np.array(personal_data['signal']).reshape(-1,1)).tolist()
        
        image = np.zeros((len(enodeb_to_idx_dict), len(time_frame_to_idx_dict), 3))

        for idx in personal_data.index:
            time_idx = time_frame_to_idx_dict[personal_data['time_frame_key'][idx]]
            status = personal_data['status'][idx]
            for enodeb_idx in personal_data['enodebs'][idx]:
                image[enodeb_idx, time_idx, status] = 1
        
        plt.imsave(f"../img/{imei}.png", image)

    except Exception as error:
        pass   

