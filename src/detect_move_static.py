import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
TIME_FRAME_INTERVAL = 120
WINDOW_SIZE = 300
PCA_DIM = 4
START_TIME = datetime.datetime(2022, 9, 17)
END_TIME = datetime.datetime(2022, 9, 20)
TRAINING_IMSI = '510010444367250'
['510018335526102', '510016935104469', '510010444367250']

# purge result folders
for folder_path in ['../img/', '../result/']:
    for root, folder, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))


data = utils.data_parsing(utils.read_raw_data("../data/150men.csv"))
data = data.loc[data['start_time'] > START_TIME]
data = data.loc[data['start_time'] < END_TIME]

# time frame
time_frame_to_idx_dict = {
    START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
        for i in range(math.ceil((END_TIME-START_TIME).total_seconds()/TIME_FRAME_INTERVAL)+1)}


# training HMM model
training_data = utils.personal_data_processing(data.loc[data['imsi']==TRAINING_IMSI], START_TIME, TIME_FRAME_INTERVAL, PCA_DIM, WINDOW_SIZE)[0]
model, reverse_switch = utils.HMM_modeling(training_data)

# predict moving/static status
with open("../result/output.csv", 'a') as f:
    w =csv.writer(f)
    w.writerow(data.columns.tolist() + ['moving_status'])

for imsi in [TRAINING_IMSI] + list(set(list(data['imsi']))):
    try:
        subset = data.loc[data['imsi']==imsi]
        subset = subset.sort_values(by='start_time')
        personal_data, enodeb_to_idx_dict_for_plot = utils.personal_data_processing(subset, START_TIME, TIME_FRAME_INTERVAL, PCA_DIM, WINDOW_SIZE)

        # moving/static status
        personal_data['status'] = utils.HMM_prediction(personal_data['signal'], model, reverse_switch)

        # mapping the status back to original dataset
        time_frame_key_to_status = {key:status for key, status in zip(personal_data['time_frame_key'], personal_data['status'])}

        with open("../result/output.csv", 'a') as f:
            w =csv.writer(f, delimiter='|')

            for row_idx in subset.index:
                key = utils.mapping_time_frame_key(subset.loc[row_idx, 'start_time'], START_TIME, TIME_FRAME_INTERVAL)
                if key in time_frame_key_to_status:
                    info = [subset.loc[row_idx, colname] for colname in subset.columns] + [time_frame_key_to_status[key]]
                    w.writerow(info)
            
            
        # result visualization
        orignal_pattern = np.zeros((len(enodeb_to_idx_dict_for_plot), len(time_frame_to_idx_dict), 3))
        condense_pattern = np.zeros((len(enodeb_to_idx_dict_for_plot), len(personal_data.index), 3))

        for idx_idx in range(len(personal_data.index)):
            idx = personal_data.index[idx_idx]
            time_idx = time_frame_to_idx_dict[personal_data['time_frame_key'][idx]]
            status = personal_data['status'][idx]
            for enodeb in personal_data['enodebs'][idx]:
                orignal_pattern[enodeb_to_idx_dict_for_plot[enodeb], time_idx, status] = 1
                condense_pattern[enodeb_to_idx_dict_for_plot[enodeb], idx_idx, status] = 1
        
        subset['time_idx'] = [(subset['start_time'].tolist()[i]-START_TIME).total_seconds()/86400 for i in range(len(subset.index))]
        fig, ax1 = plt.subplots()
        ax1.scatter(subset['time_idx'], subset['lat_first'], color='blue')
        ax1.set_ylabel("latitude")
        ax1.legend(['latitude'], loc="upper left")
        plt.xticks(rotation=30)
        ax2 = ax1.twinx()
        ax2.scatter(subset['time_idx'], subset['lon_first'], color='red')
        ax2.set_ylabel("lontitude")
        ax2.legend(['lontitude'], loc='upper right')
        plt.savefig(f"../img/{imsi}_latlon.png")
        plt.close()


        plt.imsave(f"../img/{imsi}_original_pattern.png", orignal_pattern)
        plt.imsave(f"../img/{imsi}_condense_pattern.png", condense_pattern)
        plt.plot()
        sns.lineplot(data = {
            'idx':[i for i in range(personal_data.shape[0])], 
            'signal':personal_data['signal']}
            , x='idx', y='signal')
        plt.savefig(f"../img/{imsi}_signal.png")
        plt.close()

    except Exception as error:
        with open("../result/elimiated_imsi.csv", 'a') as f:
            w = csv.writer(f)
            w.writerow([imsi])

