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
PCA_DIM = 5
TRAINING_IMEI = '510018335526102'
['510018335526102', '510016935104469']

for root, folder, files in os.walk("../img/"):
    for file in files:
        os.remove(os.path.join(root, file))


raw_data = utils.read_raw_data("../data/150men.csv")
data = utils.data_parsing(raw_data)
data = data.loc[data['start_time'] > datetime.datetime(2022, 9, 17)]
data = data.loc[data['start_time'] < datetime.datetime(2022, 9, 20)]

T_start, T_end = data['start_time'].min(), data['start_time'].max()
time_frame_to_idx_dict = {
    T_start+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
        for i in range(math.ceil((T_end-T_start).total_seconds()/TIME_FRAME_INTERVAL)+1)}


# training model
training_data = utils.personal_data_processing(data.loc[data['imei']==TRAINING_IMEI], T_start, TIME_FRAME_INTERVAL)[0]
model = utils.HMM_modeling(training_data)

for imei in [TRAINING_IMEI] + list(set(list(data['imei']))):
    try:
        subset = data.loc[data['imei']==imei]
        personal_data, enodeb_to_idx_dict_for_plot = utils.personal_data_processing(subset, T_start, TIME_FRAME_INTERVAL)

        personal_data['status'] = model.predict(np.array(personal_data['signal']).reshape(-1,1)).tolist()
        
        orignal_pattern = np.zeros((len(enodeb_to_idx_dict_for_plot), len(time_frame_to_idx_dict), 3))
        condense_pattern = np.zeros((len(enodeb_to_idx_dict_for_plot), len(personal_data.index), 3))

        for idx_idx in range(len(personal_data.index)):
            idx = personal_data.index[idx_idx]
            time_idx = time_frame_to_idx_dict[personal_data['time_frame_key'][idx]]
            status = personal_data['status'][idx]
            for enodeb in personal_data['enodebs'][idx]:
                orignal_pattern[enodeb_to_idx_dict_for_plot[enodeb], time_idx, status] = 1
                condense_pattern[enodeb_to_idx_dict_for_plot[enodeb], idx_idx, status] = 1
        
        subset['idx'] = [i for i in range(len(subset.index))]
        plt.plot()
        sns.scatterplot()


        plt.imsave(f"../img/{imei}_original_pattern.png", orignal_pattern)
        plt.imsave(f"../img/{imei}_condense_pattern.png", condense_pattern)
        plt.plot()
        sns.lineplot(data = {
            'idx':[i for i in range(personal_data.shape[0])], 
            'signal':personal_data['signal']}
            , x='idx', y='signal')
        plt.savefig(f"../img/{imei}_signal.png")
        plt.close()

    except Exception as error:
        pass   

