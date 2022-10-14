import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import math
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
VECTORIZE_METHOD = 'time_frame_base' # options: {enodeb_base, time_frame_base}
START_TIME = datetime.datetime(2022, 9, 28)
END_TIME = datetime.datetime(2022, 10, 1)
RAW_DATA_FILE_NAME = "small_jakarta_sample.csv"

TIME_FRAME_INTERVAL = 120
WINDOW_SIZE = 500
PCA_DIM = 4
TRAINING_IMSI = '510017060321551'
['510018335526102', '510011647797943', '510010444367250']

# purge result folders
for folder_path in ['../img/', '../result/', '../splitted_data']:
    for root, folder, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))
print('old data purge completed')

# time frame
time_frame_to_idx_dict = {
    START_TIME+datetime.timedelta(seconds=TIME_FRAME_INTERVAL*i):i 
        for i in range(math.ceil((END_TIME-START_TIME).total_seconds()/TIME_FRAME_INTERVAL)+1)}

# split raw data by imsi, save data of different imsi in 'splitted_data' seperately
utils.split_raw_data_by_imsi(f"../data/{RAW_DATA_FILE_NAME}")


# training HMM model

training_data = utils.personal_data_processing(
    utils.data_parsing(utils.read_raw_data(f"../splitted_data/{TRAINING_IMSI}.csv")), 
    START_TIME, 
    END_TIME, 
    TIME_FRAME_INTERVAL, 
    PCA_DIM, 
    WINDOW_SIZE, 
    VECTORIZE_METHOD)[0]

model, reverse_switch = utils.HMM_modeling(training_data)


# predict moving/static status
imsi_list = []
for root, folder, files in os.walk("../splitted_data"):
    for file in files:
        imsi_list.append(file.split(sep='.')[0])



def core_job(imsi):
    try:
        file_path = f"../splitted_data/{imsi}.csv"
        subset = utils.data_parsing(utils.read_raw_data(file_path))
        personal_data, enodeb_to_idx_dict_for_plot = utils.personal_data_processing(subset, START_TIME, END_TIME, TIME_FRAME_INTERVAL, PCA_DIM, WINDOW_SIZE, VECTORIZE_METHOD)
        

        # moving/static status
        personal_data['status'] = utils.HMM_prediction(personal_data['signal'], model, reverse_switch)

        # mapping the status back to original dataset
        time_frame_key_to_status = {key:status for key, status in zip(personal_data['time_frame_key'], personal_data['status'])}

        with open("../result/output.csv", 'a', newline='') as f:
            w = csv.writer(f)

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
        
        # subset['time_idx'] = [(subset['start_time'].tolist()[i]-START_TIME).total_seconds()/86400 for i in range(len(subset.index))]
        # fig, ax1 = plt.subplots()
        # ax1.scatter(subset['time_idx'], subset['lat_first'], color='blue')
        # ax1.set_ylabel("latitude")
        # ax1.legend(['latitude'], loc="upper left")
        # plt.xticks(rotation=30)
        # ax2 = ax1.twinx()
        # ax2.scatter(subset['time_idx'], subset['lon_first'], color='red')
        # ax2.set_ylabel("lontitude")
        # ax2.legend(['lontitude'], loc='upper right')
        # plt.savefig(f"../img/{imsi}_latlon.png")
        # plt.close()


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
        with open("../result/elimiated_imsi.csv", 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([imsi])

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    for idx in range(0, len(imsi_list), 30):
        pool.map(core_job, imsi_list[idx:idx+30])


