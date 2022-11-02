import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import datetime
import pickle

# parameters
VECTORIZE_METHOD = 'time_frame_base' # options: {enodeb_base, time_frame_base}
START_TIME = datetime.datetime(2022, 9, 28)
END_TIME = datetime.datetime(2022, 10, 1)
RAW_DATA_FILE_NAME = "small_jakarta_sample.csv"

TIME_FRAME_INTERVAL = 180
WINDOW_SIZE = 3600 
PCA_DIM = 4
TRAINING_IMSI = '510019860290892'
['510018710502489', '510018010344587', '510019860290892', '510017260321620']

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

pickle.dump([model, reverse_switch], open("../model/HMM_model.pkl", 'wb'))