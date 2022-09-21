import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import csv
import datetime
import math
import pandas as pd
import numpy as np
from hmmlearn import hmm

for path in ['../img', '../result']:
    if os.path.exists(path) == False:
        os.mkdir(path)

scheme = [
    ['imei','str', True],
    ['start_time', 'datetime', True],
    ['end_time', 'datetime', True],
    ['lat_first', 'float', True],
    ['lon_first', 'float', True], 
    ['lat_last', 'float', False],
    ['lon_last', 'float', False], 
    ['moving', 'int', True], 
    ['indoor', 'int', True], 
    ['start_call_id', 'str', True], 
    ['end_call_id', 'str', True], 
    ['start_enodeb_id', 'str', True], 
    ['end_enodeb_id', 'str', True]]


def read_raw_data(file_path):
    with open(file_path) as f:
        data = []
        for i in csv.reader(f, delimiter='|'):
            data.append(i)
    
    return data

def data_parsing(raw_data):
    data = {colname:[] for colname, data_type, use_switch in scheme if use_switch}

    for i in raw_data:
        data_point = {}
        for idx in range(len(i)):
            colname, data_type, use_switch = scheme[idx]
            value = i[idx]
            try:
                if data_type == 'datetime' and use_switch:
                    data_point[colname] = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                elif data_type == 'int' and use_switch:
                    data_point[colname] = int(value)
                elif data_type == 'float' and use_switch:
                    data_point[colname] = float(value)
                elif data_type == 'str' and use_switch:
                    data_point[colname] = value
            except Exception:
                pass
        
        if len(data_point) == len(data):
            for colname in data_point:
                data[colname].append(data_point[colname])
    
    data['start_enodeb_cell'] = [f"{enodeb}_{cell}" for enodeb, cell in zip(data['start_enodeb_id'], data['start_call_id'])]
    data['end_enodeb_cell'] = [f"{enodeb}_{cell}" for enodeb, cell in zip(data['end_enodeb_id'], data['end_call_id'])]

    return pd.DataFrame(data)

def cat_to_idx_mapping(cat_list:list):
    cat_to_idx_dict = {}
    idx = 0

    for cat in cat_list:
        if cat not in cat_to_idx_dict:
            cat_to_idx_dict[cat] = idx
            idx += 1
    
    return cat_to_idx_dict

def personal_data_processing(data:pd.DataFrame):
    pass


def HMM_modeling(training_data):
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=100, algorithm='viterbi', min_covar=0.01)
    model.fit(np.array(training_data['signal']).reshape(-1,1))

    return model