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

def personal_data_processing(data:pd.DataFrame, T_start, time_frame_interval):
    data = data.sort_values(by='start_time')
    time_frame_to_enodeb_id_dict = {}
    enodeb_list = []
    output = {'time_frame_key':[], 'enodebs':[], 'signal':[]}
    
    
    for start_enodeb, end_enodeb in zip(data['start_enodeb_cell'], data['end_enodeb_cell']):
        enodeb_list.append(start_enodeb)
        enodeb_list.append(end_enodeb)

    enodeb_to_idx_dict = cat_to_idx_mapping(enodeb_list)
    

    for idx in data.index:
        key = mapping_time_frame_key(data['start_time'][idx], T_start, time_frame_interval)
        if key in time_frame_to_enodeb_id_dict:
            time_frame_to_enodeb_id_dict[key].update([
                enodeb_to_idx_dict[data['start_enodeb_cell'][idx]], 
                enodeb_to_idx_dict[data['end_enodeb_cell'][idx]]])
            
        else:
            time_frame_to_enodeb_id_dict[key] = set([
                enodeb_to_idx_dict[data['start_enodeb_cell'][idx]],
                enodeb_to_idx_dict[data['end_enodeb_cell'][idx]],
            ])

    key_enodeb_id_list = [[key, list(time_frame_to_enodeb_id_dict[key])] for key in time_frame_to_enodeb_id_dict]
    key_enodeb_id_list.sort()

    for idx in range(1, len(key_enodeb_id_list)):
        time_frame_key, enodebs_idx_list = key_enodeb_id_list[idx]

        output['time_frame_key'].append(time_frame_key)
        output['enodebs'].append(enodebs_idx_list)
        output['signal'].append(np.mean(enodebs_idx_list) - np.mean(key_enodeb_id_list[idx-1][1]))
    
    return pd.DataFrame(output), enodeb_to_idx_dict


def mapping_time_frame_key(time_stamp, T_start, time_frame_interval):
    delta_s = math.ceil((time_stamp - T_start).total_seconds()/time_frame_interval)*time_frame_interval
    frame_key = T_start + datetime.timedelta(seconds=delta_s)
    return frame_key

def HMM_modeling(training_data):
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=100, algorithm='viterbi', min_covar=0.001)
    model.fit(np.array(training_data['signal']).reshape(-1,1))

    return model