import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import csv
import datetime
import math
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

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

def personal_data_processing(data:pd.DataFrame, T_start, time_frame_interval, dim):
    data = data.sort_values(by='start_time')
    time_frame_to_enodebs_dict = {}
    enodeb_list = []
    output = {'time_frame_key':[], 'enodebs':[], 'signal':[]}
    
    
    for start_enodeb, end_enodeb in zip(data['start_enodeb_cell'], data['end_enodeb_cell']):
        enodeb_list.append(start_enodeb)
        enodeb_list.append(end_enodeb)

    enodeb_to_idx_dict_for_plot = cat_to_idx_mapping(enodeb_list)
    

    for idx in data.index:
        key = mapping_time_frame_key(data['start_time'][idx], T_start, time_frame_interval)
        if key in time_frame_to_enodebs_dict:
            time_frame_to_enodebs_dict[key].append(data['start_enodeb_cell'][idx])
            time_frame_to_enodebs_dict[key].append(data['end_enodeb_cell'][idx])
            
        else:
            time_frame_to_enodebs_dict[key] = [
                data['start_enodeb_cell'][idx],
                data['end_enodeb_cell'][idx],
            ]

    key_enodebs_list = [[key, list(time_frame_to_enodebs_dict[key])] for key in time_frame_to_enodebs_dict]
    co_occurrence_list = [i[1] for i in key_enodebs_list]
    key_enodebs_list.sort()
    

    enodeb_vectors_array, sorted_enodeb_array, enodeb_to_idx_dict = enodebs_vectoring(co_occurrence_list, dim)
    key_enodebs_list = [i for i in key_enodebs_list if len([j for j in i[1] if j in enodeb_to_idx_dict]) > 0]

    for idx in range(1, len(key_enodebs_list)):
        _,              T1_enodebs_list = key_enodebs_list[idx-1]
        time_frame_key, T2_enodebs_list = key_enodebs_list[idx]
        T1_enodeb_idx_array = np.array([enodeb_to_idx_dict[enodeb] for enodeb in T1_enodebs_list if enodeb in enodeb_to_idx_dict])
        T2_enodeb_idx_array = np.array([enodeb_to_idx_dict[enodeb] for enodeb in T2_enodebs_list if enodeb in enodeb_to_idx_dict])
        T1_mean_vector = np.mean(enodeb_vectors_array[T1_enodeb_idx_array], axis=0).reshape((1,-1))
        T2_mean_vector = np.mean(enodeb_vectors_array[T2_enodeb_idx_array], axis=0).reshape((1,-1))
        
        distance = cosine_distances(T1_mean_vector, T2_mean_vector)[0,0]

        output['time_frame_key'].append(time_frame_key)
        output['enodebs'].append(list(set(T2_enodebs_list)))
        output['signal'].append(distance)
    
    return pd.DataFrame(output), enodeb_to_idx_dict_for_plot


def mapping_time_frame_key(time_stamp, T_start, time_frame_interval):
    delta_s = math.ceil((time_stamp - T_start).total_seconds()/time_frame_interval)*time_frame_interval
    frame_key = T_start + datetime.timedelta(seconds=delta_s)
    return frame_key

def HMM_modeling(training_data):
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=100, algorithm='viterbi', min_covar=0.001)
    model.fit(np.array(training_data['signal']).reshape(-1,1))

    return model

def enodebs_vectoring(co_occurrence_list:list, dim:int):
    co_occurrence_list = [" ".join(enodebs_list) for enodebs_list in co_occurrence_list]
    vectorizer = TfidfVectorizer(sublinear_tf = True, token_pattern = r"\S+", min_df = 3)
    decomposition = TruncatedSVD(n_components = dim, n_iter = 60)
    
    tfidf_scores = vectorizer.fit_transform(co_occurrence_list)
    decomposition.fit(tfidf_scores)

    vectors_array = decomposition.components_.T

    sorted_enodeb_array = [[vectorizer.vocabulary_[i], i] for i in vectorizer.vocabulary_]
    sorted_enodeb_array.sort()
    sorted_enodeb_array = np.array([i[1] for i in sorted_enodeb_array])
    enodeb_to_idx_dict = vectorizer.vocabulary_

    return [vectors_array, sorted_enodeb_array, enodeb_to_idx_dict]

