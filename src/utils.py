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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

for path in ['../img', '../result', '../splitted_data']:
    if os.path.exists(path) == False:
        os.mkdir(path)

scheme = [
    ['imsi','str', True],
    ['start_time', 'datetime', True],
    ['end_time', 'datetime', True],
    ['lat_first', 'float', True],
    ['lon_first', 'float', True], 
    ['lat_last', 'float', False],
    ['lon_last', 'float', False], 
    ['moving', 'int', True], 
    ['indoor', 'int', True], 
    ['start_cell_id', 'str', True], 
    ['end_cell_id', 'str', True], 
    ['start_enodeb_id', 'str', True], 
    ['end_enodeb_id', 'str', True]]


def read_raw_data(file_path):
    with open(file_path) as f:
        data = []
        for i in csv.reader(f):
            data.append(i)
    
    return data

def split_raw_data_by_imsi(file_path):
    imsi_idx = [i[0] for i in scheme].index('imsi')
    with open(file_path) as f_read:
        for i in csv.reader(f_read, delimiter='|'):
            imsi = i[imsi_idx]
            if len(i) == len(scheme):
                with open(f"../splitted_data/{imsi}.csv", 'a', newline='') as f_write:
                    w = csv.writer(f_write)
                    w.writerow(i)

def find_latlon_range(file_path):
    lat_idx = [i[0] for i in scheme].index('lat_first')
    lon_idx = [i[0] for i in scheme].index('lon_first')
    lat_range = {'min':10000, 'max':-10000}
    lon_range = {'min':10000, 'max':-10000}
    with open(file_path) as f_read:
        for i in csv.reader(f_read, delimiter='|'):
            if len(i) == len(scheme):
                lat_range['min'] = min(lat_range['min'], float(i[lat_idx]))
                lat_range['max'] = max(lat_range['max'], float(i[lat_idx]))
                lon_range['min'] = min(lon_range['min'], float(i[lon_idx]))
                lon_range['max'] = max(lon_range['max'], float(i[lon_idx]))
    
    return [lat_range, lon_range]

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
    
    data['start_enodeb_cell'] = [f"{enodeb}" for enodeb, cell in zip(data['start_enodeb_id'], data['start_cell_id'])]
    data['end_enodeb_cell'] = [f"{enodeb}" for enodeb, cell in zip(data['end_enodeb_id'], data['end_cell_id'])]
    data = pd.DataFrame(data)
    data = data.sort_values(by='start_time')

    return data

def cat_to_idx_mapping(cat_list:list):
    cat_to_idx_dict = {}
    idx = 0

    for cat in cat_list:
        if cat not in cat_to_idx_dict:
            cat_to_idx_dict[cat] = idx
            idx += 1
    
    return cat_to_idx_dict

def personal_data_processing(data:pd.DataFrame, T_start:datetime.datetime, T_END:datetime.datetime, time_frame_interval:int, pca_dim:int, window_size:int, vectorize_method:str):

    data = data.loc[data['start_time'] > T_start]
    data = data.loc[data['start_time'] < T_END]
    
    data = data.sort_values(by='start_time')
    time_frame_to_enodebs_dict = {}
    enodeb_list = []
    output = {'time_frame_key':[], 'enodebs':[], 'signal':[], 'interval':[]}
    
    
    for start_enodeb, end_enodeb in zip(data['start_enodeb_cell'], data['end_enodeb_cell']):
        enodeb_list.append(start_enodeb)
        # enodeb_list.append(end_enodeb)

    enodeb_to_idx_dict_for_plot = cat_to_idx_mapping(enodeb_list)
    

    for idx in data.index:
        key = mapping_time_frame_key(data['start_time'][idx], T_start, time_frame_interval)
        if key in time_frame_to_enodebs_dict:
            time_frame_to_enodebs_dict[key].append(data['start_enodeb_cell'][idx])
        else:
            time_frame_to_enodebs_dict[key] = [data['start_enodeb_cell'][idx]]

    # filter out imsi without enought data point
    if len(time_frame_to_enodebs_dict) < 30 or len(enodeb_to_idx_dict_for_plot) < 10:
        return ''

    # conduct enodeb/time_frame vectorization, and HMM model
    key_enodebs_list = [[key, list(time_frame_to_enodebs_dict[key])] for key in time_frame_to_enodebs_dict]
    key_enodebs_list.sort()
    co_occurrence_list = [i[1] for i in key_enodebs_list]
    

    if vectorize_method == 'enodeb_base':
        enodeb_vectors_array, enodeb_to_idx_dict = enodebs_vectoring(data, pca_dim, window_size)
        key_enodebs_list = [i for i in key_enodebs_list if len([j for j in i[1] if j in enodeb_to_idx_dict]) > 0]
    else:
        time_frame_vectors_array = time_frame_vectoring(co_occurrence_list, pca_dim)
    

    for idx in range(1, len(key_enodebs_list)):
        T1_key, T1_enodebs_list = key_enodebs_list[idx-1]
        T2_key, T2_enodebs_list = key_enodebs_list[idx]

        if vectorize_method == 'enodeb_base':
            T1_enodeb_idx_array = np.array([enodeb_to_idx_dict[enodeb] for enodeb in T1_enodebs_list if enodeb in enodeb_to_idx_dict])
            T2_enodeb_idx_array = np.array([enodeb_to_idx_dict[enodeb] for enodeb in T2_enodebs_list if enodeb in enodeb_to_idx_dict])
            T1_vector_sum = np.mean(enodeb_vectors_array[T1_enodeb_idx_array], axis=0).reshape((1,-1))
            T2_vector_sum = np.mean(enodeb_vectors_array[T2_enodeb_idx_array], axis=0).reshape((1,-1))
        
        else:
            T1_vector_sum = time_frame_vectors_array[idx-1].reshape((1,-1))
            T2_vector_sum = time_frame_vectors_array[idx].reshape((1,-1))

        signal = cosine_distances(T1_vector_sum, T2_vector_sum)[0,0]

        output['time_frame_key'].append(T2_key)
        output['enodebs'].append(list(set(T2_enodebs_list)))
        output['signal'].append(signal)
        output['interval'].append((T2_key-T1_key).total_seconds())
    
    return pd.DataFrame(output), enodeb_to_idx_dict_for_plot


def mapping_time_frame_key(time_stamp:datetime.datetime, T_start:datetime.datetime, time_frame_interval:int):
    delta_s = math.ceil((time_stamp - T_start).total_seconds()/time_frame_interval)*time_frame_interval
    frame_key = (T_start + datetime.timedelta(seconds=delta_s))
    return frame_key

def HMM_modeling(training_data):
    input_array = np.array(training_data['signal']).reshape(-1,1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=100, algorithm='viterbi', min_covar=0.1)
    model.fit(input_array)

    prediction = model.predict(input_array)
    mean_zero_status = np.mean(input_array[np.where(prediction==0)])
    mean_one_status = np.mean(input_array[np.where(prediction==1)])

    if mean_zero_status > mean_one_status:
        reverse_switch = True
    else:
        reverse_switch = False

    return model, reverse_switch

def HMM_prediction(signal_list, model, reverse_switch):
    prediction = model.predict(np.array(signal_list).reshape(-1,1))
    if reverse_switch:
        prediction = np.absolute(prediction-1)
    
    return prediction.tolist()


def enodebs_vectoring(data:pd.DataFrame, pca_dim:int, window_size:int):
    co_occurrence_list = []

    for idx in data.index:
        subset = data.loc[data['start_time'] > data.loc[idx,'start_time'] - datetime.timedelta(seconds=window_size/2)]
        subset = subset.loc[data['start_time'] < data.loc[idx, 'start_time'] +  datetime.timedelta(seconds=window_size/2)]
        enodeb_list = data.loc[subset.index, 'start_enodeb_cell'].tolist()
        if len(enodeb_list) == 1 and enodeb_list[0] in co_occurrence_list:
            pass
        else:
            co_occurrence_list.append(" ".join(enodeb_list))


    vectorizer = CountVectorizer(token_pattern = r"\S+", min_df = 1)
    tfidf_scores = vectorizer.fit_transform(co_occurrence_list)

    # decomposer = TruncatedSVD(n_components = min(pca_dim, math.ceil(len(vectorizer.vocabulary_)/10)+1), n_iter = 60)
    decomposer = TruncatedSVD(n_components = pca_dim, n_iter = 60)

    decomposer.fit(tfidf_scores.toarray())

    vectors_array = decomposer.components_.T
    enodeb_to_idx_dict = vectorizer.vocabulary_

    return [vectors_array, enodeb_to_idx_dict]

def time_frame_vectoring(co_occurrence_list:list, pca_dim:int):
    co_occurrence_list = [" ".join(enodebs_list) for enodebs_list in co_occurrence_list]
    vectorizer = TfidfVectorizer(sublinear_tf = True, token_pattern = r"\S+", min_df = 1)
    tfidf_scores = vectorizer.fit_transform(co_occurrence_list)

    decomposer = TruncatedSVD(n_components = min(pca_dim, math.ceil(len(vectorizer.vocabulary_)/10)+1), n_iter = 60)
    decomposer.fit(tfidf_scores.toarray())
    vectors_array = decomposer.transform(tfidf_scores.toarray())

    return vectors_array