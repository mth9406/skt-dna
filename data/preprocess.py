import argparse 
import pandas as pd 
import numpy as np 
import torch 

import os
from time import time
from tqdm import tqdm 
from functools import reduce 

import pickle
import json
import csv 

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type= str, default= './skt', 
                help= 'a path to the data (default= \'./\'')
parser.add_argument('--split_data_path', type= str, default= './skt_split')
parser.add_argument('--split_data_adj_path', type= str, default= './skt_split_adj')
# parser.add_argument('--time_lag', type= int, default= 2)

parser.add_argument('--verbose', action= 'store_true', 
                help= 'to print the preprocess-logs')
# parser.add_argument('--save_as_files', action= 'store_true', 
#                 help= 'to save every window in a split file')

args = parser.parse_args() 

# make a path to save some new data 
if not os.path.exists(args.split_data_path): 
    print("making a path to save splitted data...")
    os.makedirs(args.split_data_path, exist_ok= True) 
else: 
    print("The path already exists, skip making the path...")

# args.split_data_adj_path = args.split_data_adj_path+'_'+str(args.time_lag)
# args.split_data_adj_path = args.split_data_adj_path
if not os.path.exists(args.split_data_adj_path): 
    print("making a path to save time-series data...")
    os.makedirs(args.split_data_adj_path, exist_ok= True) 
else: 
    print("The path already exists, skip making the path...")

def merge_duplicate(data):
    """
    merges duplicate 'Time-Stamp' in the data and return the data (df)
    columns of the data: 
    [ 'Time_Stamp', 
      'RRC_CNT', 'RRC_FAIL_RATE',
      'CALL_RELEASE_ANOMALY_CNT', 'DL_PRB', 'CQI', 'RSRP', 'RSRQ',
      'UPLINK_SINR', 'UE_TX_POWER', 'TA', 'PRIMARY_KEY']

    """
    df = pd.DataFrame([], columns= data.columns)
    df = df.append(data.iloc[0, :])
    cnt = [1]
    for i in range(1, data.shape[0]): 
        if data.iloc[i, 0] == df.iloc[-1, 0]: 
            m_prev = ~df.iloc[-1, 1:-1].isna().values
            m_next = ~data.iloc[i, 1:-1].isna().values
            idx_tf = np.argwhere(np.logical_xor(m_prev, m_next)).flatten() +1
            idx_tt = np.argwhere(np.logical_and(m_prev, m_next)).flatten() +1
            idx_ff = np.argwhere(np.logical_and(~m_prev, ~m_next)).flatten() +1
            df.iloc[-1, idx_tt] = df.iloc[-1, idx_tt]+data.iloc[i, idx_tt] # average
            df.iloc[-1, idx_ff] = float('nan')
            df.iloc[-1, idx_tf] = df.iloc[-1, idx_tf].fillna(0) + data.iloc[i, idx_tf].fillna(0)
            cnt[-1] += 1
        else: 
            df = df.append(data.iloc[i, :])
            cnt.append(1) 
    cnt = np.array(cnt)[:, np.newaxis]
    df.iloc[:, 2:-1] = df.iloc[:, 2:-1] / cnt 
    # aggregate RRC_CNT as 'sum' 
    # remainings are aggregated as 'mean'
    return df

def main(args):  

    # read data
    print('loading data....')
    file = os.path.join(args.data_path, 'ChristmasWeek_KPI_Gangnam.csv')
    data = pd.read_csv(file)
    data = data.drop(['Unnamed: 0', 'ADONG_CD'], axis= 1) 
    print('loading data done!')
    print(data.info())

    # find the primary keys 
    print('finding and converting the ENB_ID, CELL_NO pair to a number')
    print('It takes some time... (about 7 minutes...)')
    primary_keys = np.unique(data[['ENB_ID', 'CELL_NO']].values, axis= 0)
    idx = np.argsort(primary_keys[:, 0])
    primary_keys = primary_keys[idx] 
    # sort the keys by the ENB_ID so that cells with the same ENB stay together.

    print('saving the primary_keys in csv format...')
    primary_keys_csv_file = os.path.join('./', 'primary_keys.csv')
    with open(primary_keys_csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for i, key in enumerate(primary_keys.tolist()): 
            writer.writerow([i] + key) 
    print('saving the primary_keys in csv format done!')
    print(f'saved the file in the path: \'./\'')
    print('converting the primary keys started ...')
    
    ts = time() 
    primary_key_nums = []
    for key in tqdm(data[['ENB_ID', 'CELL_NO']].values, total= data.shape[0]): 
        dif = primary_keys - key 
        dif_idx = (dif == 0)
        dif_idx = dif_idx[:, 0] * dif_idx[:, 1]
        # find the key
        primary_key_nums.append(np.argwhere(dif_idx == True).flatten()[0])

    data['PRIMARY_KEY'] = primary_key_nums
    tf = time()
    print(f'converting done! in {(tf-ts)/60:.2f} min')
    print(f'the number of primary keys is {len(primary_keys)}') # 5363.

    print(f'saving the split data into the path {args.split_data_path}')
    # split the data into the path: split_data_path (default: ./skt_split)
    # remove 'ENB_ID', 'CELL_NO'
    # set 'Time_Stamp' as an index
    primary_keys_num = np.arange(len(primary_keys)) # [0, 1, 2, ... 5362]
    for pkn in tqdm(primary_keys_num, total= len(primary_keys_num)): 
        idx = data['PRIMARY_KEY'] == pkn 
        sample = data.loc[idx, :]
        sample = sample.set_index('Time_Stamp')
        sample = sample.drop(['ENB_ID', 'CELL_NO'], axis= 1)
        file = os.path.join(args.split_data_path, f'kpi_gangnam_pk{pkn}.csv')
        sample.to_csv(file, index= True)

    
    print(f'saving the adjusted split data into the path {args.split_data_adj_path}')
    files = [os.path.join(args.split_data_path, p) for p in os.listdir(args.split_data_path)]
    samples = [pd.read_csv(f, parse_dates= ['Time_Stamp'], infer_datetime_format= True) for f in files]

    ts = [sample['Time_Stamp'].values for sample in samples]
    time_stamps_union = reduce(np.union1d, ts)
    time_stamps_union = time_stamps_union.astype('datetime64')
    time_stamps_union = np.sort(time_stamps_union)
    time_stamps_union = pd.Series(time_stamps_union, name= 'Time_Stamp')

    for i, sample in tqdm(enumerate(samples), total= len(samples)): 
        sample = pd.merge_ordered(time_stamps_union, sample, on= 'Time_Stamp', how= 'left')
        sample = sample.drop('CALL_RELEASE_CNT', axis= 1)
        if len(sample) > len(time_stamps_union):
            sample = merge_duplicate(sample)
        elif len(sample) < len(time_stamps_union):
            print(f'--(!)warning: len(sample) < time-stamp at index {files[i]}')
        f = os.path.split(files[i])[-1]
        file = os.path.join(args.split_data_adj_path, f)
        sample.to_csv(file, index= False)

    print('Preprocessing done!')
    return 

if __name__ == '__main__':
    main(args)

# Depreciated codes...

    # convert the original data into the time series-data
    # print(f'converting the split data into a set of windows')
    # file_list = os.listdir('../skt_split')
    # file_list = [os.path.join(args.split_data_path, file_name) for file_name in file_list]

    # num = 0
    # ws = {}
    # for f in tqdm(file_list, desc= 'outer', position= 0, total= len(file_list)): 
    #     df = pd.read_csv(f)
    #     cols = list(df.columns.drop('RRC_FAIL_RATE'))
    #     x, y = df[cols], df['RRC_FAIL_RATE']
    #     n = x.shape[0]
    #     for i in tqdm(range(n-args.time_lag), desc='inner', position= 1, leave= False, total= n-args.time_lag): 
    #         xt = x.iloc[i:i+args.time_lag+1].values 
    #         yt = y.iloc[i:i+args.time_lag+1].values 
    #         mt = torch.FloatTensor(np.isnan(xt) * 1.) # mask 
    #         w = {'xt': torch.FloatTensor(xt), 'mt': mt, 'yt': torch.FloatTensor(yt)} # window
    #         ws['window' + str(num)] = w
    #         num += 1
            
    # # save the windows
    # with open(os.path.join(args.time_series_path, f'window{num}.json'), 'w') as json_file: 
    #     json.dump(ws, json_file)
    # print(f'converting done!')