import torch

import numpy as np
import pandas as pd 
from tqdm import tqdm
import pickle
from utils.utils_functions import *

import os

def load_skt(args): 
    r"""
    A function to load skt-data.
    args will have the following items after running this function.
    * decoder: a dictionary which maps idx: enb 
    * columns: a list of columns of the 'skt' data 
    * target_columns: a list of target columns of the 'skt' data
    * num_heteros: the number of eNB
    * num_ts: the number of time-series (= the dimension of 'measurement')
    * time_stamps: time-stamps of test data
    
    # Parameters
    ____________
    args contains the followings...
    * data_path: a path to skt-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    _________
    dictionary containig:
    X_train, X_valid, M_train, 
    M_train, M_vald, M_test    
    (torch.FloatTensor for both X and M)
    """
    assert args.tr + args.val < 1, "No remaining portion for the test... please let args.tr + args.val < 1"

    # data_path 
    all_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path)]
    files = list(filter(lambda f: f.endswith('.csv'), all_files))
    # decoder: idx: enb
    args.decoder = {
        idx: os.path.split(f)[-1][:-4] for idx, f in enumerate(files)
    }
    # (1) load data and generate mask...
    # {1, 0} 1 for missing, 0 for not missing
    X = []
    Y = []
    # M = []

    # explanatory data columns
    args.exp_columns = [
        'RRC_CNT', 'DL_PRB', 'RSRP','RSRQ','UPLINK_SINR','UE_TX_POWER','TA'        
    ]
    args.num_src = len(args.exp_columns)

    # target columns
    args.target_columns = [
        'CQI', 'CALL_RELEASE_ANOMALY_CNT', 'RRC_FAIL_RATE'
    ]
    args.num_dst = len(args.target_columns)

    # total columns
    args.columns = args.exp_columns+args.target_columns

    # load
    try:            
        with open(args.cache_file, 'rb') as f:
            cache = pickle.load(f)
    except: 
        cache = None
    args.cache= cache

    for i, f in tqdm(enumerate(files), total= len(files)): 
        x = pd.read_csv(f)
        x = x.iloc[:, 1:]
        if i == 0: 
            args.time_stamps = x.iloc[:, 0].values # time_stamps
        # x = min_max_scaler(x, cache, columns= args.columns) if cache is not None else x 
        x = min_max_scaler_ver2(x, cache, columns= args.columns) if cache is not None else x 
        # Time_Stamp,
        # RRC_CNT, RRC_FAIL_RATE, CALL_RELEASE_ANOMALY_CNT,
        # DL_PRB, CQI, RSRP, RSRQ, 
        # UPLINK_SINR, UE_TX_POWER, TA
        # m = ~x.isna() * 1.
        X.append(x[args.exp_columns].values) 
        Y.append(x[args.target_columns].values)
        # M.append(m.values)

    X = np.stack(X)
    Y = np.stack(Y)
    # M = np.stack(M)

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    # (2) imputation
    # Imputation strategy --> fill_zero 
    # because NaN means no mobile connection
    # --> should improve the imputation strategy....
    X = torch.nan_to_num(X, nan= 0.0)
    Y = torch.nan_to_num(Y, nan= 0.0)
    # M = torch.FloatTensor(M)

    num_heteros, num_obs, num_ts = X.shape
    num_heteros_Y, num_obs_Y, num_ts_Y = Y.shape

    args.num_heteros = num_heteros
    args.num_ts = num_ts 

    print(f'the shape of X       : ({num_heteros}, {num_obs}, {num_ts})')
    print(f'the shape of Y       : ({num_heteros_Y}, {num_obs_Y}, {num_ts_Y})')
    # print(f'the shape in the dataset: : ({num_heteros}, {num_ts}, {num_obs})')

    # (4) train-validation-test split
    start_idx_val = int(X.shape[1]*args.tr)
    start_idx_te = start_idx_val + int(X.shape[1]*args.val)

    X_train, X_valid, X_test\
        = X[:, :start_idx_val, :], X[:, start_idx_val:start_idx_te, :], X[:, start_idx_te:, :]

    Y_train, Y_valid, Y_test\
        = Y[:, :start_idx_val, :], Y[:, start_idx_val:start_idx_te, :], Y[:, start_idx_te:, :]

    # M_train, M_valid, M_test\
    #     = M[:, :start_idx_val, :], M[:, start_idx_val:start_idx_te, :], M[:, start_idx_te:, :]

    args.time_stamps_train = args.time_stamps[:start_idx_val]
    args.time_stamps_valid = args.time_stamps[start_idx_val:start_idx_te]
    args.time_stamps_test = args.time_stamps[start_idx_te:]

    return {
        "train": [X_train, Y_train],
        "valid": [X_valid, Y_valid],
        "test": [X_test, Y_test]
    }