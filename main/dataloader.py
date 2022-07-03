import torch
from torch.utils.data import Dataset
from utils import *
import numpy as np
import pandas as pd 
from tqdm import tqdm

import os

class TimeSeriesDataset(Dataset):
    """
    Multi-variate time-series dataset

    # Parameters
    ____________
    X: input multi-variate time-series data (FloatTensor)
    M: mask, {0,1} to represent missing value (FloatTensor)
    D: time-interval (depreciated)
    y: independent variable (target variable - not be used in this research) 
    """
    def __init__(self, X, M, D= None, lag= 1):
        super().__init__()
        # X: (5363, 2293, 10) -> (5363, 10, 2293)
        # M: (5363, 2293, 10) -> (5363, 10, 2293)
        self.X, self.M = X.permute(0, 2, 1), M.permute(0, 2, 1)
        if D is not None: 
            self.D = D.permute(0, 2, 1)
        else: 
            self.D = D
        self.lag = lag

    def __getitem__(self, index):
        if self.D is None: 
            return {
                "input": self.X[:, :, index:index+self.lag],
                "mask": self.M[:, :, index:index+self.lag],
                "label": self.X[:, :, index+self.lag:index+self.lag+1]
            }
        else: 
            return {
                "input": self.X[:, :, index:index+self.lag],
                "mask": self.M[:, :, index:index+self.lag],
                "label": self.X[:, :, index+self.lag:index+self.lag+1],
                "time_interval":self.D[index]               
            }

    def __len__(self): 
        return self.X.shape[-1]-self.lag

def load_skt(args): 
    """
    A function to load skt-data.
    
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

    # (1) load data and generate mask...
    # {1, 0} 1 for missing, 0 for not missing
    X = []
    M = []

    for f in tqdm(files, total= len(files)): 
        x = pd.read_csv(f)
        x = x.iloc[:, 1:-1] 
        # RRC_CNT	RRC_FAIL_RATE	
        # CALL_RELEASE_CNT	CALL_RELEASE_ANOMALY_CNT	
        # DL_PRB	CQI	RSRP	RSRQ	
        # UPLINK_SINR	UE_TX_POWER	TA
        m = ~x.isna() * 1.
        X.append(x.values) 
        M.append(m.values)

    X = np.stack(X)
    M = np.stack(M)

    X = torch.FloatTensor(X)
    # (2) imputation
    # Imputation strategy --> fill_zero 
    # because NaN means no mobile connection
    # --> should improve the imputation strategy....
    X = torch.nan_to_num(X, nan= 0.0)
    M = torch.FloatTensor(M)

    num_heteros, num_obs, num_ts = X.shape
    args.num_heteros = num_heteros
    args.num_ts = num_ts 

    print(f'original shape of X       : ({num_heteros}, {num_obs}, {num_ts})')
    print(f'the shape in the dataset: : ({num_heteros}, {num_ts}, {num_obs})')

    # (4) train-validation-test split
    start_idx_val = int(X.shape[1]*args.tr)
    start_idx_te = start_idx_val + int(X.shape[1]*args.val)


    X_train, X_valid, X_test\
        = X[:, :start_idx_val, :], X[:, start_idx_val:start_idx_te, :], X[:, start_idx_te:, :]

    M_train, M_valid, M_test\
         = M[:, :start_idx_val, :], M[:, start_idx_val:start_idx_te, :], M[:, start_idx_te:, :]

    return {
        "train": [X_train, M_train],
        "valid": [X_valid, M_valid],
        "test": [X_test, M_test]
    }