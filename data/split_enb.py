import argparse 
import pandas as pd 
import numpy as np 

import os
from time import time
from tqdm import tqdm 

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type= str, default= './skt', 
                help= 'a path to the data (default= \'./skt\'')
parser.add_argument('--enb_data_path', type= str, default= './enb',
                help= 'a path to save the splid data')
                
args = parser.parse_args() 

if not os.path.exist(args.enb_data_path):
    print('making a path to save enb data...')
    os.makedirs(args.enb_data_path, exist_ok= True)
else: 
    print("The path already exists, skip making the path...")

def main(args): 
    print('loading raw-data...')
    raw = pd.read_csv('./data/ChristmasWeek_KPI_Gangnam.csv')
    data = raw.drop(['Unnamed: 0','ADONG_CD', 'CELL_NO', 'CALL_RELEASE_CNT'], axis= 1)
    # change the data type of time-stamp
    data['Time_Stamp'] = data['Time_Stamp'].astype('datetime64')
    print('loading data done!')
    print(raw.head())
    
    print('='*30)
    enb_set = list(data['ENB_ID'].unique()) # enb set
    print(f'the total number of eNBs: {len(enb_set)}')
    
    # the set of time-stamps
    ts_set = np.sort(list(data['Time_Stamp'].unique()))
    ts_set = ts_set.astype('datetime64')
    ts_series = pd.Series(ts_set, name= 'Time_Stamp')
    
    ts = time()
    for enb_id in tqdm(enb_set, total= len(enb_set)): 
        idx = data['ENB_ID'] == enb_id
        enb = data.loc[idx, :]
        enb = pd.merge_ordered(ts_series, enb, on= 'Time_Stamp', how= 'left')
        enb = enb.groupby('Time_Stamp').mean()
        enb = enb.drop(['ENB_ID'], axis =1)
        assert len(enb) == len(ts_series), f'something wrong is goining on in the eNB {enb_id}'

        # save the enb file 
        f = os.path.join(args.enb_data_path, f'enb{enb_id}.csv')
        enb.to_csv(f)
    tf = time()
    print(f'preprocessing done in {tf-ts:.2f} sec')

if __name__ == '__main__':
    main(args)