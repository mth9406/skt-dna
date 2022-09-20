import os 
import csv 
import numpy as np 

def min_max_scaler(X, cache, columns): 
    return (X-cache['min'][columns])/(cache['max'][columns]-cache['min'][columns])

def inv_min_max_scaler(X, cache, columns): 
    min = cache['min'][columns].values[np.newaxis, :]
    max = cache['max'][columns].values[np.newaxis, :]
    return (max-min)*X + min

def min_max_scaler_ver2(X, cache, columns): 
    r"""
    maps the features to the interval [-1, 1] 
    """
    return 2*(X-cache['min'][columns])/(cache['max'][columns]-cache['min'][columns]) - 1

def inv_min_max_scaler_ver2(X, cache, columns): 
    r"""
    maps the min-max scaled features to the original scale
    """
    min = cache['min'][columns].values[np.newaxis, :]
    max = cache['max'][columns].values[np.newaxis, :]
    return (X+1)/2 *(max-min) + min

def write_csv(args, path_name, file_name, data, columns= None): 
    r"""
    maps the min-max scaled features to the original scale
    """
    log_path= os.path.join(args.model_path, path_name)
    os.makedirs(log_path, exist_ok= True)
    log_file= os.path.join(log_path, file_name)
    with open(log_file, 'w', newline= '') as f:
        wr = csv.writer(f)
        n = len(data) # f: n x p numpy data
        # wr.writerow(list(logs.keys()))
        if columns is not None: 
            wr.writerow(columns)
        for i in range(n):
            wr.writerow(data[i, :])    

# graph saving fucntion 추가하기 