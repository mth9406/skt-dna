import os
import sys
import argparse
import json 

import torch
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchUtils import * 
from models import *
from dataloader import * 

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--data_type', type= str, default= 'skt', 
                    help= 'one of: skt')
parser.add_argument('--data_path', type= str, default= './data/skt')
parser.add_argument('--tr', type= float, default= 0.7, 
                help= 'the ratio of training data to the original data')
parser.add_argument('--val', type= float, default= 0.2, 
                help= 'the ratio of validation data to the original data')
parser.add_argument('--standardize', action= 'store_true', 
                help= 'standardize the inputs if it is true.')
parser.add_argument('--exclude_TA', action= 'store_true', 
                help= 'exclude TA column if it is set true.')
parser.add_argument('--lag', type= int, default= 1, 
                help= 'time-lag (default: 1)')
parser.add_argument('--cache_file', type= str, default= './data/cache.pickle', 
                help= 'a cache file to min-max scale the data')
parser.add_argument('--graph_time_range', type= int, default= 36, 
                help= 'time-range to save a graph')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--kl_loss_penalty', type= float, default= 0.01, help= 'kl-loss penalty (default= 0.01)')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--verbose', action= 'store_true', 
                    help= 'print logs about early-stopping')

# model options
parser.add_argument('--model_path', type= str, default= './data/skt/model',
                    help= 'a path to (save) the model')
parser.add_argument('--num_blocks', type= int, default= 3, 
                    help= 'the number of the HeteroBlocks (default= 3)')
parser.add_argument('--k', type= int, default= 2,
                help= 'the number of layers at every GC-Module (default= 2)')
parser.add_argument('--top_k', type= int, default= 4, 
                help= 'top_k to select as non-zero in the adjacency matrix    (default= 4)')
parser.add_argument('--embedding_dim', type= int, default= 128,
                help= 'the size of embedding dimesion in the graph-learning layer (default= 128)')
parser.add_argument('--alpha', type= float, default= 3.,
                help= 'controls saturation rate of tanh: activation function in the graph-learning layer (default= 3.0)')      
parser.add_argument('--beta', type= float , default= 0.5, 
                help= 'parameter used in the GraphConvolutionModule, must be in the interval [0,1] (default= 0.5)')
# only for the heteroNRI
parser.add_argument('--tau', type= float, default= 1., 
                help= 'smoothing parameter used in the Gumbel-Softmax, only used in the model: heteroNRI')
parser.add_argument('--hard', action= 'store_true', 
                help= 'apply hard coding the the graph (outcome of Gumbel-Softmax), only used in the model: heteroNRI')

# To test
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)
parser.add_argument('--model_type', type= str, default= 'proto', 
                    help= 'one of: \'proto\', \'heteroNRI\'... ')

parser.add_argument('--num_folds', type= int, default= 1, 
                    help = 'the number of folds')

args = parser.parse_args()
print(args)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

print(f'saving the commandline arguments in the path: {args.model_path}...')
args_file = os.path.join(args.model_path, 'commandline_args.txt')
with open(args_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def main(args):
    # read data
    # one of: gestures, physionet, mimic3
    print("Loading data...")
    if args.data_type == 'skt':
        # load gestures-data
        data = load_skt(args) if not args.exclude_TA else load_skt_without_TA(args)
    else: 
        print("Unkown data type, data type should be \"skt\"")
        sys.exit()

    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = TimeSeriesDataset(*data['train'], lag= args.lag),\
          TimeSeriesDataset(*data['valid'], lag= args.lag),\
          TimeSeriesDataset(*data['test'], lag= args.lag)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    print("Loading data done!")

    # model
    if args.model_type == 'proto':
        model = HeteroMTGNN(
            num_heteros= args.num_heteros,
            num_ts= args.num_ts,  
            time_lags= args.lag, 
            num_blocks= args.num_blocks, 
            k= args.k, 
            embedding_dim= args.embedding_dim,
            device= device,
            alpha= args.alpha,
            top_k= args.top_k
        ).to(device)
        print('The model is on GPU') if next(model.parameters()).is_cuda else print('The model is on CPU')
    elif args.model_type == 'heteroNRI':
        model = HeteroNRI(
            num_heteros= args.num_heteros,
            num_ts= args.num_ts,  
            time_lags= args.lag, 
            num_blocks= args.num_blocks, 
            k= args.k, 
            device= device,
            tau= args.tau,           
        ).to(device)
    elif args.model_type == 'heteroSpatialNRI': 
        model = HeteroSpatialNRI(
            num_heteros= args.num_heteros,
            num_ts = args.num_ts, 
            time_lags = args.lag, 
            num_blocks = args.num_blocks,
            k= args.k, 
            tau= args.tau,
            device= device
        ).to(device)
    else:
        print("The model is yet to be implemented.")
        sys.exit()
    
    # setting training args...
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= args.verbose,
        delta = args.delta,
        path= args.model_path
    ) 
       
    if args.test: 
        model_file = os.path.join(args.model_path, args.model_file)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])
    else: 
        train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)

    print("Testing the model...") 
    perf = test_regr(args, model, test_loader, criterion, device)
    return perf 

if __name__ =='__main__':
    if args.num_folds == 1:
        main(args)
    else: 
        perf = main(args)
        perfs = dict().fromkeys(perf, None)
        for k in perfs.keys():
            perfs[k] = [perf[k]]

        for i in range(1, args.num_folds): 
            perf = main(args)
            for k in perfs.keys():
                perfs[k].append(perf[k])
        
        for k, v in perfs.items():
            perfs[k] = [np.mean(perfs[k]), np.std(perfs[k])]
        
        for k, v in perfs.items(): 
            print(f"{k}: mean= {v[0]:.3f}, std= {v[1]:.3f}")