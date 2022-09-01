import os 
import sys 
import argparse
import json 

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import * 
from utils.torchUtils import *
from layers.models import *
from utils.dataloader import * 

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--data_type', type= str, default= 'skt', 
                    help= 'one of: skt')
parser.add_argument('--data_path', type= str, default= './data/skt')
parser.add_argument('--pred_steps', type= int, default= 3,
                help= 'the number of steps to predict')
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
parser.add_argument('--fine_tunning_every', type= int, default= 12, 
                help= 'fine tune a model every \'fine_tunning_every\'')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--kl_loss_penalty', type= float, default= 0.01, help= 'kl-loss penalty (default= 0.01)')
parser.add_argument('--patience', type=int, default=5, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0.01, help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--verbose', action= 'store_true', 
                help= 'print logs about early-stopping')

# model options
parser.add_argument('--model_path', type= str, default= './data/skt/model',
                    help= 'a path to (save) the model')
parser.add_argument('--model_configs', type= str, default= './data/skt/model/commandline_args.txt'
                    ,help= 'model configurations (a text file to be dumped as a json)')
# To test
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint(online).pth.tar'
                    ,help= 'model name to save', required= False)

args = parser.parse_args() 
print(args) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

print(f'saving the commandline arguments in the path: {args.model_path}...')
args_file = os.path.join(args.model_path, 'commandline_args(online).txt')
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
        = TimeSeriesDataset(*data['train'], lag= args.lag, pred_steps= args.pred_steps),\
          TimeSeriesDataset(*data['valid'], lag= args.lag, pred_steps= args.pred_steps),\
          TimeSeriesDataset(*data['test'], lag= args.lag, pred_steps= args.pred_steps)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(test_data, batch_size = 2*args.fine_tunning_every, shuffle = False)

    print("Loading data done!")

    with open(args.model_configs, 'r') as f:
        backbone_model_option = json.load(f) 
    
    args.beta = backbone_model_option['beta']

    # backbone
    backbone = HeteroNRI(
        num_heteros= args.num_heteros,
        num_ts= args.num_ts,  
        time_lags= backbone_model_option['lag'], 
        num_blocks= backbone_model_option['num_blocks'], 
        k= backbone_model_option['k'], 
        device= device,
        tau= backbone_model_option['tau'],           
    ).to(device)
    
    model_file = os.path.join(backbone_model_option['model_path'], backbone_model_option['model_file'])
    ckpt = torch.load(model_file)
    backbone.load_state_dict(ckpt['state_dict'])

    # freeze model parameters 
    # graph learning layers
    for param in backbone.glem.parameters():
        param.requires_grad = False
    # projection layer
    for param in backbone.projection.parameters():
        param.requires_grad = False
    # hetero blocks
    for i in range(backbone_model_option['num_blocks']): 
        for param in getattr(backbone, f'hetero_block{i}').parameters():
            param.requires_grad = False
            
    model = HeteroNRIMulti(
        backbone= backbone,
        pred_steps= args.pred_steps,
        device= device
    ).to(device)
    print('The model is on GPU') if next(model.parameters()).is_cuda else print('The model is on CPU')
    
    # setting training args...
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)    
    # optimizer.load_state_dict(ckpt['optimizer'])
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= args.verbose,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    ) 

    # train the multi-step heads using the training data 
    train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    
    # test the multi-step HeteroNRI (model) using test data 
    # fine tune the model every 'args.fine_tunning_every'
    # test the fine-tuned model using the next batch of the test dataset.
    print('start online-learning')
    # record time elapsed of the fine-tunning 
    # the time should be less than 5 minutes...

    te_mse = []; te_r2 = []; te_mae= []
    weights = []

    criterion_mask = nn.BCELoss()
    # test_loader_iter = iter(test_loader)
    
    x_train = {
        'input':None,
        'mask':None,
        'label':None,
        'label_mask':None
    }
    
    x_test = {
        'input':None,
        'mask':None,
        'label':None,
        'label_mask':None
    }

    for batch_idx, x in enumerate(test_loader): 

        x['input'], x['mask'], x['label'], x['label_mask'] \
        = x['input'].to(device), x['mask'].to(device), x['label'].to(device), x['label_mask'].to(device)

        # split the data 
        bs = x['input'].shape[0]
        split = bs//2
        for k, v in x.items(): 
            x_train[k] = x[k][:split,...]
            x_test[k]  = x[k][split:,...]

        model.train()
        # feed forward
        with torch.set_grad_enabled(True):
            out = model(x_train, args.beta)
            mse_loss = criterion(out['outs_label'], x_train['label'])
            if out['outs_mask'] is not None: 
                bce_loss = criterion_mask(out['outs_mask'], x_train['label_mask'])
                loss = mse_loss + bce_loss
            else: 
                loss = mse_loss
            if out['kl_loss'] is not None: 
                loss += args.kl_loss_penalty * out['kl_loss']
            # if out['regularization_loss'] is not None: 
            #     loss += args.reg_loss_penalty * out['regularization_loss']

        # backward 
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test
        model.eval() 
        with torch.no_grad():
            out = model(x_test, args.beta)
            preds = out['preds'].detach().cpu().numpy().flatten()
            label = x_test['label'].detach().cpu().numpy().flatten()

            te_mse.append(mean_squared_error(label, preds))
            te_mae.append(mean_absolute_error(label, preds))
            te_r2.append(r2_score(label, preds))
            weights.append(len(x_test))

    te_mse = np.array(te_mse)
    te_mae = np.array(te_mae)
    te_r2 = np.array(te_r2)
        
    te_mse_mean = np.average(te_mse, weights= weights)
    te_r2_mean  = np.average(te_r2, weights= weights)
    te_mae_mean  = np.average(te_mae, weights=weights)

    te_mse_std = np.average((te_mse-te_mse_mean)**2, weights= weights)
    te_r2_std = np.average((te_r2-te_r2_mean)**2, weights= weights)
    te_mae_std = np.average((te_mae-te_mae_mean)**2, weights= weights)

    perf = {
        'r2': [te_r2_mean],
        'mae': [te_mae_mean],
        'mse': [te_mse_mean],
        'r2_std': [te_r2_std],
        'mae_std': [te_mae_std],
        'mse_std': [te_mse_std]
    }

    return perf

if __name__ == '__main__': 
    perf = main(args)
    print("Test done!")
    for k, v in perf.items(): 
        print(f'{k}: {v[0]:.4f}')
    csv_file = os.path.join(args.model_path, 'perf.csv')
    pd.DataFrame(perf).to_csv(csv_file, index= False)
