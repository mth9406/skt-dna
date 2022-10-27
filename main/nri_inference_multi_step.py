import os
import sys 
import argparse
import json 

import pandas as pd
from time import time

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

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--fine_tunning_every', type= int, default= 12, 
                help= 'fine tune a model every \'fine_tunning_every\'')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--epoch_online', type= int, default=30, help= 'the number of epoches to train online for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--kl_loss_penalty', type= float, default= 0.01, help= 'kl-loss penalty (default= 0.01)')
parser.add_argument('--patience', type=int, default=5, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0.01, help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--verbose', action= 'store_true', 
                help= 'print logs about early-stopping')
parser.add_argument('--train_ar', action= 'store_true', 
                help= 'train autoregressive predictions if set true')
parser.add_argument('--train_online', action= 'store_true', 
                help= 'train online if set true')

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
parser.add_argument('--beta', type= float , default= 0.5, 
                help= 'parameter used in the GraphConvolutionModule, must be in the interval [0,1] (default= 0.5)')
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint(online).pth.tar'
                    ,help= 'model name to save', required= False)
parser.add_argument('--tau', type= float, default= 1., 
                help= 'smoothing parameter used in the Gumbel-Softmax, only used in the model: heteroNRI')
parser.add_argument('--n_hid_encoder', type = int, default= 256, 
                help= 'dimension of a hidden vector in the nri-encoder')
parser.add_argument('--msg_hid', type= int, default= 256, 
                help= 'dimension of a message vector in the nri-decoder')
parser.add_argument('--msg_out', type= int, default= 256, 
                help= 'dimension of a message vector (out) in the nri-decoder')
parser.add_argument('--n_hid_decoder', type= int, default= 256, 
                help= 'dimension of a hidden vector in the nri-decoder')

# to save predictions and graphs
parser.add_argument('--save_results', action= 'store_true', 
                help= 'to save graphs and figures in the model_path') 

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
            DataLoader(test_data, batch_size = args.fine_tunning_every, shuffle = False)

    print("Loading data done!")
            
    model = NRIMulti(
        num_heteros= args.num_heteros,
        num_time_series= args.num_ts,  
        time_lags= args.lag,  
        tau= args.tau,  
        n_hid_encoder= args.n_hid_encoder,
        msg_hid = args.msg_hid,
        msg_out = args.msg_out,
        n_hid_decoder = args.n_hid_decoder,    
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
    if args.train_ar: 
        train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    else: 
        print('skip training auto-regressives predictions...')
    # test the multi-step HeteroNRI (model) using test data 
    # fine tune the model every 'args.fine_tunning_every'
    # test the fine-tuned model using the next batch of the test dataset.
    if args.train_online:
        print('start online-learning...')
    else: 
        print('start evaluating...')
    # record time elapsed of the fine-tunning 
    # the time should be less than 5 minutes...

    te_mse = [[] for _ in range(args.pred_steps)]; te_r2 = [[] for _ in range(args.pred_steps)]; 
    te_mae= [[] for _ in range(args.pred_steps)]
    weights = []
    time_ellapsed = []

    criterion_mask = nn.BCELoss()
    # test_loader_iter = iter(test_loader)

    predictions = []
    labels = []
    graphs = []

    for batch_idx, x in enumerate(test_loader): 

        x['input'], x['mask'], x['label'], x['label_mask'] \
        = x['input'].to(device), x['mask'].to(device), x['label'].to(device), x['label_mask'].to(device)
        
        # test
        model.eval() 
        with torch.no_grad():
            out = model(x, args.beta)
            preds = out['preds'].detach().cpu().numpy()
            label = x['label'].detach().cpu().numpy()

            for t in range(args.pred_steps):
                te_mse[t].append(mean_squared_error(label[...,t,:].flatten(), preds[...,t,:].flatten()))
                te_mae[t].append(mean_absolute_error(label[...,t,:].flatten(), preds[...,t,:].flatten()))
                te_r2[t].append(r2_score(label[...,t,:].flatten(), preds[...,t,:].flatten()))
            weights.append(len(out['preds']))

            # record labels and predictions 
            predictions.append(out['preds'].detach().cpu()) # bs, c, t, n
            labels.append(x['label'].detach().cpu()) # bs, c, t, n
            if out['adj_mat'] is not None: 
                graphs.append(out['adj_mat'].detach().cpu()) # bs, c, n, n or bs, n, n
        
        if args.train_online:
            model.train()
            ts = time()
            # feed forward
            print(f'[Batch: {batch_idx+1} / {len(test_loader)}] online learning...')
            for epoch in range(args.epoch_online):
                with torch.set_grad_enabled(True):
                    out = model(x, args.beta)
                    mse_loss = criterion(out['outs_label'], x['label'])
                    if out['outs_mask'] is not None: 
                        bce_loss = criterion_mask(out['outs_mask'], x['label_mask'])
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
            tf = time()
            time_ellapsed.append(tf-ts)
            print(f'[Batch: {batch_idx+1} / {len(test_loader)}] online learning done in {tf-ts:4f} sec')

    te_mse = np.array(te_mse)
    te_mae = np.array(te_mae)
    te_r2 = np.array(te_r2)
    time_ellapsed = np.array(time_ellapsed) if args.train_online else float('nan')
   
    te_mse_mean = np.average(te_mse, weights= weights, axis= 1)
    te_r2_mean  = np.average(te_r2, weights= weights, axis= 1)
    te_mae_mean  = np.average(te_mae, weights=weights, axis= 1)
    time_ellapsed_mean = np.average(time_ellapsed, weights=weights) if args.train_online else float('nan')

    te_mse_std = np.average((te_mse-te_mse_mean[:, np.newaxis])**2, weights= weights, axis= 1)
    te_r2_std = np.average((te_r2-te_r2_mean[:, np.newaxis])**2, weights= weights, axis= 1)
    te_mae_std = np.average((te_mae-te_mae_mean[:, np.newaxis])**2, weights= weights, axis= 1)
    time_ellapsed_std = np.average((time_ellapsed-time_ellapsed_mean)**2, weights=weights) if args.train_online else float('nan')
    
    perf = {}
    for t in range(args.pred_steps):
        perf[f'r2_{t}'] = [te_r2_mean[t]]
        perf[f'mae_{t}'] = [te_mae_mean[t]]
        perf[f'mse_{t}'] = [te_mse_mean[t]]
        perf[f'r2_std_{t}'] = [te_r2_std[t]]
        perf[f'mae_std_{t}'] = [te_mae_std[t]]
        perf[f'te_mse_std_{t}'] = [te_mse_std[t]]
    perf['mean_fine_tunning_time'] = [time_ellapsed_mean]
    perf['std_fine_tunning_time'] = [time_ellapsed_std]

    print(perf)

    if args.save_results: 
        
        print('saving the predictions...')

        predictions = torch.concat(predictions, dim=0) # num_obs, num_cells, preds_steps, num_time_series
        labels = torch.concat(labels, dim=0) # num_obs, num_cells, preds_steps, num_time_series    

        for t in range(args.pred_steps):
            p = torch.permute(predictions[:,:,t, :], (1, 0, 2)) # num_cells, num_obs, num_time_series 
            p = p.numpy()
            if args.cache is not None: 
                # preds = inv_min_max_scaler(preds, args.cache, args.columns)
                p = inv_min_max_scaler_ver2(p, args.cache, args.columns)

            l = torch.permute(labels[:,:,t, :], (1, 0, 2)) # num_cells, num_obs, num_time_series
            l = l.numpy()
            num_cells = l.shape[0]
            if args.cache is not None: 
                # labels = inv_min_max_scaler(labels, args.cache, args.columns)
                l = inv_min_max_scaler_ver2(l, args.cache, args.columns)
        
            # saving figures: predictions vs labels
            for i in tqdm(range(num_cells), total= num_cells):
                enb_id = args.decoder.get(i)
                write_csv(args, f'test/predictions_{t}_step', f'predictions_{enb_id}.csv', p[i, ...], args.columns)
                write_csv(args, f'test/labels_{t}_step', f'labels_{enb_id}.csv', l[i, ...], args.columns)   
                
                fig, axes = plt.subplots(len(args.columns), 1, figsize= (10,3*len(args.columns)))

                for j in range(len(args.columns)):
                    col_name = args.columns[j]
                    fig.axes[j].set_title(f'time-seris plot: {col_name}')
                    fig.axes[j].plot(p[i,:,j], label= 'prediction')
                    fig.axes[j].plot(l[i,:,j], label= 'label')
                    fig.axes[j].legend()
                
                fig.suptitle(f"Prediction and True label plot of {enb_id}", fontsize=20, position= (0.5, 1.0+0.05))
                fig.tight_layout()
                # make a path to save a figures 
                fig_path = os.path.join(args.model_path, f'test/figures/{t}_step')
                if not os.path.exists(fig_path):
                    # print("Making a path to save figures...")
                    print(f"{fig_path}")
                    os.makedirs(fig_path, exist_ok= True)
                # else:
                #     print("The path to save figures already exists, skip making the path...")
                fig_file = os.path.join(fig_path, f'figure_{enb_id}.png')
                fig.savefig(fig_file)
                plt.close('all')

    return perf

if __name__ == '__main__': 
    perf = main(args)
    print("Test done!")
    for k, v in perf.items(): 
        print(f'{k}: {v[0]:.4f}')
    csv_file = os.path.join(args.model_path, 'perf.csv')
    pd.DataFrame(perf).to_csv(csv_file, index= False)
