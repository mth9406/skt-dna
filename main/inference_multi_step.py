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
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint(online).pth.tar'
                    ,help= 'model name to save', required= False)

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
            DataLoader(test_data, batch_size = args.fine_tunning_every, shuffle = False)

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
            preds = out['preds'].detach().cpu().numpy().flatten()
            label = x['label'].detach().cpu().numpy().flatten()

            te_mse.append(mean_squared_error(label, preds))
            te_mae.append(mean_absolute_error(label, preds))
            te_r2.append(r2_score(label, preds))
            weights.append(len(out['preds']))

            # record labels and predictions 
            predictions.append(out['preds'].detach().cpu()) # bs, c, t, n
            labels.append(x['label'].detach().cpu()) # bs, c, t, n
            if out['adj_mat'] is not None: 
                graphs.append(out['adj_mat'].detach().cpu()) # bs, c, n, n or bs, n, n

        model.train()
        ts = time()
        # feed forward
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

    te_mse = np.array(te_mse)
    te_mae = np.array(te_mae)
    te_r2 = np.array(te_r2)
    time_ellapsed = np.array(time_ellapsed)
        
    te_mse_mean = np.average(te_mse, weights= weights)
    te_r2_mean  = np.average(te_r2, weights= weights)
    te_mae_mean  = np.average(te_mae, weights=weights)
    time_ellapsed_mean  = np.average(time_ellapsed, weights=weights)

    te_mse_std = np.average((te_mse-te_mse_mean)**2, weights= weights)
    te_r2_std = np.average((te_r2-te_r2_mean)**2, weights= weights)
    te_mae_std = np.average((te_mae-te_mae_mean)**2, weights= weights)
    time_ellapsed_std = np.average((time_ellapsed-time_ellapsed_mean)**2, weights=weights)

    perf = {
        'r2': [te_r2_mean],
        'mae': [te_mae_mean],
        'mse': [te_mse_mean],
        'r2_std': [te_r2_std],
        'mae_std': [te_mae_std],
        'mse_std': [te_mse_std],
        'mean_fine_tunning_time': [time_ellapsed_mean],
        'std_fine_tunning_time': [time_ellapsed_std]
    }

    if args.save_results: 
        
        print('saving the predictions...')
        # make a path to save a graphs 
        graph_path = os.path.join(args.model_path, 'test/graphs')
        if not os.path.exists(graph_path):
            print("Making a path to save graphs...")
            print(f"{graph_path}")
            os.makedirs(graph_path, exist_ok= True)
        else:
            print("The path to save graphs already exists, skip making the path...")

        predictions = torch.concat(predictions, dim=0) # num_obs, num_cells, preds_steps, num_time_series
        labels = torch.concat(labels, dim=0) # num_obs, num_cells, preds_steps, num_time_series
        graphs = torch.concat(graphs, dim=0) # num_obs, num_preds_steps, num_cells, num_time_series, num_time_series
        graphs = torch.permute(graphs, (1, 2, 0, 3, 4)) # num_preds_steps, num_cells, num_obs, num_time_series, num_time_series
        graphs = graphs.numpy()         

        # graph-options
        options = {
                    'node_color': 'skyblue',
                    'node_size': 3000,
                    'width': 0.5 ,
                    'arrowstyle': '-|>',
                    'arrowsize': 20,
                    'alpha' : 1,
                    'font_size' : 15
                }

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
            # saving graphs 
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
                
            for i in tqdm(range(num_cells), total= num_cells):
                graph_path = os.path.join(args.model_path, f'test/graphs_{t}_step/{enb_id}')
                os.makedirs(graph_path, exist_ok= True)
                plt.figure(figsize =(15,15))
                for j in range(args.graph_time_range):
                    # num_obs, num_time_series, num_time_series
                    graph_file = os.path.join(graph_path, f'{enb_id}_graph_{j}.png') 
                    adj_mat = np.transpose(graphs[t, i, j, ...]) # num_time_series, num_time_series 
                    adj_mat = pd.DataFrame(adj_mat, columns = args.columns, index= args.columns)
                    # save the adj-matrix in csv format 
                    adj_mat.to_csv(os.path.join(graph_path, f'{enb_id}_graph_{j}.csv'))
                    G = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)
                    G = nx.DiGraph(G)
                    pos = nx.circular_layout(G)
                    nx.draw_networkx(G, pos=pos, **options)
                    plt.savefig(graph_file, format="PNG")
                plt.close('all')

    return perf

if __name__ == '__main__': 
    perf = main(args)
    print("Test done!")
    for k, v in perf.items(): 
        print(f'{k}: {v[0]:.4f}')
    csv_file = os.path.join(args.model_path, 'perf.csv')
    pd.DataFrame(perf).to_csv(csv_file, index= False)
