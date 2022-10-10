import os 
import sys 
import json
import argparse 

import torch 
import numpy as np 
import pandas as pd 
pd.options.display.max_columns = 20
import networkx as nx 
from matplotlib import pyplot as plt 

import torch.optim as optim 
from torch.utils.data import DataLoader 

# from data.dataloaders import * 
from data.dataset import * 
from data.load_data import * 

from models.causal_inference_model import CausalInferenceModel

from utils.trainer import Trainer, EarlyStopping
from utils.utils_functions import * 

parser = argparse.ArgumentParser() 

# Data path
parser.add_argument('--data_type', type= str, default= 'skt', 
                    help= 'one of: skt')
parser.add_argument('--data_path', type= str, default= './data/skt')
parser.add_argument('--tr', type= float, default= 0.7, 
                    help= 'the ratio of training data to the original data')
parser.add_argument('--val', type= float, default= 0.2, 
                    help= 'the ratio of validation data to the original data')
parser.add_argument('--lag', type= int, default= 1, 
                    help= 'time-lag (default: 1)')
parser.add_argument('--cache_file', type= str, default= './data/cache.pickle', 
                    help= 'a cache file to min-max scale the data')
parser.add_argument('--graph_time_range', type= int, default= 12, 
                    help= 'time-range to save a graph')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--kl_loss_penalty', type= float, default= 0.01, help= 'kl-loss penalty (default= 0.01)')
parser.add_argument('--exp_loss_penalty', type= float, default= 0.01, 
                    help= 'explanatory variable (reconstruction)-loss penalty (default= 0.01)')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--verbose', action= 'store_true', 
                    help= 'print logs about early-stopping')

# model options
parser.add_argument('--num_blocks_src', type= int, default= 20, 
                    help= 'the number of temporal convolution blocks for source node (default= 20)')
parser.add_argument('--num_blocks_dst', type= int, default= 20, 
                    help= 'the number of temporal convolution blocks for destination node (default= 20)')
parser.add_argument('--num_gcn_blocks', type= int, default= 3, 
                    help= 'the number of graph convolution blocks (default= 3)')
parser.add_argument('--beta', type= float, default= 0.5, 
                    help= 'decay factor in the temporal convolution module')
parser.add_argument('--tau', type= float, default= 0.1,
                    help= 'tau (default: 0.1)')

# model-file options
parser.add_argument('--model_path', type= str, default= './data/skt/model',
                    help= 'a path to (save) the model')
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model name to save')
parser.add_argument('--pretrained_model_file', type= str
                    ,help= 'pretrained model file', required= False)

# to save predictions and graphs
parser.add_argument('--save_results', action= 'store_true', 
                    help= 'to save graphs and figures in the model_path') 

# to test
parser.add_argument('--test', action= 'store_true', 
                    help= 'test the model without training')

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
    print("Loading data...")
    if args.data_type == 'skt':
        data = load_skt(args)
    else: 
        print("Unkown data type, data type should be \"skt\"")
        sys.exit()

    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = MultiTaskTimeSeriesDataset(*data['train'], lag= args.lag),\
          MultiTaskTimeSeriesDataset(*data['valid'], lag= args.lag),\
          MultiTaskTimeSeriesDataset(*data['test'], lag= args.lag)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    print("Loading data done!")
    
    model = CausalInferenceModel(
        num_heteros= args.num_heteros,
        num_src = args.num_src,
        num_dst = args.num_dst, 
        time_lags = args.lag, 
        num_blocks_src = args.num_blocks_src, 
        num_blocks_dst = args.num_blocks_dst,
        num_gcn_blocks = args.num_gcn_blocks, 
        tau = args.tau, 
        beta = args.beta,
        device= device,         
    ).to(device)
    print('The model is on GPU') if next(model.parameters()).is_cuda else print('The model is on CPU')

    # setting training args...
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10) 
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= args.verbose,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    ) 

    trainer = Trainer()

    if args.test: 
        if not hasattr(args, 'pretrained_model_file'): 
            print('[Warning!] the model has not been trained')
        print('loading the saved model')
        ckpt = torch.load(args.pretrained_model_file)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('loading done!')
    else: 
        print('start training...')
        trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, scheduler, device)      

    print("==============================================")
    print("Testing the model...") 
    if not args.test:
        print('loading the saved model')
        model_file = os.path.join(args.model_path, args.model_name)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])    
        print('loading done!')  
    perfs, out = trainer.test(args, model, test_loader, device)
    # print(perfs)
    # for k, perf in perfs.items(): 
    #     print(f'{k} mean: {perf[0]:.4f}')  
    #     print(f'{k} std: {perf[1]:.4f}')  

    return perfs, out

if __name__ == '__main__':
    perf, out = main(args) 
    perf_df = pd.DataFrame(perf) 
    perf_df.index = ['mean', 'std']
    perf_df_file = os.path.join(args.model_path, f'test_results.csv')
    perf_df.to_csv(perf_df_file)
    print(perf_df)

    if args.save_results:
        print("==============================================")
        print("Saving the results...") 
        preds_exp = out.get('preds_exp')
        preds_res = out.get('preds_res')
        labels_exp = out.get('labels_exp')
        labels_res = out.get('labels_res')
        graphs = out.get('graphs')

        pred_steps = preds_exp.shape[2]

        # graph-options
        options = {
                    'node_color': 'skyblue',
                    'node_size': 3000,
                    'width': 0.5 ,
                    # 'arrowstyle': '-|>',
                    'arrowsize': 20,
                    'alpha' : 1,
                    'font_size' : 15
                }

        for t in range(pred_steps):
            p_res = np.transpose(preds_res[:,:,t,:], (1, 0, 2)) # num_cells, num_obs, num_src
            p_exp = np.transpose(preds_exp[:,:,t,:], (1, 0, 2)) 
            p = np.concatenate([p_exp, p_res], axis= 2)
            if args.cache is not None: 
                # preds = inv_min_max_scaler(preds, args.cache, args.columns)
                p = inv_min_max_scaler_ver2(p, args.cache, args.columns)

            l_res = np.transpose(labels_exp[:,:,t,:], (1, 0, 2)) # num_cells, num_obs, num_dst 
            l_exp = np.transpose(labels_res[:,:,t,:], (1, 0, 2)) 
            l = np.concatenate([l_exp,l_res], axis= 2)
            num_cells = l.shape[0]
            if args.cache is not None: 
                # labels = inv_min_max_scaler(labels, args.cache, args.columns)
                l = inv_min_max_scaler_ver2(l, args.cache, args.columns)
        
            # saving figures: predictions vs labels
            # saving graphs 
            for i in tqdm(range(num_cells), total= num_cells):
                enb_id = args.decoder.get(i)
                write_csv(args, f'test/predictions_{t+1}_step', f'predictions_{enb_id}.csv', p[i, ...], args.columns)
                write_csv(args, f'test/labels_{t+1}_step', f'labels_{enb_id}.csv', l[i, ...], args.columns)   
                
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
                fig_path = os.path.join(args.model_path, f'test/figures/{t+1}_step')
                if not os.path.exists(fig_path):
                    # print("Making a path to save figures...")
                    print(f"{fig_path}")
                    os.makedirs(fig_path, exist_ok= True)
                # else:
                #     print("The path to save figures already exists, skip making the path...")
                fig_file = os.path.join(fig_path, f'figure_{enb_id}.png')
                fig.savefig(fig_file)
                plt.close('all')
        
        graphs = np.transpose(graphs, (1, 0, 2, 3)) # c, num_obs, num_src, num_dst
        for i in tqdm(range(num_cells), total= num_cells):
            enb_id = args.decoder.get(i)
            graph_path = os.path.join(args.model_path, f'test/graphs/1_step/{enb_id}')
            os.makedirs(graph_path, exist_ok= True)
            plt.figure(figsize =(15,15))
            for j in range(args.graph_time_range):
                graph_file = os.path.join(graph_path, f'{enb_id}_graph_{j}.png') 
                adj_mat = np.transpose(graphs[i, j, ...]) # num_src, num_dst 
                adj_mat = pd.DataFrame(adj_mat, columns = args.exp_columns, index= args.target_columns)
                # save the adj-matrix in csv format 
                adj_mat.to_csv(os.path.join(graph_path, f'{enb_id}_graph_{j}.csv'))
                # make a graph 
                row, col = np.nonzero(adj_mat.values)
                edges = [[i, j] for i,j in  zip(adj_mat.index[row], adj_mat.columns[col])]
                G = nx.Graph()
                G.add_nodes_from(args.exp_columns, bipartite=0)
                G.add_nodes_from(args.target_columns, bipartite= 1)
                G.add_edges_from(edges)

                pos = nx.drawing.layout.bipartite_layout(G, args.exp_columns)
                nx.draw_networkx(G, pos=pos, **options)
                plt.savefig(graph_file, format="PNG")
            plt.close('all')
