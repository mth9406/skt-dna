import torch
from torch import nn 
import numpy as np
import pandas as pd

import os
import csv
from tqdm import tqdm 
import networkx as nx

# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from utils import *

class EarlyStopping:
    r"""
    Applies early stopping condition... 
    """
    def __init__(self, 
                patience: int= 10,
                verbose: bool= False,
                delta: float= 0,
                path= './',
                model_name= 'latest_checkpoint.pth.tar'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_name = model_name
        self.path = os.path.join(path, model_name)
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        ckpt_dict = {
            'epoch':epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        torch.save(ckpt_dict, self.path)
        self.val_loss_min = val_loss

def train(args, 
          model, 
          train_loader, valid_loader, 
          optimizer, criterion, early_stopping,
          device):
    logs = {
        'tr_loss':[],
        'tr_mse_loss':[],
        'tr_bce_loss':[],
        'valid_loss':[],
        'valid_mse_loss':[],
        'valid_bce_loss':[]
    }

    num_batches = len(train_loader)
    print('Start training...')
    criterion_mask = nn.BCELoss()
    for epoch in range(args.epoch):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        tr_mse_loss, tr_bce_loss = 0, 0 
        valid_mse_loss, valid_bce_loss = 0, 0

        # a training loop
        for batch_idx, x in enumerate(train_loader):
            x['input'], x['mask'], x['label'], x['label_mask'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device), x['label_mask'].to(device)

            model.train()
            # feed forward
            with torch.set_grad_enabled(True):
                out = model(x, args.beta)
                mse_loss = criterion(out['outs_label'], x['label'])
                bce_loss = criterion_mask(out['outs_mask'], x['label_mask'])
                loss = mse_loss + bce_loss
                if out['kl_loss'] is not None: 
                    loss += args.kl_loss_penalty * out['kl_loss']
                # if out['regularization_loss'] is not None: 
                #     loss += args.reg_loss_penalty * out['regularization_loss']
            
            # backward 
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.gradient_max_norm)
            optimizer.step()

            # store the d_tr_loss
            tr_loss += loss.detach().cpu().item()
            tr_mse_loss += mse_loss.detach().cpu().item() 
            tr_bce_loss += bce_loss.detach().cpu().item() 

            if (batch_idx+1) % args.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                    loss = {loss.detach().cpu().item()}')

        # a validation loop 
        for batch_idx, x in enumerate(valid_loader):
            x['input'], x['mask'], x['label'], x['label_mask'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device), x['label_mask'].to(device)
            
            model.eval()
            loss = 0
            with torch.no_grad():
                out = model(x, args.beta)
                mse_loss = criterion(out['outs_label'], x['label'])
                bce_loss = criterion_mask(out['outs_mask'], x['label_mask'])
                loss = mse_loss + bce_loss
                if out['kl_loss'] is not None: 
                    loss += args.kl_loss_penalty * out['kl_loss']
            valid_loss += loss.detach().cpu().item()
            valid_mse_loss += mse_loss.detach().cpu().item() 
            valid_bce_loss += bce_loss.detach().cpu().item()         
        # save current loss values
        tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
        tr_mse_loss, tr_bce_loss = tr_mse_loss/len(train_loader), tr_bce_loss/len(train_loader)
        valid_mse_loss, valid_bce_loss = valid_mse_loss/len(valid_loader), valid_bce_loss/len(valid_loader)
        
        logs['tr_loss'].append(tr_loss)
        logs['tr_bce_loss'].append(tr_bce_loss)
        logs['tr_mse_loss'].append(tr_mse_loss)
        logs['valid_loss'].append(valid_loss)
        logs['valid_bce_loss'].append(valid_bce_loss)
        logs['valid_mse_loss'].append(valid_mse_loss)
        
        print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, training mse loss= {tr_mse_loss:.6f}, training bce loss= {tr_bce_loss:.6f}')
        empty = ' '*len(f'Epoch [{epoch+1}/{args.epoch}]')
        print(f'{empty}: validation loss= {valid_loss:.6f}, validation mse loss= {valid_mse_loss:.6f}, validation bce loss= {valid_bce_loss:.6f}')
        early_stopping(valid_loss, model, epoch, optimizer)

        if early_stopping.early_stop:
            break     

    print("Training done! Saving logs...")
    log_path= os.path.join(args.model_path, 'training_logs')
    os.makedirs(log_path, exist_ok= True)
    log_file= os.path.join(log_path, 'training_logs.csv')
    with open(log_file, 'w', newline= '') as f:
        wr = csv.writer(f)
        n = len(logs['tr_loss'])
        rows = np.array(list(logs.values())).T
        wr.writerow(list(logs.keys()))
        for i in range(1, n):
            wr.writerow(rows[i, :])

def test_regr(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    te_tot_loss = 0
    te_mse_loss = 0
    te_bce_loss = 0 
    te_preds_loss = 0

    te_r2 = 0
    te_mae = 0
    te_mse = 0
    
    preds = [] # to store predictions
    labels = []
    graphs = [] 

    criterion_mask = nn.BCELoss()
    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label'], x['label_mask'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device), x['label_mask'].to(device)
        
        model.eval()
        loss = 0
        with torch.no_grad():
            out = model(x, args.beta)
            mse_loss = criterion(out['outs_label'], x['label'])
            bce_loss = criterion_mask(out['outs_mask'], x['label_mask'])
            loss = mse_loss + bce_loss 
            if out['kl_loss'] is not None: 
                loss += args.kl_loss_penalty * out['kl_loss']
            preds_loss = criterion(out['preds'], x['label'])
            # store predictions
            preds.append(out['preds'].detach().cpu()) # bs, c, 1, n
            labels.append(x['label'].detach().cpu()) # bs, c, 1, n
            if out['adj_mat'] is not None: 
                graphs.append(out['adj_mat'].detach().cpu()) # bs, c, n, n

        te_tot_loss += loss.detach().cpu().numpy() 
        te_mse_loss += mse_loss.detach().cpu().numpy() 
        te_bce_loss += bce_loss.detach().cpu().numpy()
        te_preds_loss += preds_loss.detach().cpu().numpy()

        te_r2 += r2_score(out['preds'].detach().cpu().numpy().flatten(), x['label'].detach().cpu().numpy().flatten())
        te_mae += mean_absolute_error(out['preds'].detach().cpu().numpy().flatten(), x['label'].detach().cpu().numpy().flatten()) 
        te_mse += mean_squared_error(out['preds'].detach().cpu().numpy().flatten(), x['label'].detach().cpu().numpy().flatten()) 

    # te_loss_imp = te_loss_imp/len(test_loader)
    te_tot_loss = te_tot_loss/len(test_loader)
    te_mse_loss = te_mse_loss/len(test_loader) 
    te_bce_loss = te_bce_loss/len(test_loader) 
    te_preds_loss = te_preds_loss/len(test_loader)
    te_r2 = te_r2/len(test_loader)
    te_mae = te_mae/len(test_loader)
    te_mse = te_mse/len(test_loader)
    print("Test done!")
    # print(f"imputation loss: {te_loss_imp:.2f}")
    print(f"total loss: {te_tot_loss:.2f}")
    print(f"mse loss: {te_mse_loss:.2f}")
    print(f"bce loss: {te_bce_loss:.2f}")
    print(f"prediction loss: {te_preds_loss:.2f}")
    print(f"r2: {te_r2:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"mse: {te_mse:.2f}")
    print()    

    # concatenate predictions
    print('saving the predictions...')
    preds = torch.concat(preds, dim=0) # num_obs, num_cells, num_time_series, 1 
    preds = torch.permute(torch.squeeze(preds), (1, 0, 2)) # num_cells, num_obs, num_time_series 
    preds = preds.numpy()
    if args.cache is not None: 
        # preds = inv_min_max_scaler(preds, args.cache, args.columns)
        preds = inv_min_max_scaler_ver2(preds, args.cache, args.columns)

    labels = torch.concat(labels, dim=0) # num_obs, num_cells, num_time_series, 1
    labels = torch.permute(torch.squeeze(labels), (1, 0, 2)) # num_cells, num_obs, num_time_series
    labels = labels.numpy()
    if args.cache is not None: 
        # labels = inv_min_max_scaler(labels, args.cache, args.columns)
        labels = inv_min_max_scaler_ver2(labels, args.cache, args.columns)
    
    if len(graphs) > 0: 
        graphs = torch.concat(graphs, dim=0) # num_obs, num_cells, num_time_series, num_time_series
        graphs = torch.permute(graphs, (1, 0, 2, 3)) # num_cells, num_obs, num_time_series, num_time_series
        graphs = graphs.numpy() 
    
    num_cells = labels.shape[0]

    # make a path to save a figures 
    fig_path = os.path.join(args.model_path, 'test/figures')
    if not os.path.exists(fig_path):
        print("Making a path to save figures...")
        print(f"{fig_path}")
        os.makedirs(fig_path, exist_ok= True)
    else:
        print("The path to save figures already exists, skip making the path...")
    
    # make a path to save a graphs 
    graph_path = os.path.join(args.model_path, 'test/graphs')
    if not os.path.exists(graph_path):
        print("Making a path to save graphs...")
        print(f"{graph_path}")
        os.makedirs(graph_path, exist_ok= True)
    else:
        print("The path to save graphs already exists, skip making the path...")

    options = {
        'node_color': 'skyblue',
        'node_size': 3000,
        'width': 0.5 ,
        'arrowstyle': '-|>',
        'arrowsize': 20,
        'alpha' : 1,
        'font_size' : 15
    }
    idx = torch.LongTensor(np.arange(len(args.columns))).to(device)
    for i in tqdm(range(num_cells), total= num_cells):
        enb_id = args.decoder.get(i)
        write_csv(args, 'test/predictions', f'predictions_{enb_id}.csv', preds[i, ...], args.columns)
        write_csv(args, 'test/labels', f'labels_{enb_id}.csv', labels[i, ...], args.columns)   
        
        fig, axes = plt.subplots(len(args.columns), 1, figsize= (10,3*len(args.columns)))

        for j in range(len(args.columns)):
            col_name = args.columns[j]
            fig.axes[j].set_title(f'time-seris plot: {col_name}')
            fig.axes[j].plot(preds[i,:,j], label= 'prediction')
            fig.axes[j].plot(labels[i,:,j], label= 'label')
            fig.axes[j].legend()

        fig.suptitle(f"Prediction and True label plot of {enb_id}", fontsize=20, position= (0.5, 1.0+0.05))
        fig.tight_layout()
        fig_file = os.path.join(fig_path, f'figure_{enb_id}.png')
        fig.savefig(fig_file)

        if args.model_type == 'proto': 
            adj_mat = model.gen_adj[i](idx).data.cpu().numpy() 
            adj_mat = pd.DataFrame(adj_mat, columns = args.columns, index= args.columns)
            plt.figure(figsize =(15,15))
            # plt.xkcd()
            G = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)
            G = nx.DiGraph(G)
            pos = nx.circular_layout(G)
            nx.draw_networkx(G, pos=pos, **options)
            plt.savefig(os.path.join(graph_path, f"graph_{enb_id}.png"), format="PNG")
        else:
            graph_path = os.path.join(args.model_path, f'test/graphs/{enb_id}')
            os.makedirs(graph_path, exist_ok= True)
            plt.figure(figsize =(15,15))
            for j in range(args.graph_time_range):
                # num_obs, num_time_series, num_time_series
                graph_file = os.path.join(graph_path, f'{enb_id}_graph_{j}.png') 
                adj_mat = np.transpose(graphs[i, j, ...]) # num_time_series, num_time_series 
                adj_mat = pd.DataFrame(adj_mat, columns = args.columns, index= args.columns)
                # save the adj-matrix in csv format 
                adj_mat.to_csv(os.path.join(graph_path, f'{enb_id}_graph_{j}.csv'))
                G = nx.from_pandas_adjacency(adj_mat)
                G = nx.DiGraph(G)
                pos = nx.circular_layout(G)
                nx.draw_networkx(G, pos=pos, **options)
                plt.savefig(graph_file, format="PNG")
            # plt.close('all')
        plt.close('all')

    perf = {
        'r2': te_r2,
        'mae': te_mae,
        'mse': te_mse
    }
    # perf = {
    #     'mae': te_mae,
    #     'mse': te_mse
    # }

    return perf 

