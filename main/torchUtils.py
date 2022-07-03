import torch
import numpy as np

import os
import csv

# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot as plt

from utils import *

class EarlyStopping:
    """
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
        'valid_loss':[]
    }

    num_batches = len(train_loader)
    print('Start training...')
    for epoch in range(args.epoch):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        # a training loop
        for batch_idx, x in enumerate(train_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)

            model.train()
            # feed forward
            with torch.set_grad_enabled(True):
                out = model(x, args.beta)
                loss = criterion(out['preds'], x['label'])
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

            if (batch_idx+1) % args.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                    loss = {loss.detach().cpu().item()}')

        # a validation loop 
        for batch_idx, x in enumerate(valid_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
            
            model.eval()
            loss = 0
            with torch.no_grad():
                out = model(x, args.beta)
                loss = criterion(out['preds'], x['label'])
                # if out['regularization_loss'] is not None: 
                #     loss += args.reg_loss_penalty * out['regularization_loss']
            valid_loss += loss.detach().cpu().item()
        
        # save current loss values
        tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
        logs['tr_loss'].append(tr_loss)
        logs['valid_loss'].append(valid_loss)

        print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.6f}')
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
    
    te_loss_preds = 0
    te_loss_tot = 0
    te_r2 = 0
    te_mae = 0
    te_mse = 0
    
    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label'] \
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        model.eval()
        loss = 0
        with torch.no_grad():
            out = model(x, args.beta)
            loss = criterion(out['preds'], x['label'])
            loss_reg = 0. 
            # if out['regularization_loss'] is not None: 
            #     loss_reg += args.reg_loss_penalty * out['regularization_loss']
            tot_loss = loss + loss_reg
      
        te_loss_preds += loss.detach().cpu().numpy()
        te_loss_tot += tot_loss.detach().cpu().numpy()


        # te_r2 += r2_score(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy())
        te_mae += mean_absolute_error(out['preds'].detach().cpu().numpy().flatten(), x['label'].detach().cpu().numpy().flatten()) 
        te_mse += mean_squared_error(out['preds'].detach().cpu().numpy().flatten(), x['label'].detach().cpu().numpy().flatten()) 

    # te_loss_imp = te_loss_imp/len(test_loader)
    te_loss_preds = te_loss_preds/len(test_loader)
    te_loss_tot = te_loss_tot/len(test_loader)
    # te_r2 = te_r2/len(test_loader)
    te_mae = te_mae/len(test_loader)
    te_mse = te_mse/len(test_loader)
    print("Test done!")
    # print(f"imputation loss: {te_loss_imp:.2f}")
    print(f"prediction loss: {te_loss_preds:.2f}")
    print(f"total loss: {te_loss_tot:.2f}")
    # print(f"r2: {te_r2:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"mse: {te_mse:.2f}")
    print()    

    # perf = {
    #     'r2': te_r2,
    #     'mae': te_mae,
    #     'mse': te_mse
    # }
    perf = {
        'mae': te_mae,
        'mse': te_mse
    }

    return perf 

