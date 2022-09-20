import torch
import torch.nn.functional as F
import numpy as np

import os

class Trainer: 

    def __init__(self):
        super().__init__()

    def __call__(self, args, model, 
                train_loader, valid_loader, 
                early_stopping, 
                optimizer, 
                scheduler=None, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        num_batches = len(train_loader)
        # # to store losses per epoch
        # tr_loss, valid_loss = 0, 0 # todo

        for epoch in range(args.epoch):
            # a training loop
            for batch_idx, batch in enumerate(train_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                model.train()
                # feed forward
                with torch.set_grad_enabled(True):
                    tr_loss = model.train_step(batch, args.exp_loss_penalty, args.kl_loss_penalty)
                    model.zero_grad()
                    optimizer.zero_grad()
                    tr_loss['total_loss'].backward()
                    optimizer.step() 

            # a validation loop 
            for batch_idx, batch in enumerate(valid_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                model.eval()
                valid_loss = model.val_step(batch, args.exp_loss_penalty, args.kl_loss_penalty)

            training_loss = tr_loss['total_loss'].detach().cpu().item()
            validation_loss = valid_loss['total_loss'].detach().cpu().item()
            print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {training_loss:.6f}, validation loss= {validation_loss:.6f}')
            early_stopping(validation_loss, model, epoch, optimizer)

            if scheduler is not None: 
                scheduler.step()

            if early_stopping.early_stop:
                break     
    
    def test(self, args, model, test_loader, device):
        # initiate performance.
        perfs = {}        
        weights = []
        
        preds_exp = []
        preds_res = []
        labels_exp = []
        labels_res = []
        graphs = []

        for batch_idx, batch in enumerate(test_loader): 
            
            for key in batch.keys():
                batch[key] = batch[key].to(device)
           
            model.eval() 
            loss, out = model.test_step(batch)
            num_batch = len(batch['exp_input'])
            weights.append(num_batch)
            for k, v in loss.items(): 
                if perfs.get(k) is None: 
                    perfs[k] = []
                perfs.get(k).append(v)

            # record labels and predictions 
            preds_exp.append(out['exp_label'].detach().cpu()) # bs, c, t, n
            preds_res.append(out['res_label'].detach().cpu()) # bs, c, t, n
            labels_exp.append(batch['exp_label'].detach().cpu()) # bs, c, t, n
            labels_res.append(batch['res_label'].detach().cpu())
            if out['relation'] is not None: 
                graphs.append(out['relation'].detach().cpu()) # bs, c, num_src, num_dst

        # save performance
        for k, v in perfs.items(): 
            mean = np.average(perfs.get(k), weights= weights)
            std = np.math.sqrt(np.average((np.array(perfs.get(k))-mean)**2, weights= weights))
            perfs[k] = [mean, std]
        
        # save labels and predictions
        preds_exp = torch.concat(preds_exp, dim= 0).numpy() # num_obs, c, 1, num_src
        preds_res = torch.concat(preds_res, dim= 0).numpy() # num_obs, c, 1, num_dst
        labels_exp = torch.concat(labels_exp, dim= 0).numpy() # num_obs, c, 1, num_src
        labels_res = torch.concat(labels_res, dim= 0).numpy() # num_obs, c, 1, num_dst
        graphs = torch.concat(graphs, dim= 0).numpy() # num_obs, c, num_src, num_dst

        out = {
            'preds_exp': preds_exp,
            'preds_res': preds_res,
            'labels_exp': labels_exp,
            'labels_res': labels_res,
            'graphs': graphs
        }
            
        return perfs, out
                
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