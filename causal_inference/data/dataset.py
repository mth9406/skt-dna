import torch 
from torch.utils.data import Dataset

class MultiTaskTimeSeriesDataset(Dataset):
    r"""
    Multi-variate time-series dataset
    X: explainable variables 
    Y: response variables 

    # Arguments
    ____________
    X: input multi-variate time-series data (FloatTensor)
    Y: multi-variate response variable (FloatTensor) 
    """
    def __init__(self, X, Y,  lag= 1, pred_steps= 1):
        super().__init__()
        # skt data as an example,
        # X: (5363, 2293, 10)
        # M: (5363, 2293, 10)
        self.X = X
        self.Y = Y
        self.lag = lag
        self.pred_steps = pred_steps 

    def __getitem__(self, index):
        return {
            "exp_input": self.X[:, index:index+self.lag, :], # bs, c, t, n
            "exp_label": self.X[:, index+self.lag:index+self.lag+self.pred_steps, :], # bs, c, pred_steps, n 
            "res_input": self.Y[:, index:index+self.lag, :], # bs, c, t, n
            "res_label": self.Y[:, index+self.lag:index+self.lag+self.pred_steps, :] # bs, c, pred_steps, n
        }

    def __len__(self): 
        return self.X.shape[1]-self.lag-self.pred_steps+1
