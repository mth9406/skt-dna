import torch 
from torch import nn 
from torch.nn import functional as F 
import numpy as np  

from graphLearningLayers import * 

class HeteroNRIDecoderModule(nn.Module): 
    r"""
    Hetero NRI decoder   
    This module is based on the work by 
    
    \"""
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). 
    Neural relational inference for interacting systems. 
    In International Conference on Machine Learning (pp. 2688-2697). PMLR.
    
    Only differnce made by us is as follows. 
    (1) We apply the decoder module to '(heterogenuous) groups'  
    
    In addition, the decoder uses recurrent-neural-net
    \"""

    # Arguments 
    ___________    
    num_heteros : int 
        the number of heterogeneous groups           
    rel_rec : torch.FloatTensor 
        relation-receiver 
        the shape of rel_rec is 'bs x c x n^2 x n 
        (this argumnet is a property of GraphLearningEncoder)
    rel_send : torch.FloatTensor 
        relation-sender 
        the shape of rel_send is 'bs x c x n^2 x n
        (this argumnet is a oroperty of GraphLearningEncoder)
    rel_adj : torch.FlaotTensor 
        relation-adjacency matrix 
        the shape of rel_adj is 'bs x c x n x n
        (this argumnet is an ouput of GraphLearningEncoder-forward)
    x : torch.FloatTensor (in the forward function)
        shape of x is 'bs x c x t x n' 
        where, 
        bs: batch size 
        c : the number of eNB 
        t : the number of time-stamps 
        n : the number of time-seris  
    """

    def __init__(self, 
                num_heteros: int, 
                rel_rec, 
                rel_send): 
        super().__init__() 