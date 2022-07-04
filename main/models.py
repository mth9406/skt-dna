import torch 
from torch import nn
from torch.nn import functional as F

from layers import * 

class HeteroBlock(nn.Module):
    """Hetero block 
    This block contains TC-Module + GC-Module and its residual connection.

    # Arguments
    ___________
    num_heteros : int
        the number of heterogeneous groups
    k : int 
        the number of layers at every GC-Module
    It takes the followings in the feed-forward procedures.
    * Adjacency matrix from the 'Graph-Learning-Layer' 
    * Another input from the previous 'HeteroBlock' module
    
    # Returns
    _________
    It returns two outputs 
    * output from TC-Module
    * output from GC-Module which takes an output of TC-Module as an input.
    """
    def __init__(self, num_heteros:int, k:int, **kwargs): 
        super().__init__() 
        self.tc_module = TemporalConvolutionModule(num_heteros, num_heteros, num_heteros, **kwargs)
        self.gc_module = GraphConvolutionModule(num_heteros, num_heteros, k=k, **kwargs)

        self.num_heteros = num_heteros
        self.k = k 
        # input shape: b, c, n, l 

    def forward(self, x, A, beta= 0.5): 
        """Feed forward        
        returns two outputs
        * output from TC-Module
        * output from GC-Module which takes an output of TC-Module as an input.     

        # Arguments
        ___________
        x : torch-tensor 
            Input tensor
        A : torch-tensor
            Adjacency matrix    
        """
        res = x 
        out_tc = self.tc_module(x) 
        x = self.gc_module(out_tc, A, beta= beta)
        return out_tc, torch.relu(x+res)

class HeteroMTGNN(nn.Module): 
    """Heterogeneous Multivariate Time Series Forecasting with Graph Neural Networks 
    
    This work is heavily based on the work by 
    
    \"""
    Connecting the dots: multivariate time series forecasting with graph neural networks 

    Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020, August). 
    Connecting the dots: Multivariate time series forecasting with graph neural networks. 
    In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 753-763).
    \"""

    # Arguments 
    ___________
    num_heteros : int 
        the number of heterogeneous groups (stack along the channel dimension)
    num_ts : int
        the number of time-series 
        should be 10 for the skt-data
    time_lags : int 
        the size of 'time_lags'
    num_blocks : int
        the number of the HeteroBlocks 
    k : int
        the number of layers at every GC-Module
    embedding_dim : int
        the size of embedding dimesion in the graph-learning layer 
    top_k : int
        top_k to select as non-zero in the adjacency matrix      
    alpha : float
        controls saturation rate of tanh: activation function in the graph-learning layer
        default = 3.0
    kwargs : key word arguments
        * groups
        * drop_p 
        * ...
    """
    def __init__(self, 
                num_heteros:int,
                num_ts: int,  
                time_lags: int, 
                num_blocks:int, 
                k:int, 
                embedding_dim: int, 
                device, 
                alpha: float = 3.0, top_k: int = 4, **kwargs): 
        super().__init__()
        
        # projection layer
        self.projection = ProjectionConv1x1Layer(2*num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, **kwargs))
        
        # hetero adjacency matrices
        self.ts_idx = torch.LongTensor(list(range(num_ts))).to(device) # to device...
        self.gen_adj = nn.ModuleList([AdjConstructor(num_ts, embedding_dim, alpha, top_k= top_k) for _ in range(num_heteros)])
    
        # output_module
        self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        # bs, c, n, l  -> bs, c, n, 1

        self.num_heteros = num_heteros
        self.num_ts = num_ts 
        self.time_lags = time_lags
        self.num_blocks = num_blocks
        self.k = k 
        self.embedding_dim = embedding_dim 
        self.top_k = top_k
        self.device = device 
        self.alpha = alpha 

    def forward(self, x, beta):
        """Feed forward of the model 
        Assume, x is a pair of x['input'] and x['mask']
        """
        x_batch = make_input_n_mask_pairs(x, self.device)
        x_batch = self.projection(x_batch) # bs, c, n, l 
        A = torch.stack([gll(self.ts_idx) for gll in self.gen_adj]).to(self.device) # c, n, n 
        out = x_batch.clone().detach()
        for i in range(self.num_blocks): 
            tc_out, out = getattr(self, f'hetero_block{i}')(out, A, beta)
            x_batch += tc_out 
        x_batch += out # bs, c, n, l 
        return {
            'preds': self.fc_out(x_batch)
        }
        
