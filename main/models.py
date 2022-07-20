import torch 
from torch import nn

from layers import * 
from graphLearningLayers import * 

class HeteroMTGNN(nn.Module): 
    r"""Heterogeneous Multivariate Time Series Forecasting with Graph Neural Networks 
    
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
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, **kwargs))
        
        # hetero adjacency matrices
        self.ts_idx = torch.LongTensor(list(range(num_ts))).to(device) # to device...
        self.gen_adj = nn.ModuleList([AdjConstructor(num_ts, embedding_dim, alpha, top_k= top_k) for _ in range(num_heteros)])
    
        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            nn.Conv2d(num_heteros*(num_blocks+2), num_heteros, kernel_size=1,  groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros),
            nn.GELU(), 
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.GELU(),
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, padding= 0)
            )), 
            nn.GELU()
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), padding= 0), 
            nn.ReLU()
        )
            
        self.mask_block = nn.Sequential(
            ResidualAdd(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros)),
            nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags,1), groups= num_heteros)
        )

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
        r"""Feed forward of the model 
        Assume, x is a pair of x['input'] and x['mask']
        """
        # x_batch = make_input_n_mask_pairs(x, self.device)
        x_batch, mask_batch = x['input'], x['mask']
        x_batch = self.projection(x_batch) # bs, c (=num_heteros), t, n 
        bs, c, t, n = x_batch.shape
        A = torch.stack([gll(self.ts_idx) for gll in self.gen_adj]).to(self.device) # c, n, n 
        outs_label = torch.zeros((bs, c * (self.num_blocks+2), t, n)).to(self.device) # to collect outputs from modules
        out = x_batch.clone().detach()
        outs_label[:, ::(self.num_blocks+2), ...] = out
        for i in range(self.num_blocks): 
            tc_out, out = getattr(self, f'hetero_block{i}')(out, A, beta)
            # x_batch += tc_out 
            outs_label[:, (i+1)::(self.num_blocks+2), ...] = tc_out
            # x_batch = torch.cat([x_batch, tc_out], dim= 1)
        # x_batch += out # bs, c, n, l 
        outs_label[:, (self.num_blocks+1)::(self.num_blocks+2), ...] = out
        # x_batch = torch.cat([x_batch, out], dim=1)
        
        # fc_out 
        outs_label = self.fc_decode(outs_label)
        outs_label = self.fc_out(outs_label)
        
        outs_mask = torch.sigmoid(self.mask_block(mask_batch)) # masks

        return {
            'preds': outs_label * outs_mask, 
            'outs_label': outs_label, 
            'outs_mask': outs_mask,
            'kl_loss': None, 
            'adj_mat': None
        }

def gumbel_softmax(logits, tau, hard= True, dim= 1):
    """
    returns a continuous approximation of the discrete distribution 
    # Arguments   
    ___________       
    logits: torch FloatTensor type 
        logits    
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples   
    hard: bool 
        hard sampling if set true 
    """ 
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim= dim)

def kl_categorical_uniform(preds, num_ts, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    # if add_const:
    #     const = np.log(num_edge_types)
    #     kl_div += const
    return kl_div.sum() / (num_ts * preds.size(0))    

class HeteroNRI(nn.Module): 
    r"""
    
    This work is based on the work by 
    
    \"""
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). 
    Neural relational inference for interacting systems. 
    In International Conference on Machine Learning (pp. 2688-2697). PMLR.
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
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples   
    hard: bool 
        hard sampling if set true     
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
                tau:float, 
                hard: bool, 
                device, 
                alpha: float = 3.0, top_k: int = 4, **kwargs):
        super().__init__() 
        
        # encoder 
        self.glem = GraphLearningEncoderModule(num_heteros, time_lags, num_ts, device)
        
        # decoder
        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, **kwargs))

        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            nn.Conv2d(num_heteros*(num_blocks+2), num_heteros, kernel_size=1,  groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros),
            nn.GELU(), 
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.GELU(),
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0)
            )), 
            nn.GELU()
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), groups= num_heteros, padding= 0), 
            nn.ReLU()
        )
            
        self.mask_block = nn.Sequential(
            ResidualAdd(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros)),
            nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags,1), groups= num_heteros)
        )

        # decoder arguments
        self.num_heteros = num_heteros
        self.num_ts = num_ts 
        self.time_lags = time_lags 
        self.num_blocks = num_blocks 
        self.k = k 
        self.top_k = top_k
        self.alpha = alpha 
        self.embedding_dim = embedding_dim
        # encoder arguments
        self.tau = tau 
        self.hard = hard  
        # device
        self.device= device 
        
    def forward(self, x, beta): 
        x_batch, mask_batch = x['input'], x['mask']
        kl_loss = None
        # encoder
        h = self.glem(x_batch) # bs, c, n, n 
        z = F.softmax(h, dim= -2) # softmax along row dimension. (col-sum = 1.)
        # obtain kl_loss 
        kl_loss = kl_categorical_uniform(z, self.num_ts)
        A = gumbel_softmax(h, self.tau, self.hard, dim=1)
        # decoder 
        x_batch = self.projection(x_batch) 
        bs, c, t, n = x_batch.shape 

        outs_label = torch.zeros((bs, c * (self.num_blocks+2), t, n)).to(self.device) # to collect outputs from modules
        out = x_batch.clone().detach()
        outs_label[:, ::(self.num_blocks+2), ...] = out
        for i in range(self.num_blocks): 
            tc_out, out = getattr(self, f'hetero_block{i}')(out, A, beta)
            # x_batch += tc_out 
            outs_label[:, (i+1)::(self.num_blocks+2), ...] = tc_out
            # x_batch = torch.cat([x_batch, tc_out], dim= 1)
        # x_batch += out # bs, c, n, l 
        outs_label[:, (self.num_blocks+1)::(self.num_blocks+2), ...] = out
        # x_batch = torch.cat([x_batch, out], dim=1)
        
        # fc_out 
        outs_label = self.fc_decode(outs_label)
        outs_label = self.fc_out(outs_label)
        
        outs_mask = torch.sigmoid(self.mask_block(mask_batch)) # masks

        return {
            'preds': outs_label * outs_mask, 
            'outs_label': outs_label, 
            'outs_mask': outs_mask, 
            'kl_loss': kl_loss, 
            'adj_mat': A
        }
