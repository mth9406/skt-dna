import torch 
from torch import nn

from layers.layers import * 
from layers.graphLearningLayers import * 
from layers.nriLayers import *

class MTGNN(nn.Module): 
    r"""Multivariate Time Series Forecasting with Graph Neural Networks 
    
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
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, num_ts, **kwargs))
        
        # hetero adjacency matrices
        self.ts_idx = torch.LongTensor(list(range(num_ts))).to(device) # to device...
        self.gen_adj = nn.ModuleList([AdjConstructor(num_ts, embedding_dim, alpha, top_k= top_k) for _ in range(num_heteros)])
    
        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            nn.Conv2d(num_heteros*(num_blocks+2), num_heteros, kernel_size=1,  groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros),
            nn.LeakyReLU(negative_slope= 0.5), 
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.LeakyReLU(negative_slope= 0.5),
            nn.Conv2d(num_heteros, num_heteros, groups= num_heteros, kernel_size= 1, padding= 0)
            )), 
            nn.LeakyReLU(negative_slope= 0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), padding= 0), 
            nn.Tanh()
        )
            
        # self.mask_block = nn.Sequential(
        #     ResidualAdd(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros)),
        #     nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags,1), groups= num_heteros)
        # )

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
        
        # outs_mask = torch.sigmoid(self.mask_block(mask_batch)) # masks

        return {
            'preds': outs_label, 
            'outs_label': outs_label, 
            'outs_mask': None,
            'kl_loss': None, 
            'adj_mat': None
        }

def gumbel_softmax(logits, tau, hard= True, dim= 1):
    r"""
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
    return -kl_div.sum() / (num_ts * preds.size(0))    

class NRI(nn.Module): 
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
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples       
    kwargs : key word arguments
        * groups
        * drop_p 
        * ...
    """
    def __init__(self, 
                 num_heteros: int,
                 num_time_series: int, 
                 time_lags: int, 
                 tau: float, 
                 n_hid_encoder: int, 
                 msg_hid: int, 
                 msg_out: int, 
                 n_hid_decoder: int,
                 device,
                 **kwargs
                ):
        super().__init__()

        #edge weights have dim 2
        self.encoder = MLPEncoder(time_lags * num_time_series, n_hid_encoder, 2)
        self.decoder = MLPDecoder(n_in_node=num_time_series,
                                edge_types=2,
                                msg_hid=msg_hid,
                                msg_out=msg_out,
                                n_hid=n_hid_decoder)
        self.rel_rec, self.rel_send = generate_off_diag(num_heteros, device= device)
        
        self.num_heteros = num_heteros
        self.num_time_series = num_time_series
        self.time_lags = time_lags 
        self.tau = tau  
        self.device= device

    def forward(self, x, beta= None):
        x_batch = x['input'] # bs, c, t, n

        logits = self.encoder(x_batch, self.rel_rec, self.rel_send)
        if self.training: 
            edges = nri_gumbel_softmax(logits, self.tau, hard= False)
        else: 
            edges = nri_gumbel_softmax(logits, self.tau, hard= True)
        prob = nri_softmax(logits, -1)
        output = self.decoder(x_batch, edges, self.rel_rec, self.rel_send, self.time_lags)
        kl_loss = kl_categorical_uniform(prob, self.num_heteros, 2)
        recon_loss = ((x_batch[:, :, 1:, :] - output[:, :, :-1, :]) ** 2).mean() 
        _, rel = logits.max(-1)
        A = []
        for i in range(rel.shape[0]):
            A.append(coo_to_adj(rel[i], self.num_heteros)) # bs, c, c 
        A = torch.stack(A, dim= 0).to(self.device)
        return {
            'preds': output[:, :, -1:, :], 
            'outs_label': output[:, :, -1:, :], 
            'outs_mask': None, 
            'kl_loss': -kl_loss+recon_loss, 
            'adj_mat': A          
        }

class HeteroNRI(nn.Module): 
    r"""
    
    This work is based on the work by 
    
    \"""
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). 
    Neural relational inference for interacting systems. 
    In International Conference on Machine Learning (pp. 2688-2697). PMLR.
    \"""

    It differs from NRI inthat it considers the followings
    (1) Heterogenous properties of relational graph among objects : GroupedConvolution
    (2) Temporal-spatial property of a time-series : Fully Connected Graph is used / CausalConvolution
    (3) Considers Periods : Dilated convolution
    (4) GraphResidualNet is used as a decoder block

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
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples       
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
                tau:float, 
                device, 
                **kwargs):
        super().__init__() 
        
        # encoder 
        self.glem = GraphLearningEncoderModule(num_heteros, time_lags, num_ts, device)
        
        # decoder
        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, num_ts, **kwargs))

        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            nn.Conv2d(num_heteros*(num_blocks+2), num_heteros, kernel_size=1,  groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros),
            nn.LeakyReLU(negative_slope= 0.5), 
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.LeakyReLU(negative_slope= 0.5),
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, padding= 0)
            )), 
            nn.LeakyReLU(negative_slope= 0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), groups= num_heteros, padding= 0), 
            nn.Tanh()
        )
            
        self.mask_block = nn.Sequential(
            ResidualAdd(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros, num_ts)),
            nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags,1), groups= num_heteros)
        )

        # decoder arguments
        self.num_heteros = num_heteros
        self.num_ts = num_ts 
        self.time_lags = time_lags 
        self.num_blocks = num_blocks 
        self.k = k 
        # encoder arguments
        self.tau = tau  
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
        if self.training: 
            A = gumbel_softmax(h, self.tau, hard= False, dim=-2)
        else: 
            A = gumbel_softmax(h, self.tau, hard= True, dim=-2)
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

class HeteroNRIMulti(nn.Module): 
    r"""    

    This work is based on the work by 
    
    \"""
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). 
    Neural relational inference for interacting systems. 
    In International Conference on Machine Learning (pp. 2688-2697). PMLR.
    \"""

    It differs from NRI inthat it considers the followings
    (1) Heterogenous properties of relational graph among objects : GroupedConvolution
    (2) Temporal-spatial property of a time-series : Fully Connected Graph is used / CausalConvolution
    (3) Considers Periods : Dilated convolution
    (4) GraphResidualNet is used as a decoder block

    In addition, 
    it provides 'online learning' structure and 'multi-time stamp' prediction.
    """

    def __init__(self, backbone, pred_steps, device):
        super().__init__() 
        self.backbone = backbone 
        self.device= device

        self.pred_steps = pred_steps

    def forward(self, x, beta): 
        return self.auto_regressive_forward(x, beta)     

    def auto_regressive_forward(self, x, beta): 
        x_batch, mask_batch = x['input'], x['mask']
        bs, c, n, d = x_batch.shape
        outs = torch.zeros((bs, c, self.pred_steps, d), requires_grad= False, device= self.device)
        outs_label = torch.zeros((bs, c, self.pred_steps, d), requires_grad= False, device= self.device)
        outs_mask = torch.zeros((bs, c, self.pred_steps, d), requires_grad= False, device= self.device)
        adj_mats = torch.zeros((bs, self.pred_steps, c, d, d), requires_grad= False, device= self.device)
        kl_loss = 0 

        out = self.backbone.forward(x, beta)
        outs[...,0:1,:] = out['preds']
        outs_label[...,0:1,:] = out['outs_label']
        outs_mask[...,0:1,:] = out['outs_mask']
        adj_mats[:, 0, ...]  = out['adj_mat']
        kl_loss = kl_loss + out['kl_loss'] if out['kl_loss'] is not None else None 

        for t in range(1, self.pred_steps): 
            if self.training:
                # teacher-forcing 
                ar_x_batch, ar_mask_batch = torch.cat((x_batch[..., t:, :], x['label'][..., :t, :]), dim= -2), torch.cat((mask_batch[..., t:, :], x['label_mask'][..., :t, :]), dim= -2)
            else:
                ar_x_batch, ar_mask_batch = torch.cat((x_batch[..., t:, :], outs_label[..., :t, :]), dim= -2), torch.cat((mask_batch[..., t:, :], outs_mask[..., :t, :]), dim= -2)
            ar_x = {
                'input': ar_x_batch,
                'mask': ar_mask_batch
            }
            out = self.backbone.forward(ar_x, beta)
            outs[...,t:t+1,:] = out['preds']
            outs_label[...,t:t+1,:] = out['outs_label']
            outs_mask[...,t:t+1,:] = out['outs_mask']
            adj_mats[:,t, ...]  = out['adj_mat']
            kl_loss = kl_loss + out['kl_loss'] if out['kl_loss'] is not None else None

        if kl_loss is not None: 
            kl_loss /= self.pred_steps

        return {
            'preds': outs, 
            'outs_label': outs_label, 
            'outs_mask': outs_mask, 
            'kl_loss': kl_loss, 
            'adj_mat': adj_mats            
        }

class MTGNNMulti(nn.Module): 
    r"""Multivariate Time Series Forecasting with Graph Neural Networks 
    
    This work is heavily based on the work by 
    
    \"""
    Connecting the dots: multivariate time series forecasting with graph neural networks 

    Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020, August). 
    Connecting the dots: Multivariate time series forecasting with graph neural networks. 
    In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 753-763).
    \"""

    but predicts multi-step auto-regressive way 
    so that it does not change the pretrained structure.
    """

    def __init__(self, backbone, pred_steps, device):
        super().__init__() 
        self.backbone = backbone 
        self.device= device

        self.pred_steps = pred_steps

    def forward(self, x, beta): 
        return self.auto_regressive_forward(x, beta)     

    def auto_regressive_forward(self, x, beta): 
        x_batch, mask_batch = x['input'], x['mask']
        bs, c, n, d = x_batch.shape
        outs = torch.zeros((bs, c, self.pred_steps, d), requires_grad= False, device= self.device)

        out = self.backbone.forward(x, beta)
        outs[...,0:1,:] = out['preds']

        for t in range(1, self.pred_steps): 
            if self.training:
                # teacher-forcing 
                ar_x_batch = torch.cat((x_batch[..., t:, :], x['label'][..., :t, :]), dim= -2)
            else:
                ar_x_batch = torch.cat((x_batch[..., t:, :], out[..., :t, :]), dim= -2)
            ar_x = {
                'input': ar_x_batch,
                'mask': None
            }
            out = self.backbone.forward(ar_x, beta)
            outs[...,t:t+1,:] = out['preds']

        return {
            'preds': outs, 
            'outs_label': outs, 
            'outs_mask': None, 
            'kl_loss': None, 
            'adj_mat': None           
        }

class NRIMulti(nn.Module): 
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
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples       
    kwargs : key word arguments
        * groups
        * drop_p 
        * ...
    """
    def __init__(self, 
                 num_heteros: int,
                 num_time_series: int, 
                 time_lags: int, 
                 tau: float, 
                 n_hid_encoder: int, 
                 msg_hid: int, 
                 msg_out: int, 
                 n_hid_decoder: int,
                 pred_steps: int,
                 device,
                 **kwargs
                ):
        super().__init__()

        #edge weights have dim 2
        self.encoder = MLPEncoder(time_lags * num_time_series, n_hid_encoder, 2)
        self.decoder = MLPDecoder(n_in_node=num_time_series,
                                edge_types=2,
                                msg_hid=msg_hid,
                                msg_out=msg_out,
                                n_hid=n_hid_decoder)
        self.rel_rec, self.rel_send = generate_off_diag(num_heteros, device= device)
        
        self.num_heteros = num_heteros
        self.num_time_series = num_time_series
        self.time_lags = time_lags 
        self.tau = tau  
        self.pred_steps = pred_steps
        self.device= device

    def forward(self, x, beta= None):
        x_batch = x['input'] # bs, c, t, n

        logits = self.encoder(x_batch, self.rel_rec, self.rel_send)
        if self.training: 
            edges = nri_gumbel_softmax(logits, self.tau, hard= False)
        else: 
            edges = nri_gumbel_softmax(logits, self.tau, hard= True)
        prob = nri_softmax(logits, -1)
        output = self.decoder(x_batch, edges, self.rel_rec, self.rel_send, self.time_lags)
        kl_loss = kl_categorical_uniform(prob, self.num_heteros, 2)
        # recon_loss = ((x_batch[:, :, 1:, :] - output[:, :, :-1, :]) ** 2).mean() 
        _, rel = logits.max(-1)
        A = []
        for i in range(rel.shape[0]):
            A.append(coo_to_adj(rel[i], self.num_heteros)) # bs, c, c 
        A = torch.stack(A, dim= 0).to(self.device)
        return {
            'preds': output[:, :, -self.pred_steps:, :], 
            'outs_label': output[:, :, -self.pred_steps:, :], 
            'outs_mask': None, 
            'kl_loss': -kl_loss, 
            'adj_mat': A          
        }

class HeteroSpatialNRI(nn.Module): 
    r"""
    
    This work is based on the work by 
    
    \"""
    Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). 
    Neural relational inference for interacting systems. 
    In International Conference on Machine Learning (pp. 2688-2697). PMLR.
    \"""

    It differs from the HeteroNRI in-that it considers inter-eNB relationship.

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
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples       
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
                tau:float, 
                device, 
                **kwargs):
        super().__init__() 
        
        # encoder 
        self.glem = GraphLearningEncoderModule(num_heteros, time_lags, num_ts, device)
        
        # eNB-graph 
        # todo 
        self.eNB_adj = GraphLearningEncoderModule(1, time_lags, num_heteros, device, **kwargs)

        # decoder
        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, num_ts, **kwargs))

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

        self.eNB_emb = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, (1, num_ts), groups= num_heteros),
            nn.BatchNorm2d(num_heteros),
            nn.GELU()
        )
        
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), groups= num_heteros, padding= 0), 
            nn.Tanh()
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
        # encoder arguments
        self.tau = tau  
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
        if self.training: 
            A = gumbel_softmax(h, self.tau, hard= False, dim=-2)
        else: 
            A = gumbel_softmax(h, self.tau, hard= True, dim=-2)
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
        
        # generate eNB-graph
        # eNB_rep: (output_label) bs, c, 1, n --> bs, c, 1, 1 --> bs, 1, 1, c --> bs, 1, c, c
        # kl_loss_eNB = None      

        # fc_out 
        outs_label = self.fc_decode(outs_label) # bs, c, t, n
        eNB_rep = self.eNB_emb(outs_label)
        # (1, n) kernel covolution --> bs, c, t, 1  
        eNB_rep = torch.transpose(eNB_rep.squeeze(), -1, -2).contiguous() # bs, c, t
        eNB_rep = eNB_rep[:, None, ...]
        # resahpe --> bs ,1, t, c  

        h_enb = self.eNB_adj(eNB_rep) # bs, 1, t, c -->  bs, 1, c, c
        z_enb = F.softmax(h_enb, dim= -1) 
        kl_loss_eNB = kl_categorical_uniform(z_enb, self.num_ts)
        if self.training: 
            A_eNB = gumbel_softmax(h_enb, self.tau, hard= False, dim=-1).squeeze() # bs x c x c
        else: 
            A_eNB = gumbel_softmax(h_enb, self.tau, hard= True, dim=-1).squeeze() # bs x c x c  
        
        outs_label = self.fc_out(outs_label).permute(0, 2, 1, 3).squeeze() # bs, c, 1, n --> bs, 1, c, n
        outs_label = torch.matmul(A_eNB, outs_label) # (bs, c, c), (bs, c, n) --> (bs, c, n)
        outs_label = outs_label[:, :, None, :] 
        
        outs_mask = torch.sigmoid(self.mask_block(mask_batch)) # masks

        return {
            'preds': outs_label * outs_mask, 
            'outs_label': outs_label, 
            'outs_mask': outs_mask, 
            'kl_loss': kl_loss+kl_loss_eNB, 
            'adj_mat': A, 
            'adj_mat_eNB': A_eNB
        }

class Multistep_MTGNN(nn.Module):
    r"""Multivariate Time Series Forecasting with Graph Neural Networks

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
                 num_heteros: int,
                 num_ts: int,
                 time_lags: int,
                 num_blocks: int,
                 k: int,
                 embedding_dim: int,
                 device,
                 alpha: float = 3.0, top_k: int = 4,
                 pred_steps = 1,
                 **kwargs):
        super().__init__()

        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups=num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks):
            setattr(self, f'hetero_block{i}', HeteroBlock(num_heteros, k, num_ts, **kwargs))

        # hetero adjacency matrices
        self.ts_idx = torch.LongTensor(list(range(num_ts))).to(device)  # to device...
        self.gen_adj = nn.ModuleList(
            [AdjConstructor(num_ts, embedding_dim, alpha, top_k=top_k) for _ in range(num_heteros)])

        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            nn.Conv2d(num_heteros * (num_blocks + 2), num_heteros, kernel_size=1, groups=num_heteros, padding=0),
            nn.BatchNorm2d(num_heteros),
            nn.LeakyReLU(negative_slope=0.5),
            ResidualAdd(nn.Sequential(
                nn.Conv2d(num_heteros, num_heteros, kernel_size=1, groups=num_heteros, padding=0),
                nn.BatchNorm2d(num_heteros),
                nn.LeakyReLU(negative_slope=0.5),
                nn.Conv2d(num_heteros, num_heteros, groups=num_heteros, kernel_size=1, padding=0)
            )),
            nn.LeakyReLU(negative_slope=0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags-pred_steps+1, 1), padding=0),
            nn.Tanh()
        )

        # self.mask_block = nn.Sequential(
        #     ResidualAdd(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros)),
        #     nn.Conv2d(num_heteros, num_heteros, kernel_size=(time_lags,1), groups= num_heteros)
        # )

        self.num_heteros = num_heteros
        self.num_ts = num_ts
        self.time_lags = time_lags
        self.num_blocks = num_blocks
        self.k = k
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.device = device
        self.alpha = alpha
        self.pred_steps = pred_steps

    def forward(self, x, beta):
        r"""Feed forward of the model
        Assume, x is a pair of x['input'] and x['mask']
        """
        # x_batch = make_input_n_mask_pairs(x, self.device)
        x_batch, mask_batch = x['input'], x['mask']

        x_batch = self.projection(x_batch)  # bs, c (=num_heteros), t, n
        bs, c, t, n = x_batch.shape
        A = torch.stack([gll(self.ts_idx) for gll in self.gen_adj]).to(self.device)  # c, n, n
        outs_label = torch.zeros((bs, c * (self.num_blocks + 2), t, n)).to(
            self.device)  # to collect outputs from modules
        out = x_batch.clone().detach()
        outs_label[:, ::(self.num_blocks + 2), ...] = out
        for i in range(self.num_blocks):
            tc_out, out = getattr(self, f'hetero_block{i}')(out, A, beta)
            # x_batch += tc_out
            outs_label[:, (i + 1)::(self.num_blocks + 2), ...] = tc_out
            # x_batch = torch.cat([x_batch, tc_out], dim= 1)
        # x_batch += out # bs, c, n, l
        outs_label[:, (self.num_blocks + 1)::(self.num_blocks + 2), ...] = out
        # x_batch = torch.cat([x_batch, out], dim=1)

        # fc_out
        outs_label = self.fc_decode(outs_label)
        outs_label = self.fc_out(outs_label)

        # outs_mask = torch.sigmoid(self.mask_block(mask_batch)) # masks

        return {
            'preds': outs_label,
            'outs_label': outs_label,
            'outs_mask': None,
            'kl_loss': None,
            'adj_mat': None
        }