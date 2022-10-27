import torch 
from torch import nn

from layers.layers import * 
from layers.graphLearningLayers import * 
from layers.nriLayers import *

class CCM(nn.Module): 
    r"""    
    CausalConvolutionModel
    Ablation study of HeteroNRI
    only uses causal convolution (w/o grouped)
    """

    def __init__(self, 
                num_heteros:int,
                num_ts: int,  
                time_lags: int, 
                num_blocks:int, 
                device, 
                **kwargs):
        super().__init__() 
        
        # decoder
        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'tcm{i}', TemporalConvolutionModule(num_heteros, num_heteros, num_heteros, num_ts, **kwargs))

        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.LeakyReLU(negative_slope= 0.5),
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1,  groups= num_heteros, padding= 0)
            )), 
            nn.LeakyReLU(negative_slope= 0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), groups= num_heteros, padding= 0), 
            nn.Tanh()
        )

        # decoder arguments
        self.num_heteros = num_heteros
        self.num_ts = num_ts 
        self.time_lags = time_lags 
        self.num_blocks = num_blocks 
        # device
        self.device= device 
        
    def forward(self, x, beta): 
        x_batch, mask_batch = x['input'], x['mask']
        kl_loss = None
        # encoder
        # decoder 
        x_batch = self.projection(x_batch) 
        bs, c, t, n = x_batch.shape 

        outs_label = self.tcm0(x_batch)
        for i in range(1, self.num_blocks):
            out_res = outs_label 
            outs_label = F.leaky_relu(getattr(self, f'tcm{i}')(outs_label)+out_res, negative_slope= 0.5)
        
        # fc_out 
        outs_label = self.fc_decode(outs_label)
        outs_label = self.fc_out(outs_label)

        return {
            'preds': outs_label, 
            'outs_label': outs_label, 
            'outs_mask': None, 
            'kl_loss': kl_loss, 
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


class HeteroNRINoGroup(nn.Module): 
    r"""
    HeteroNRINoGroup
    Ablation study of HeteroNRI
    without groupwise convolution in the decder.
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
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, groups= num_heteros, padding= 0)
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

class HCCM(nn.Module): 
    r"""
    HeteroCausalConvolutionModel
    Ablation study of HeteroNRI
    only uses causal convolution 
    """
    def __init__(self, 
                num_heteros:int,
                num_ts: int,  
                time_lags: int, 
                num_blocks:int, 
                device, 
                **kwargs):
        super().__init__() 
        
        # decoder
        # projection layer
        self.projection = ProjectionConv1x1Layer(num_heteros, num_heteros, groups= num_heteros, **kwargs)
        # hetero blocks
        for i in range(num_blocks): 
            setattr(self, f'tcm{i}', TemporalConvolutionModule(num_heteros, num_heteros, num_heteros, num_ts, **kwargs))

        # output_module
        # self.fc_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags), padding= 0)
        self.fc_decode = nn.Sequential(
            ResidualAdd(nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1, padding= 0), 
            nn.BatchNorm2d(num_heteros), 
            nn.LeakyReLU(negative_slope= 0.5),
            nn.Conv2d(num_heteros, num_heteros, kernel_size= 1,  groups= num_heteros, padding= 0)
            )), 
            nn.LeakyReLU(negative_slope= 0.5)
        )
        self.fc_out = nn.Sequential(
            nn.Conv2d(num_heteros, num_heteros, kernel_size= (time_lags,1), groups= num_heteros, padding= 0), 
            nn.Tanh()
        )

        # decoder arguments
        self.num_heteros = num_heteros
        self.num_ts = num_ts 
        self.time_lags = time_lags 
        self.num_blocks = num_blocks 
        # device
        self.device= device 
        
    def forward(self, x, beta): 
        x_batch, mask_batch = x['input'], x['mask']
        kl_loss = None
        # encoder
        # decoder 
        x_batch = self.projection(x_batch) 
        bs, c, t, n = x_batch.shape 

        outs_label = self.tcm0(x_batch)
        for i in range(1, self.num_blocks):
            out_res = outs_label 
            outs_label = F.leaky_relu(getattr(self, f'tcm{i}')(outs_label)+out_res, negative_slope= 0.5)
        
        # fc_out 
        outs_label = self.fc_decode(outs_label)
        outs_label = self.fc_out(outs_label)

        return {
            'preds': outs_label, 
            'outs_label': outs_label, 
            'outs_mask': None, 
            'kl_loss': kl_loss, 
            'adj_mat': None
        }