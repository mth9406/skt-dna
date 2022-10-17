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