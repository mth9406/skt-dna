import torch  
from torch import nn 
from torch.nn import functional as F
import numpy as np 

from layers.temporal_convolution_layers import * 
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_sum

def generate_bipartite(num_src:int, num_dst:int, device= None): 
    r"""Generates a bipartite graph
    
    # Arguments          
    ___________                
    num_src : int
        the number of source nodes
    num_dst : int 
        the number of destination nodes               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    src : torch.LongTensor    
        source nodes     
    dst : torch.LongTensor    
        destination nodes     
    """
    bipartite_fcn = np.ones([num_src, num_dst])
    src, dst =  np.nonzero(bipartite_fcn)
    
    src = torch.LongTensor(src) if device is None \
        else torch.LongTensor(src).to(device)
    dst = torch.LongTensor(dst) if device is None \
        else torch.LongTensor(dst).to(device)    

    return src, dst

# Graph Learning Layer - Encoder 
class GraphLearningEncoder(nn.Module): 
    r""" Encoder module using TemporalConvolutionModule defined in /layers.temporal_convolution_layers.py 
    # Arguments
    ___________
    num_heteros : int 
        the number of heterogeneous groups      
    
    # forwards
    __________
    returns adjacency matrix (logits) for every item in a batch

    """
    def __init__(self, num_heteros:int, time_lags:int, num_ts:int,
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                **kwargs): 
        super().__init__() 
        self.num_heteros = num_heteros
        self.time_lags = time_lags
        self.device = device
        self.tcm = nn.Sequential(
            TemporalConvolutionModule(num_heteros, num_heteros, num_heteros=num_heteros, num_time_series=num_ts, **kwargs),
            nn.Conv2d(num_heteros, num_heteros, (time_lags-1, 1), groups= num_heteros)) 
        self.node2edge_conv = nn.Conv2d(num_heteros, num_heteros, (1, 2), groups= num_heteros)
        self.edge2src_node_conv = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.edge2dst_node_conv = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.node2edge_conv_2 = nn.Conv2d(num_heteros, num_heteros, (1, 3), groups= num_heteros) 
    
    def forward(self, x, y, src, dst): 
        bs, c, t, num_src = x.shape
        bs, c, _, num_dst = y.shape
        # (0) x_cause embedding 
        x_cause = self.tcm(x[..., :-1, :]) # bs, c, 1, num_src
        x_cause = x_cause[..., -1:, :].transpose(-2, -1) # bs, c, num_src, 1
        y_response = y[..., -1:, :].transpose(-2, -1) # bs, c, num_dst, 1
        # (1) node2edge 
        h_e = torch.cat([x_cause[..., src, :], y_response[..., dst, :]], dim= 3) # bs, c, num_src * num_dst, 2
        h_e = self.node2edge_conv(h_e) # bs, c, num_src * num_dst, 1
        h_e_skip = h_e # bs, c, num_src * num_dst, 1
        # (2) edge2node 
        x_cause =  F.leaky_relu(self.edge2src_node_conv(scatter_mean(h_e, src, dim= 2))) # bs, c, num_src, 1
        y_response = F.leaky_relu(self.edge2dst_node_conv(scatter_mean(h_e, dst, dim= 2))) # bs, c, num_dst, 1
        # (3) node2edge 
        h_e = torch.cat([x_cause[..., src, :], y_response[..., dst, :]], dim= 3) # bs, c, num_src * num_dst, 2
        h_e = torch.cat([h_e, h_e_skip], dim= 3) # bs, c, num_src * num_dst, 3
        h_e = F.leaky_relu(self.node2edge_conv_2(h_e)) # bs, c, num_src * num_dst, 1 
        h_e = h_e.squeeze().reshape((bs, c, num_src, num_dst)) # bs, c, num_src, num_dst 

        return h_e

class GraphLearningEncoderModule(nn.Module): 
    r"""GraphLearningEncoderModule    
    VAE is used as a graph learning encoder.   
    It uses 2D-group-convolution to send and aggregate messages from nodes and edges     
    # Arguments       
    ___________              
    num_heteros : int    
        the number of heterogeneous groups (stack along the channel dimension)
    time_lags: int 
        the size of 'time_lags'       
    num_ts : int     
        the number of time-series    
        should be 10 for the skt-data   

    # Fowards    
    _________    
    returns logits of size (bs, c, num_src, num_dst)

    """

    def __init__(self, num_heteros, time_lags, num_src, num_dst, num_ts,
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                **kwargs): 
        super().__init__()

        self.src, self.dst = generate_bipartite(num_src, num_dst, device= device)
        self.gle = GraphLearningEncoder(num_heteros, time_lags, num_ts, device= device, **kwargs) 
        self.num_heteros, self.time_lags, self.num_src, self.num_dst = num_heteros, time_lags, num_src, num_dst

    def forward(self, x, y): 
        logits = self.gle(x, y, self.src, self.dst)
        return logits
