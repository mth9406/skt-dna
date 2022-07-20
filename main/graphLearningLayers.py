import torch  
from torch import nn 
from torch.nn import functional as F
import numpy as np 

from layers import *

def encode_onehot(labels): 
    """ Encode some relational masks specifying which vertices receive messages from which other ones.
    # Arguments          
    ___________             
    labels : np.array type 
    
    # Returns        
    _________          
    labels_one_hot : np.array type            
        adjacency matrix
    
    # Example-usage       
    _______________            
    >>> labels = [0,0,0,1,1,1,2,2,2]
    >>> labels_onehot = encode_onehot(labels)
    >>> labels_onehot 
    array(
        [[1, 0, 0],
         [1, 0, 0],             
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 1],            
         [0, 0, 1]], dtype=int32)      
    """
    classes = set(labels) 
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype= np.int32)
    return labels_onehot

def generate_fcn(num_objects, device= None): 
    """Generates fcn (Fully Connected Graph)
    
    # Arguments          
    ___________                
    num_objects : int
        the number of objects               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    rel_rec : torch.FloatTensor    
        relation-receiver     
    rel_send : torch.FloatTensor    
        relation-receiver     
    """
    fcn = np.ones([num_objects, num_objects])
    
    rec, send = np.where(fcn)
    rel_rec = np.array(encode_onehot(rec), dtype= np.float32)
    rel_send = np.array(encode_onehot(send), dtype= np.float32)
    
    rel_rec = torch.FloatTensor(rel_rec) if device is None \
        else torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send) if device is None \
        else torch.FloatTensor(rel_send).to(device)
    
    return rel_rec, rel_send  

# Graph Learning Layer - Encoder 
class GraphLearningEncoder(nn.Module): 
    """ Encoder module using TemporalConvolutionModule defined in layers.py 
    # Arguments
    ___________
    num_heteros : int 
        the number of heterogeneous groups      
    
    # forwards
    __________
    returns adjacency matrix for every item in a batch

    """
    def __init__(self, num_heteros, time_lags, **kwargs): 
        super().__init__() 
        self.num_heteros = num_heteros
        self.time_lags = time_lags
        self.tcm = nn.Sequential(TemporalConvolutionModule(num_heteros, num_heteros, num_heteros=num_heteros, **kwargs),nn.Conv2d(num_heteros, num_heteros, (time_lags, 1))) 
        self.node2edge_conv = nn.Conv2d(num_heteros, num_heteros, (1, 2), groups= num_heteros)
        self.edge2node_conv = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.node2edge_conv_2 = nn.Conv2d(num_heteros, num_heteros, (1, 3), groups= num_heteros) 
        self.conv_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags))

    def edge2node(self, x, rel_rec, rel_send):
        # fully-connected-graph. 
        incoming = torch.matmul(rel_rec.permute(0,2,1), x) # (c, n, n^2) x (bs, c, n^2, 1) --> (bs, c, n, 1)
        return incoming / incoming.size(2)

    def node2edge(self, x, rel_rec, rel_send):
        # fully-connected-graph
        receivers = torch.matmul(rel_rec, x) # (c, n, n^2) x (bs, c, n, 1) --> (bs, c, n^2, 1)
        senders = torch.matmul(rel_send, x) # (c, n, n^2) x (bs, c, n, 1) --> (bs, c, n^2, 1)
        edges = torch.cat([senders, receivers], dim=-1) # (bs, c, n^2, 2)
        return edges
    
    def forward(self, x, rel_rec, rel_send): 
        """
        # forwards
        __________
        feed-forwards works as follows...     
        x : torch.FloatTensor     
            shape of x is 'bs x c x t x n'

        (1) TemporalConvolutionModule   
        h :nn.Conv2d(tcm(x))  
            shape of h is 'bs x c x 1 x n'
            reshape so that, 
            the shape of h is 'bs x c x n x 1' 
            'n' is the number of 'nodes' 
        
        (2) Node2Edge operation 
        h_e = [rel_rec @ h; rel_send @ h]
            the shape of h_e is 'bs x c x n^2 x 2
        h_e = conv2d(h_e)
            the shape of h_e is 'bs x c x n^2 x 1
        h_e_skip = h_e 

        (3) Edge2Node operation 
        h_n = rel_rec.t @ h_e 
            the shape of h_n is 'bs x c x n x 1'
        h_n = conv2d(h_n) 
            the shape of h_n is 'bs x c x n x 1' 
        
        (4) Node2Edge operation 
        h_e = [rel_rec @ h_n ; rel_send @ h_n] 
            the shape of h_e is 'bs x c x n^2 x 2'            
                     
        (5) Skip connection 
        h_e = [h_e; h_e_skip] 
            the shape of h_e is 'bs x c x n^2 x 3'    
        h_e = conv2d(h_e)     
            the shape of h_e is 'bs x c x n^2 x 1'    

        (6) reshape the logits
        adj = h_e.reshape(bs, c, n, n) 
        """
        bs, c, t, n = x.shape
        # (1)
        h = self.tcm(x).permute(0, 1, 3, 2) # bs, c, n, 1 
        # print(f'(1) {h.shape}')
        # (2) 
        h = self.node2edge(h, rel_rec, rel_send) # bs, c, n^2, 2
        h = self.node2edge_conv(h) # bs, c, n^2, 1 
        h_skip = h # bs, c, n^2, 1 
        # print(f'(2) {h.shape}')
        # (3) 
        h = self.edge2node(h, rel_rec, rel_send) # bs, c, n, 1 
        h = self.edge2node_conv(h) # bs, c, n, 1 
        # print(f'(3) {h.shape}')
        # (4) 
        h = self.node2edge(h, rel_rec, rel_send) # bs x c x n^2 x 2
        # print(f'(4) {h.shape}')
        # (5) 
        h = torch.concat((h, h_skip), dim= -1) # bs x c x n^2 x 3 
        h = self.node2edge_conv_2(h) # bs x c x n^2 x 1
        # print(f'(5) {h.shape}')
        h = h.squeeze().reshape((bs, c, n, n)) # bs x c x n x n
        return h

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
    """

    def __init__(self, num_heteros, time_lags, num_ts, device, **kwargs): 
        super().__init__()

        # generates fully-connected-graph
        rel_rec, rel_send = [], []
        for i in range(num_heteros): 
            rec, send = generate_fcn(num_ts)
            rel_rec.append(rec); rel_send.append(send) 
        self.rel_rec = torch.stack(rel_rec, dim= 0).to(device)
        self.rel_send = torch.stack(rel_send, dim= 0).to(device)

        self.gle = GraphLearningEncoder(num_heteros, time_lags, **kwargs) 
        self.num_heteros, self.time_lags, self.num_ts = num_heteros, time_lags, num_ts
        # self.tau = tau
        # self.hard = hard

    def forward(self, x): 
        logits = self.gle(x, self.rel_rec, self.rel_send)
        return logits

