import torch  
from torch import nn 

from layers.temporal_convolution_layers import * 

class GCNBlock(nn.Module):
    r"""Graph Convolution Network Block 
    """
    def __init__(self, num_heteros):
        super().__init__()
        self.conv2d_src = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.conv2d_dst = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups = num_heteros)
        self.self_conv2d_src = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.self_conv2d_dst = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros) 

    def forward(self, emb_src, emb_dst, adj_mat):
        r"""
        emb_src: embedding vectors of input multi-variate time-series data (bs, c, 1, num_src) 
        emb_dst: multi-variate response variable (bs, c, 1, num_dst)
        adj_mat: adjacency matrix (bs, c, num_src, num_dst)  
        """
        emb_src_t, emb_dst_t = emb_src.transpose(2,3), emb_dst.transpose(2,3) # (bs, c, num_src (num_dst), 1) 
        adj_mat2dst = adj_mat.transpose(2,3) # aggregate dst nodes # (bs, c, num_dst, num_src)

        # emb_src_t = self.self_conv2d_src(emb_src_t) + self.conv2d_src(adj_mat@emb_dst_t)
        emb_dst_t = self.self_conv2d_dst(emb_dst_t) + self.conv2d_dst(adj_mat2dst@emb_src_t)
        
        # emb_src = emb_src_t.transpose(2,3)
        emb_dst = emb_dst_t.transpose(2,3)

        return emb_dst