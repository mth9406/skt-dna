import torch 
from torch import nn 

from layers.graph_convolution_layers import *
from layers.graph_learning_layer import * 
from layers.temporal_convolution_layers import * 

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

class CausalInferenceModel(nn.Module): 
    r"""
    The main contributions of this work are as follow.

    (1) Heterogenous properties of relational graph among objects (eNB in our case) : GroupedConvolution
    (2) Temporal property of a time-series : CausalConvolution (Grouped)
    (3) Considers Periods : Dilated convolution
    (4) Bipartite graph structure to infer causal relationship between explanatory variables and target variables
    (5) Basically a multi-task learning w.r.t. target variables. 

    # Arguments 
    ___________
    num_heteros : int 
        the number of heterogeneous groups (stack along the channel dimension)
    num_src : int
        the number of explanatory variables 
        should be 7 for the skt-data
    num_dst : int 
        the number of response variabales 
        should be 3 for the skt-data
    time_lags : int 
        the size of 'time_lags'
    num_blocks : int
        the number of the HeteroBlocks 
    tau: float 
        softmax temperature - a parameter that controls the smoothness of the samples       
    """

    def __init__(self, 
                num_heteros:int,
                num_src: int,  
                num_dst: int, 
                time_lags: int, 
                num_blocks_src:int=8, 
                num_blocks_dst:int=8,
                num_gcn_blocks:int=3, 
                tau:float= 0.1, 
                beta:float= 0.5,
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                ):
        super().__init__() 

        # encoder layers
        # tcm_src: To learn the representation of explanatory variables 
        for i in range(num_blocks_src): 
            setattr(self, f'tcm_src{i}', TemporalConvolutionModule(num_heteros, num_heteros, num_heteros))
        self.decode_src = nn.Conv2d(num_heteros, num_heteros, (time_lags, 1), groups= num_heteros)
        # tcm_dst: To learn the representation of response (target) variables 
        for i in range(num_blocks_dst): 
            setattr(self, f'tcm_dst{i}', TemporalConvolutionModule(num_heteros, num_heteros, num_heteros))
        self.decode_dst = nn.Conv2d(num_heteros, num_heteros, (time_lags, 1), groups= num_heteros)
        
        # graph sampling layer
        self.glem = GraphLearningEncoderModule(num_heteros, time_lags, num_src, num_dst, device)
        self.reconstruct_src = nn.Conv2d(num_heteros, num_heteros, kernel_size= (1,1), groups= num_heteros, padding= 0, bias= False)
        
        # graph convolution layers
        for i in range(num_gcn_blocks): 
            setattr(self, f'gcn{i}', GCNBlock(num_heteros))

        # encoder/decoder arguments
        self.num_heteros = num_heteros # the number of heterogenous groups
        self.num_src = num_src # the number of explanatory variables 
        self.num_dst = num_dst # the number of response variabales 
        self.time_lags = time_lags 
        self.num_blocks_src = num_blocks_src 
        self.num_blocks_dst = num_blocks_dst 
        self.num_gcn_blocks = num_gcn_blocks
        self.tau = tau  
        self.beta = beta 

        # device
        self.device= device 
        
        # criterion 
        self.criterion = nn.MSELoss()

    def forward(self, batch): 
        # get access to inputs 
        x_batch, y_batch = batch['exp_input'], batch['res_input'] # 'res' stands for 'response variable' 
        # shape of x_batch: bs, c, t, num_src 
        # shape of y_batch: bs, c, t, num_dst

        # (1) graph sampling 
        logits = self.glem(x_batch, y_batch) # bs, c, num_src, num_dst 
        adj_mat = gumbel_softmax(logits, self.tau, hard= False, dim= -1)
        
        # (2.1) obtain representation of explanatory variables
        h_x = self.tcm_src0(x_batch)
        for i in range(1, self.num_blocks_dst):
            h_x_res = h_x 
            h_x = F.leaky_relu((1-self.beta) * getattr(self, f'tcm_src{i}')(h_x) +  self.beta * h_x_res) 
        h_x = x_batch[...,-1:,:] + torch.tanh(self.decode_src(h_x))

        # (2.2) obtain representation of response variables 
        h_y = self.tcm_dst0(y_batch)
        for i in range(1, self.num_blocks_src):
            h_y_res = h_y 
            h_y = F.leaky_relu((1-self.beta) * getattr(self, f'tcm_dst{i}')(h_y) +  self.beta * h_y_res) 
        
        # (3) graph convolution
        for i in range(self.num_gcn_blocks): 
            # h_x, h_y = getattr(self, f'gcn{i}')(h_x, h_y, adj_mat)
            h_y = getattr(self, f'gcn{i}')(h_x, h_y, adj_mat)
            # h_x, h_y = F.leaky_relu(h_x), F.leaky_relu(h_y)
            h_y = F.leaky_relu(h_y) 
        # h_x, h_y =getattr(self, f'gcn{self.num_gcn_blocks-1}')(h_x, h_y, adj_mat)
        # h_y = getattr(self, f'gcn{self.num_gcn_blocks-1}')(h_x, h_y, adj_mat)
        # h_x, h_y = torch.tanh(h_x), torch.tanh(h_y)
        h_y = y_batch[..., -1:, :] + torch.tanh(self.decode_dst(h_y))

        # (4) relation 
        relation = gumbel_softmax(logits, self.tau, hard= True, dim= -1)

        return {
            'res_label': h_y,
            'exp_label': h_x, 
            'logits': logits,
            'adj_mat': adj_mat,
            'relation': relation
        }
    
    def train_step(self, batch, exp_loss_penalty= 0.1, kl_loss_penalty= 0.1):
        
        out = self.forward(batch)
        label_loss = self.criterion(out['exp_label'], batch['exp_label'])
        exp_loss = self.criterion(out['res_label'], batch['res_label'])
        probs = F.softmax(out['logits'], dim= -1) # bs, c, num_src, num_ts
        kl_loss = kl_categorical_uniform(probs, self.num_heteros * self.num_dst * self.num_src) 
        total_loss =  label_loss + exp_loss_penalty * exp_loss + kl_loss_penalty * kl_loss
        
        return {
            'total_loss': total_loss,
            'label_loss': label_loss,
            'exp_loss': exp_loss,
            'kl_loss': kl_loss    
        }

    @torch.no_grad()
    def val_step(self, batch, exp_loss_penalty= 0.1, kl_loss_penalty= 0.1): 
        
        out = self.forward(batch)
        label_loss = torch.mean((out['res_label'] - batch['res_label'])**2)
        exp_loss = torch.mean(out['exp_label']- batch['exp_label']**2)
        probs = F.softmax(out['logits'], dim= -1)
        kl_loss = kl_categorical_uniform(probs, self.num_dst*self.num_src) 
        total_loss =  label_loss + exp_loss_penalty * exp_loss + kl_loss_penalty * kl_loss

        return {
            'total_loss': total_loss,            
            'label_loss': label_loss,
            'exp_loss': exp_loss,
            'kl_loss': kl_loss    
        }

    @torch.no_grad()
    def test_step(self, batch):

        out = self.forward(batch)
        probs = F.softmax(out['logits'], dim= -1)
        kl_loss = kl_categorical_uniform(probs, self.num_dst*self.num_src) 

        res_preds, exp_preds = out['res_label'].detach().cpu().numpy(), out['exp_label'].detach().cpu().numpy()
        res_label, exp_label = batch['res_label'].detach().cpu().numpy(), batch['exp_label'].detach().cpu().numpy()

        # r2 score
        res_r2 = r2_score(res_label.flatten(), res_preds.flatten())
        exp_r2 = r2_score(exp_label.flatten(), exp_preds.flatten())

        # mse 
        res_mse = mean_squared_error(res_preds.flatten(), res_label.flatten())
        exp_mse = mean_squared_error(exp_preds.flatten(), exp_label.flatten())

        # mae
        res_mae = mean_absolute_error(res_preds.flatten(), res_label.flatten())
        exp_mae = mean_absolute_error(exp_preds.flatten(), exp_label.flatten())

        perf = {
            'r2_response': res_r2,
            'mse_response': res_mse,
            'mae_response': res_mae, 
            'r2_explanatory': exp_r2,
            'mse_explanatory': exp_mse,
            'mae_explanatory': exp_mae
        }
        return perf, out