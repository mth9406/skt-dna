import torch 
from torch.nn import functional as F
from torch import nn 
from torch.autograd import Variable

from layers.graphLearningLayers import *

def nri_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return nri_softmax(y / tau, axis=-1)

def nri_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def generate_off_diag(num_objects, device= None): 
    r"""Generates off-diagonal graph 
    
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
    off_diag = np.ones([num_objects, num_objects]) - np.eye(num_objects)

    rec, send = np.where(off_diag)
    rel_rec = np.array(encode_onehot(rec), dtype= np.float32)
    rel_send = np.array(encode_onehot(send), dtype= np.float32)
    
    rel_rec = torch.FloatTensor(rel_rec) if device is None \
        else torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send) if device is None \
        else torch.FloatTensor(rel_send).to(device)
    
    return rel_rec, rel_send  

def coo_to_adj(rel, num_objects, device= None):
    r"""Converts COO-type to adjacency matrix
    
    # Arguments          
    ___________ 
    rel : torch.FloatTensor 
        relation - tensor                  
    num_objects : int
        the number of objects               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    adj : torch.FloatTensor
        adjacency matrix  
        a square matrix of shape: num_object x num_object 
    """

    adj = torch.zeros((num_objects*num_objects))
    for i in range(num_objects-1):
        adj[i*num_objects+i+1:(i+1)*num_objects+(i+1)] = rel[i*num_objects:(i+1)*num_objects]
    if device is not None:
        adj = adj.to(device)
    return adj.reshape((num_objects,num_objects))

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out=2, do_prob=0.,):
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        x = self.mlp3(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)

        return self.fc_out(x)

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.msg_fc1 = (nn.Linear(2 * n_in_node, msg_hid))
        self.msg_fc2 = (nn.Linear(msg_hid, msg_out))
        self.msg_out_shape = msg_out

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        # print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        msg = F.relu(self.msg_fc1(pre_msg))
        msg = F.dropout(msg, p=self.dropout_prob)
        msg = F.relu(self.msg_fc2(msg))
        msg = msg * single_timestep_rel_type[:, :, :, 1:2]

        # Aggregate all msgs to receiver
        agg_msgs = msg.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = torch.tanh(self.out_fc3(pred))

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # initial step
        last_pred = inputs[:, 0:1, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        curr_rel_type = rel_type[:, 0:1, :, :]

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        #output = Variable(torch.zeros(sizes))
        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i:i+1, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()
