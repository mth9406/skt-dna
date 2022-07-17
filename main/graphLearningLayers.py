import torch  
from torch import nn 
import numpy as np 

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
