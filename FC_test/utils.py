import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import dgl
import torch_geometric as pyg
from torch_geometric.data import Data
from torch.utils.data import Dataset

class FC_Dataset(nn.Module):
    def __init__(self, data, mask):
        super(FC_Dataset, self).__init__()
        self.x = data.x[mask]
        self.y = data.y[mask]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def convert_dgl_to_pyg(dgl_dataset):
    # Assuming the DGL dataset has only one graph for simplicity
    dgl_graph = dgl_dataset[0]

    # Convert edge indices from DGL to PyG format
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # Extract node features
    # Assuming 'feat' is the name of the node features in the DGL graph
    if 'feat' in dgl_graph.ndata:
        x = dgl_graph.ndata['feat']
    else:
        raise ValueError("Node features 'feat' not found in the DGL graph.")

    # Extract labels
    # Assuming 'label' is the name of the node labels in the DGL graph
    if 'label' in dgl_graph.ndata:
        y = dgl_graph.ndata['label']
    else:
        raise ValueError("Node labels 'label' not found in the DGL graph.")
    
    # Create the PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y, train_mask=dgl_graph.ndata['train_mask'], val_mask=dgl_graph.ndata['val_mask'], test_mask=dgl_graph.ndata['test_mask'], num_classes=torch.unique(y).shape[0], num_features=x.shape[1])

    return pyg_data

def generate_mask(num_nodes, sample_rate=[0.7, 0.1, 0.2]):
    mask = torch.randperm(num_nodes)
    train_rate, valid_rate, test_rate = sample_rate
    train_mask = mask[:int(train_rate*num_nodes)]
    valid_mask = mask[int(train_rate*num_nodes):int((train_rate+valid_rate)*num_nodes)]
    test_mask = mask[int((train_rate+valid_rate)*num_nodes):]
    return train_mask, valid_mask, test_mask