import torch
import os
from torch_geometric.datasets import OGB_MAG


def cache_Laplacian_score(path):
    dataset = OGB_MAG(root='../data')
    data = dataset[0]
    edge_index = data.edge_index
    degree = torch.zeros(data.num_nodes, dtype=torch.int64)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.int64))
    degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.int64))

    