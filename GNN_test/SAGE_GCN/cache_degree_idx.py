import torch
import os
import numpy as np
import torch_geometric as pyg
from torch_geometric.datasets import OGB_MAG
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *


def cache_degree_idx(data):
    n = data.num_nodes
    degree_counter = torch.zeros(n, dtype=torch.long)
    for i, j in tqdm(zip(data.edge_index[0], data.edge_index[1]), desc='Counting degrees', total=data.edge_index.shape[1]):
        degree_counter[i] += 1
        degree_counter[j] += 1
    degree_idx = torch.argsort(degree_counter, descending=True)
    return degree_idx

if __name__ == "__main__":
    dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset']
    for dataset_name in dataset_name_ls:
        data = dataset_name2dataset(dataset_name)
        degree_idx = cache_degree_idx(data)
        torch.save(degree_idx, f'degree_idx_cache/{dataset_name}_degree_idx.pt')