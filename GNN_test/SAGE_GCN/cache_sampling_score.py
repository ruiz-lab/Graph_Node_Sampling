import torch
import os
import numpy as np
import torch_geometric as pyg
from torch_geometric.datasets import OGB_MAG
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *


def batch_range_loader(num_nodes, batch_size):
    idx_ls = list(range(0, num_nodes, batch_size))
    if num_nodes % batch_size != 0:
        idx_ls.append(num_nodes)
    return iter([(idx_ls[i], idx_ls[i+1]) for i in range(len(idx_ls)-1)])

def cache_Laplacian_score(data, batch_size=1000, dataset_name=None):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    out_edge_dict = {} # key cites value
    in_edge_dict = {} # key cited by value
    score_path = f"cache/{dataset_name}_Laplacian_score.pt"
    indices_path = f"cache/{dataset_name}_Laplacian_score_descending_indices.pt"
    for k in tqdm(range(num_nodes), desc=f'Initializing {dataset_name} edge dict', total=num_nodes):
        out_edge_dict[k] = []
        in_edge_dict[k] = []
    for i in tqdm(range(num_nodes), desc=f'Building {dataset_name} edge dict', total=num_nodes):
        src, dst = edge_index[0][i].item(), edge_index[1][i].item()
        out_edge_dict[src].append(dst)
        in_edge_dict[dst].append(src)
    
    idx_range_iter = batch_range_loader(num_nodes, batch_size)
    Laplacian_Score = torch.zeros(num_nodes, dtype=torch.float32)
    for (start, end) in tqdm(idx_range_iter, desc=f'Computing {dataset_name} Laplacian Score', total=num_nodes//batch_size + 1):
        crnt_batch_size = end - start
        partial_out_A = torch.zeros(crnt_batch_size, num_nodes, dtype=torch.float32)
        partial_in_A = torch.zeros(num_nodes, crnt_batch_size, dtype=torch.float32)
        for i in range(crnt_batch_size):
            for j in out_edge_dict[start+i]:
                partial_out_A[i][j] = 1
            for j in in_edge_dict[start+i]:
                partial_in_A[j][i] = 1
        crnt_batch_score = torch.sum(partial_out_A * partial_in_A.t(), dim=1)
        Laplacian_Score[start:end] = crnt_batch_score
    Laplacian_Score, descending_indices = torch.sort(Laplacian_Score, descending=True)
    torch.save(Laplacian_Score, score_path)
    torch.save(descending_indices, indices_path)

def cache_Feature_score(data, batch_size=1000, dataset_name=None):
    idx_range_iter = batch_range_loader(data.num_nodes, batch_size)
    Feature_Score = torch.zeros(data.num_nodes, dtype=torch.float32)
    score_path = f"cache/{dataset_name}_Feature_score.pt"
    indices_path = f"cache/{dataset_name}_Feature_score_descending_indices.pt"
    for (start, end) in tqdm(idx_range_iter, desc=f'Computing {dataset_name} Feature Score', total=data.num_nodes//batch_size + 1):
        crnt_batch_size = end - start
        partial_X = data.x[start:end] # (batch_size, num_features)
        partial_rst = torch.matmul(partial_X, data.x.t()) # (batch_size, num_nodes)
        Feature_Score[start:end] = torch.sum(partial_rst ** 2, dim=1)
    Feature_Score, descending_indices = torch.sort(Feature_Score, descending=True)
    torch.save(Feature_Score, score_path)
    torch.save(descending_indices, indices_path)
    
if __name__ == '__main__':

    dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'OGB_MAG']
    batch_size = 2000

    for dataset_name in dataset_name_ls:
        if dataset_name != 'OGB_MAG':
            data = dataset_name2dataset(dataset_name)
            cache_Laplacian_score(data, batch_size, dataset_name)
        else:
            data = load_ogbn_mag()
            cache_Laplacian_score(data, batch_size, dataset_name)
    
    for dataset_name in dataset_name_ls:
        if dataset_name != 'OGB_MAG':
            data = dataset_name2dataset(dataset_name)
            cache_Feature_score(data, batch_size, dataset_name)
        else:
            data = load_ogbn_mag()
            cache_Feature_score(data, batch_size, dataset_name)