import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
import os
import matplotlib.pyplot as plt
import dgl
from torch_geometric.datasets import Planetoid
from scipy.sparse.linalg import eigs, eigsh
import seaborn as sns
from tqdm import tqdm
from utils import *
from model import *

dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'TexasDataset', 'WisconsinDataset', 'CornellDataset', 'SquirrelDataset', 'ChameleonDataset']


sample_rate_ls = torch.linspace(0.1, 1, 10)
trace_type = "Laplacian"
dataset2homophily = {}


for dataset_name in dataset_name_ls:
    dataset = dgl.data.__getattribute__(dataset_name)()
    data = convert_dgl_to_pyg(dataset)
    sampler = A_Opt_Sampler(data)
    Laplacian_Homophily_ls = []
    for sample_rate in sample_rate_ls:
        num_excluded = int(data.num_nodes * (1 - sample_rate))
        if trace_type == "Laplacian":
            idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded)
        elif trace_type == "Feature":
            idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded)
        sample_idx = torch.ones(data.num_nodes, dtype=bool)
        sample_idx[idx] = 0
        new_edge_idx = pyg.utils.subgraph(sample_idx, data.edge_index, relabel_nodes=False)[0]
        new_x = data.x
        new_x[idx] = 0
        A = pyg.utils.to_dense_adj(data.edge_index).squeeze()
        A[idx, :] = 0
        A[:, idx] = 0
        D = torch.diag(torch.sum(A, dim = 1))
        L = D - A
        norm = torch.linalg.vector_norm(new_x, ord=2, dim=1)
        normed_new_x = new_x / torch.maximum(norm, torch.full_like(norm, 1e-6)).unsqueeze(1)
        Laplacian_Homophily = torch.trace( -L @ normed_new_x @ normed_new_x.t() ) / new_edge_idx.shape[1]
        Laplacian_Homophily_ls.append(Laplacian_Homophily)
    dataset2homophily[dataset_name] = Laplacian_Homophily_ls
print(sample_rate_ls)
save_path = f"img/A_opt_{trace_type}/Laplacian_Homophily.png"
plt.figure(figsize=(10, 5))
for dataset_name in dataset_name_ls:
    print(len(Laplacian_Homophily_ls), Laplacian_Homophily_ls[-1])
    Laplacian_Homophily_ls = dataset2homophily[dataset_name]
    plt.plot(sample_rate_ls, Laplacian_Homophily_ls, 'o-', label = dataset_name)
plt.xlabel("Sample Rate")
plt.ylabel("Laplacian Homophily")
plt.legend()
plt.title("Laplacian Homophily Score")
plt.savefig(save_path, dpi=300)