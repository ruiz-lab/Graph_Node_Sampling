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

def Laplacian_Homophily(sample_idx, data):
    processed_X = data.x
    new_x = processed_X[sample_idx]
    edge_idx = pyg.utils.subgraph(sample_idx, data.edge_index, relabel_nodes=True)[0]
    sub_data = pyg.data.Data(x=new_x, edge_index=edge_idx)
    A = pyg.utils.to_dense_adj(edge_idx, max_num_nodes=sub_data.num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim = 1))
    L = D - A
    norm = torch.linalg.vector_norm(new_x, ord=2, dim=1)
    normed_new_x = new_x / torch.maximum(norm, torch.full_like(norm, 1e-6)).unsqueeze(1)
    numerator = torch.trace( -L @ normed_new_x @ normed_new_x.t() )
    denominator = edge_idx.shape[1]
    Homophily = numerator / denominator
    return Homophily


dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'TexasDataset', 'WisconsinDataset', 'CornellDataset', 'SquirrelDataset', 'ChameleonDataset']
# dataset_name_ls = ['SquirrelDataset']

sample_rate_ls = torch.linspace(0.1, 1, 10)
trace_type = "Laplacian"
exclusion_type = "Both"
dataset2homophily = {}


for dataset_name in dataset_name_ls:
    print("dataset_name: ", dataset_name)
    dataset = dgl.data.__getattribute__(dataset_name)()
    data = convert_dgl_to_pyg(dataset)
    if dataset_name == "PubmedGraphDataset":
        data = TFIDF2BOW_Feature(data)
    original_X = data.x
    X_norm_feature_wise = torch.linalg.norm(original_X, ord=2, dim=0)
    feature_normed_X = original_X / torch.maximum(X_norm_feature_wise, torch.full_like(X_norm_feature_wise, 1e-6))
    n, d = original_X.shape
    processed_X = feature_normed_X / (d)
    data.x = processed_X
    sampler = A_Opt_Sampler(data)
    # sampler = GraphNodeSampler(data)
    Homophily_ls = []
    Homophily_Exclude_Largest_ls = []
    Homophily_Exclude_Smallest_ls = []
    for sample_rate in sample_rate_ls:
        num_excluded = int(data.num_nodes * (1 - sample_rate))
        if trace_type == "Laplacian":
            if exclusion_type == "Largest":
                idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded, exclude_largest=True)
            elif exclusion_type == "Smallest":
                idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded, exclude_largest=False)
            elif exclusion_type == "Both":
                largest_exclusion_idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded, exclude_largest=True)
                smallest_exclusion_idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded, exclude_largest=False)
        elif trace_type == "Feature":
            if exclusion_type == "Largest":
                idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded, exclude_largest=True)
            elif exclusion_type == "Smallest":
                idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded, exclude_largest=False)
            elif exclusion_type == "Both":
                largest_exclusion_idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded, exclude_largest=True)
                smallest_exclusion_idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded, exclude_largest=False)
        if exclusion_type != "Both":
            sample_idx = torch.ones(data.num_nodes, dtype=bool)
            sample_idx[idx] = 0
            homophily = Laplacian_Homophily(sample_idx, data)
            Homophily_ls.append(homophily)
        elif exclusion_type == "Both":
            largest_sample_idx = torch.ones(data.num_nodes, dtype=bool)
            largest_sample_idx[largest_exclusion_idx] = 0
            smallest_sample_idx = torch.ones(data.num_nodes, dtype=bool)
            smallest_sample_idx[smallest_exclusion_idx] = 0
            homophily_exclude_largest = Laplacian_Homophily(largest_sample_idx, data)
            homophily_exclude_smallest = Laplacian_Homophily(smallest_sample_idx, data)
            Homophily_Exclude_Largest_ls.append(homophily_exclude_largest)
            Homophily_Exclude_Smallest_ls.append(homophily_exclude_smallest)
    if exclusion_type != "Both":
        dataset2homophily[dataset_name] = Homophily_ls
    elif exclusion_type == "Both":
        dataset2homophily[dataset_name] = {}
        dataset2homophily[dataset_name]["Exclude_Largest"] = Homophily_Exclude_Largest_ls
        dataset2homophily[dataset_name]["Exclude_Smallest"] = Homophily_Exclude_Smallest_ls
if exclusion_type != "Both":
    save_path = f"img/A_opt_{trace_type}/Laplacian_Homophily_exclude_smallest.png"
    # save_path = f"img/leverage/Laplacian_Homophily.png"
    plt.figure(figsize=(10, 5))
    for dataset_name in dataset_name_ls:
        Laplacian_Homophily_ls = dataset2homophily[dataset_name]
        print(Laplacian_Homophily_ls)
        plt.plot(sample_rate_ls, Laplacian_Homophily_ls, 'o-', label = dataset_name)
    plt.xlabel("Sample Rate")
    plt.ylabel("Laplacian Homophily")
    y_ticks = np.linspace(-1.2, 0, 7)
    plt.yticks(y_ticks)
    plt.legend()
    plt.title("Laplacian Homophily Score")
    plt.savefig(save_path, dpi=300)
elif exclusion_type == "Both":
    for dataset_name in dataset_name_ls:
        save_path = f"img/A_opt_{trace_type}/{dataset_name}_LapHom_Both.png"
        plt.plot(sample_rate_ls, dataset2homophily[dataset_name]["Exclude_Largest"], 'o-', c='red', label = f"Exclude_Largest")
        plt.plot(sample_rate_ls, dataset2homophily[dataset_name]["Exclude_Smallest"], 'o-', c='green', label = f"Exclude_Smallest")
        plt.xlabel("Sample Rate")
        plt.ylabel("Laplacian Homophily")
        y_ticks = np.linspace(-1.2, 0, 7)
        plt.yticks(y_ticks)
        plt.legend()
        plt.title(f"{dataset_name} Laplacian Homophily")
        plt.savefig(save_path, dpi=300)
        plt.close()