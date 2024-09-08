import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from utils import *
from tqdm import tqdm
import yaml
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def sparse_trace_rank_estimation(data, rank_threshold=1e-10):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    trace_start = time.time()
    A = sp.coo_matrix((np.ones(num_edges), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    D = sp.diags(A.sum(axis=0).A1)
    L = D - A
    L = L.tocsr()
    trace = np.sum(L.diagonal())
    trace_end = time.time()
    print(f"Trace calculation takes {trace_end - trace_start} seconds, num_nodes: {num_nodes}, num_edges: {num_edges}")
    # # rank estimation using SVD
    # rank_start = time.time()
    # _, S, _ = spla.svds(L, k=num_nodes - 1)
    # rank = np.sum(S > rank_threshold)
    # rank_end = time.time()
    # print(f"Rank calculation takes {rank_end - rank_start} seconds")
    rank = None
    return trace/num_nodes, rank


dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset']
trace_type = 'Feature'
exclusion_type = 'Largest'
sample_rate_ls = np.linspace(0.125, 1, 8)
trace_rst = {}
rank_rst = {}
num_random_trails = 100

for dataset_name in dataset_name_ls:
    data = dataset_name2dataset(dataset_name)
    trace_rst[dataset_name] = {}
    trace_rst[dataset_name]["heuristic"] = {}
    trace_rst[dataset_name]["random"] = {}
    trace_rst[dataset_name]["degree"] = {}
    rank_rst[dataset_name] = {}
    rank_rst[dataset_name]["heuristic"] = {}
    rank_rst[dataset_name]["random"] = {}
    rank_rst[dataset_name]["degree"] = {}
    cached_indices_path = f"SAGE_GCN/cache/{dataset_name}_{trace_type}_score_descending_indices.pt"
    heuristic_sampler = Cached_Sampler(dataset_name, trace_type, exclusion_type, cached_indices_path)
    random_sampler = RandomSampler()
    degree_sampler = Cached_Degree_Sampler(dataset_name, "SAGE_GCN/degree_idx_cache/")
    for sample_rate in sample_rate_ls:
        trace_rst[dataset_name]["heuristic"][sample_rate] = []
        trace_rst[dataset_name]["random"][sample_rate] = []
        trace_rst[dataset_name]["degree"][sample_rate] = []
        rank_rst[dataset_name]["heuristic"][sample_rate] = []
        rank_rst[dataset_name]["random"][sample_rate] = []
        rank_rst[dataset_name]["degree"][sample_rate] = []
        heuristic_sampled_data = heuristic_sampler.sample(data, sample_rate=sample_rate)
        heuristic_trace, heuristic_rank = sparse_trace_rank_estimation(heuristic_sampled_data)
        trace_rst[dataset_name]["heuristic"][sample_rate].append(heuristic_trace)
        rank_rst[dataset_name]["heuristic"][sample_rate].append(heuristic_rank)
        degree_sampled_data = degree_sampler.sample(data, sample_rate=sample_rate)
        degree_trace, degree_rank = sparse_trace_rank_estimation(degree_sampled_data)
        trace_rst[dataset_name]["degree"][sample_rate].append(degree_trace)
        for i in tqdm(range(num_random_trails), desc=f'{dataset_name} Random Baseline at {sample_rate}', total=num_random_trails):
            random_sampled_data = random_sampler.sample(data, sample_rate=sample_rate)
            random_trace, random_rank = sparse_trace_rank_estimation(random_sampled_data)
            trace_rst[dataset_name]["random"][sample_rate].append(random_trace)
            rank_rst[dataset_name]["random"][sample_rate].append(random_rank)

    # dump the results
    path = f"./trace_rank_vs_sample_rate/{dataset_name}_trace_rank.yaml"
    with open(path, 'w') as f:
        yaml.dump({"trace": trace_rst[dataset_name], "rank": rank_rst[dataset_name]}, f)

# boxplot for trace
for dataset_name in dataset_name_ls:
    # boxplot random results
    sns.boxplot(data=[trace_rst[dataset_name]["random"][sample_rate] for sample_rate in sample_rate_ls], color='skyblue', width=0.5)
    # scatter heuristic results
    for _ in range(len(sample_rate_ls)):
        if _ == 0:
            label = 'Feature Heuristic'
            degree_label = 'Degree Heuristic'
        else:
            label = None
            degree_label = None
        plt.scatter(_, trace_rst[dataset_name]["heuristic"][sample_rate_ls[_]], color='red', s=10, label=label, zorder=10)
        plt.scatter(_, trace_rst[dataset_name]["degree"][sample_rate_ls[_]], color='blue', s=50, label=degree_label, zorder=9)
    plt.title(f"{dataset_name} trace estimation")
    plt.legend()
    plt.xlabel('Sample Rate')
    plt.xticks(range(len(sample_rate_ls)), [f"{sample_rate * 100:.1f}%" for sample_rate in sample_rate_ls])
    plt.ylabel('Trace')
    save_path = f"./img/trace_rank_vs_sample_rate/{dataset_name}_trace_boxplot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

# boxplot for rank
# for dataset_name in dataset_name_ls:
#     # boxplot random results
#     sns.boxplot(data=[rank_rst[dataset_name]["random"][_] for _ in range(len(sample_rate_ls))], color='skyblue', width=0.5, showmeans=True)
#     # scatter heuristic results
#     for _ in range(len(sample_rate_ls)):
#         plt.scatter(_, rank_rst[dataset_name]["heuristic"][_], color='red', s=100, label='Feature Largest')
#     plt.title(f"{dataset_name} rank estimation")
#     plt.xlabel('Sample Rate')
#     plt.xticks(range(len(sample_rate_ls)), [f"{sample_rate * 100:.1f}%" for sample_rate in sample_rate_ls])
#     plt.ylabel('Rank')
#     plt.legend()
#     save_path = f"./img/trace_rank_vs_sample_rate/{dataset_name}_rank_boxplot.png"
#     plt.savefig(save_path, dpi=300)
