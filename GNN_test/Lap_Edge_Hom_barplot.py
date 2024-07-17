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
from Laplacian_homophily import Laplacian_Homophily
from utils import *
plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 18  # Title font size
plt.rcParams['axes.labelsize'] = 16  # X and Y label font size
plt.rcParams['xtick.labelsize'] = 12  # X tick label font size
plt.rcParams['ytick.labelsize'] = 12  # Y tick label font size
plt.rcParams['legend.fontsize'] = 14  # Legend font size
# plt.rcParams['text.usetex'] = True


def homophily_vs_edge_homophily_barplot(dataset_name_ls, sample_rate_ls, trace_type, exclusion_type):
    dataset2homophily = {}
    dataset2edge_homophily = {}

    # initialize the dictionary
    for dataset_name in dataset_name_ls:
        dataset2homophily[dataset_name] = {}
        for samplerate in sample_rate_ls:
            if exclusion_type != "Both":
                dataset2homophily[dataset_name] = []
            elif exclusion_type == "Both":
                dataset2homophily[dataset_name] = {"Exclude_Largest": [], "Exclude_Smallest": []}

    for dataset_name in dataset_name_ls:
        if exclusion_type != "Both":
            dataset2edge_homophily[dataset_name] = []
        elif exclusion_type == "Both":
            dataset2edge_homophily[dataset_name] = {"Exclude_Largest": [], "Exclude_Smallest": []}

    for dataset_name in dataset_name_ls:
        dataset = dgl.data.__getattribute__(dataset_name)()
        data = convert_dgl_to_pyg(dataset)
        if dataset_name == "PubmedGraphDataset":
            data = TFIDF2BOW_Feature(data)
        original_X = data.x
        X_norm_feature_wise = torch.linalg.norm(original_X, ord=2, dim=0)
        vector_norm = torch.linalg.norm(original_X, ord=2, dim=1)
        feature_normed_X = original_X / torch.maximum(X_norm_feature_wise, torch.full_like(X_norm_feature_wise, 1e-6))
        processed_X = feature_normed_X / torch.maximum(vector_norm, torch.full_like(vector_norm, 1e-6)).unsqueeze(1)
        n, d = original_X.shape
        processed_X /= d
        data.x = processed_X
        sampler = A_Opt_Sampler(data)
        Homophily_ls = []
        Edge_homphily_ls = []
        Homophily_Exclude_Largest_ls = []
        Homophily_Exclude_Smallest_ls = []
        Edge_homophily_Exclude_Largest_ls = []
        Edge_homophily_Exclude_Smallest_ls = []
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
                sub_data = subgraph_from_index(sample_idx, data)
                homophily = Laplacian_Homophily(sub_data)
                Homophily_ls.append(homophily)
                Edge_homphily_ls.append(edge_homophily_score(sub_data))
            elif exclusion_type == "Both":
                largest_sample_idx = torch.ones(data.num_nodes, dtype=bool)
                largest_sample_idx[largest_exclusion_idx] = 0
                smallest_sample_idx = torch.ones(data.num_nodes, dtype=bool)
                smallest_sample_idx[smallest_exclusion_idx] = 0
                largest_exclusion_sub_data = subgraph_from_index(largest_sample_idx, data)
                smallest_exclusion_sub_data = subgraph_from_index(smallest_sample_idx, data)
                homophily_exclude_largest = Laplacian_Homophily(largest_exclusion_sub_data)
                homophily_exclude_smallest = Laplacian_Homophily(smallest_exclusion_sub_data)
                Homophily_Exclude_Largest_ls.append(homophily_exclude_largest)
                Homophily_Exclude_Smallest_ls.append(homophily_exclude_smallest)
                Edge_homophily_Exclude_Largest_ls.append(edge_homophily_score(largest_exclusion_sub_data))
                Edge_homophily_Exclude_Smallest_ls.append(edge_homophily_score(smallest_exclusion_sub_data))
        if exclusion_type != "Both":
            dataset2homophily[dataset_name] = Homophily_ls
            dataset2edge_homophily[dataset_name] = Edge_homphily_ls
        elif exclusion_type == "Both":
            dataset2homophily[dataset_name]["Exclude_Largest"] = Homophily_Exclude_Largest_ls
            dataset2homophily[dataset_name]["Exclude_Smallest"] = Homophily_Exclude_Smallest_ls
            dataset2edge_homophily[dataset_name]["Exclude_Largest"] = Edge_homophily_Exclude_Largest_ls
            dataset2edge_homophily[dataset_name]["Exclude_Smallest"] = Edge_homophily_Exclude_Smallest_ls

    # convert the coordinates of the dictionary
    homophily_rst = np.zeros((len(sample_rate_ls), len(dataset_name_ls)))
    edge_homophily_rst = np.zeros((len(sample_rate_ls), len(dataset_name_ls)))
    for i, dataset_name in enumerate(dataset_name_ls):
        homophily_rst[:, i] = dataset2homophily[dataset_name]
        edge_homophily_rst[:, i] = dataset2edge_homophily[dataset_name]

    homophily_multiplier = -20
    homophily_rst *= homophily_multiplier
    sample_rate2homophily = {}
    sample_rate2edge_homophily = {}
    for i, sample_rate in enumerate(sample_rate_ls):
        sample_rate2homophily[sample_rate] = homophily_rst[i]
        sample_rate2edge_homophily[sample_rate] = edge_homophily_rst[i]
    # plotting
    if exclusion_type != "Both":
        bar_width = 0.35
        x = np.arange(len(dataset_name_ls))
        x1 = x - bar_width/2
        x2 = x + bar_width/2
        fig, axs = plt.subplots(len(sample_rate_ls), 1, figsize=(10, 20))
        for i, sample_rate in enumerate(sample_rate_ls):
            axs[i].bar(x1, sample_rate2homophily[sample_rate], bar_width, label=r"homophily x {}".format(homophily_multiplier))
            axs[i].bar(x2, sample_rate2edge_homophily[sample_rate], bar_width, label=r"edge homophily")
            axs[i].set_xlabel("Dataset")
            axs[i].set_title(f"Sample Rate: {round(sample_rate, 1)}")
            axs[i].set_xticks(x)
            axs[i].set_ylim(0, 1)
            axs[i].set_xticklabels([dataset_name.replace("Dataset", "").replace("Graph", "") for dataset_name in dataset_name_ls])
            axs[i].legend()
        fig.tight_layout()
        path = f"img/A_opt_{trace_type}/{exclusion_type}_Exclusion_{trace_type}_Homophily_Edge_Homophily.png"
        fig.savefig(path)

    return 
        
if __name__ == "__main__":
    dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'TexasDataset', 'WisconsinDataset', 'CornellDataset', 'SquirrelDataset', 'ChameleonDataset']
    sample_rate_ls = np.linspace(0.2, 1, 5)

    trace_type_ls = ["Laplacian", "Feature"]
    exclusion_type_ls = ["Largest", "Smallest"]
    for trace_type in trace_type_ls:
        for exclusion_type in exclusion_type_ls:
            print(f"trace_type: {trace_type}, exclusion_type: {exclusion_type} ...")
            homophily_vs_edge_homophily_barplot(dataset_name_ls, sample_rate_ls, trace_type, exclusion_type)
