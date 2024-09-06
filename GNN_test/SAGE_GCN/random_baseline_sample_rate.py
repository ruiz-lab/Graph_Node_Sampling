import os
import yaml
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch_geometric as pyg
import dgl
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *
from model import Modified_GCN, Modified_SAGE 
from train import validation, test

import matplotlib.pyplot as plt
import seaborn as sns

def random_baseline(run_config, runtimes=100, k=None):
    model_type = run_config['model_type']
    hidden_dim = int(run_config['hidden_dim'])
    dataset_name = run_config['dataset_name']
    num_epochs = int(run_config['num_epochs'])
    sample_rate = run_config['sample_rate']
    weight_decay = run_config['weight_decay']
    learning_rate = run_config['learning_rate']
    activation_type = run_config['activation_type']
    num_hidden_layers = int(run_config['num_hidden_layers'])

    # load dataset
    data = dataset_name2dataset(dataset_name)
    
    # load model
    input_dim = data.num_features
    output_dim = data.num_classes
    if model_type == 'GCN':
        model = Modified_GCN(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)
    elif model_type == 'SAGE':
        model = Modified_SAGE(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)

    # load optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # load loss function
    criterion = nn.NLLLoss()

    # load device
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # load sampler
    sampler = RandomSampler()
    sampled_data = sampler.sample(data, sample_rate=sample_rate).to(device)

    # training
    test_acc_rst_ls = []
    for i in tqdm(range(runtimes), desc=f'{dataset_name} Random Baseline at {k}', total=runtimes):
        crnt_test_acc_ls = []
        crnt_valid_acc_ls = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(sampled_data)
            loss = criterion(out[sampled_data.train_mask], sampled_data.y[sampled_data.train_mask])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                test_acc = test(model, sampled_data)
                valid_acc = validation(model, sampled_data)
                crnt_test_acc_ls.append(test_acc)
                crnt_valid_acc_ls.append(valid_acc)
        best_valid_acc_idx = np.argmax(crnt_valid_acc_ls)
        best_test_acc = crnt_test_acc_ls[best_valid_acc_idx]
        test_acc_rst_ls.append(best_test_acc)

    return test_acc_rst_ls


dataset_name_ls = ["CoraGraphDataset", "CiteseerGraphDataset", "PubmedGraphDataset"]
stability_check_config_path = 'stability_check/{}_{}_undirected.csv'
sample_rate_ls = np.linspace(0.125, 1, 8)
# sample_rate_ls = [0.125]


for dataset_name in dataset_name_ls:
    boxplot_random_test_acc_ls = []
    biggest_diff_idx = []
    for sample_rate in sample_rate_ls:
        config_df = pd.read_csv(stability_check_config_path.format(dataset_name, sample_rate))
        num_configs = config_df.shape[0]
        acc_diff = []
        random_test_acc_rst = []
        for i in range(num_configs):
            run_config = config_df.iloc[i]
            crnt_best_acc = run_config['mean']
            test_acc_rst_ls = random_baseline(run_config, runtimes=100, k=i)
            test_acc_rst_df = pd.DataFrame({'test_acc': test_acc_rst_ls})
            random_mean = test_acc_rst_df['test_acc'].mean()
            acc_diff.append(crnt_best_acc - random_mean)
            random_test_acc_rst.append(test_acc_rst_ls)
        idx = np.argmax(acc_diff)
        biggest_diff_idx.append(idx)
        boxplot_random_test_acc_ls.append(random_test_acc_rst[idx])
    title = f"{dataset_name}"
    plt.figure()
    sns.boxplot(data=boxplot_random_test_acc_ls, width=0.5, showmeans=True)
    for _, sample_rate in enumerate(sample_rate_ls):
        if _ == 0:
            label = 'Sampling Acc'
        else:
            label = None
        crnt_df = pd.read_csv(stability_check_config_path.format(dataset_name, sample_rate))
        plt.scatter(_, crnt_df.iloc[biggest_diff_idx[_]]['mean'], color='red', s=30, label=label, zorder=10)
    plt.title(title)
    plt.xticks(range(len(sample_rate_ls)), sample_rate_ls)
    plt.xlabel('Sample Rate')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plot_path = f"../img/random_baseline/{dataset_name}_undirected_random_baseline_boxplot.png"
    plt.savefig(plot_path)
    plt.close()


# for dataset_name in dataset_name_ls:
#     crnt_project_name = project_name_format.format(dataset_name)
#     rst_df = project_name2rst_df(crnt_project_name)
#     test_acc_sorted_rst_df = rst_df.sort_values(by='best_test_acc', ascending=False)[:top_k]
#     max_test_acc_ls = []
#     min_test_acc_ls = []
#     mean_test_acc_ls = []
#     std_test_acc_ls = []
#     box_plot_test_acc_dict = {}
#     for _ in range(top_k):
#         run_config = test_acc_sorted_rst_df.iloc[_]
#         test_acc_rst_ls = random_baseline(run_config, runtimes=100, k=_)
#         max_test_acc = max(test_acc_rst_ls)
#         min_test_acc = min(test_acc_rst_ls)
#         mean_test_acc = np.mean(test_acc_rst_ls)
#         std_test_acc = np.std(test_acc_rst_ls)
#         max_test_acc_ls.append(max_test_acc)
#         min_test_acc_ls.append(min_test_acc)
#         mean_test_acc_ls.append(mean_test_acc)
#         std_test_acc_ls.append(std_test_acc)
#         box_plot_test_acc_dict[_] = test_acc_rst_ls
#         title = f"{dataset_name} random baseline {_+1}"
#         plt.figure()
#         sns.histplot(test_acc_rst_ls, kde=True)
#         plt.title(title)
#         plt.xlabel('test accuracy')
#         plt.ylabel('count')
#         plot_path = f"../img/random_baseline/{dataset_name}_undirected_random_baseline_{_+1}.png"
#         plt.savefig(plot_path)
#         plt.close()
#     sns.boxplot(data=[box_plot_test_acc_dict[_] for _ in range(top_k)], width=0.5, showmeans=True)
#     for _ in range(top_k):
#         plt.scatter(_, test_acc_sorted_rst_df.iloc[_]['best_test_acc'], color='red', s=100, label='Sampling Acc')
#     plt.title(f"{dataset_name} random baseline")
#     plt.xlabel('Different Configurations')
#     plt.ylabel('Test Accuracy')
#     plt.legend()
#     plot_path = f"../img/random_baseline/{dataset_name}_undirected_random_baseline_boxplot.png"
#     plt.savefig(plot_path)
#     plt.close()
#     test_acc_sorted_rst_df['max_test_acc'] = max_test_acc_ls
#     test_acc_sorted_rst_df['min_test_acc'] = min_test_acc_ls
#     test_acc_sorted_rst_df['mean_test_acc'] = mean_test_acc_ls
#     test_acc_sorted_rst_df['std_test_acc'] = std_test_acc_ls
#     csv_path = f"random_baseline/{dataset_name}_undirected_random_baseline.csv"
#     test_acc_sorted_rst_df.to_csv(csv_path, index=False)
    
        