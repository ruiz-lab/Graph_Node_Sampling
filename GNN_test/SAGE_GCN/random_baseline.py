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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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


dataset_name_ls = ['cora', 'citeseer', 'pubmed']
project_name_format = 'SAGE_GCN_hyperparameter_tuning_{}_icassp_undirected'
# the number of best runs to be baselined
top_k = 5

for dataset_name in dataset_name_ls:
    crnt_project_name = project_name_format.format(dataset_name)
    rst_df = project_name2rst_df(crnt_project_name)
    test_acc_sorted_rst_df = rst_df.sort_values(by='best_test_acc', ascending=False)[:top_k]
    max_test_acc_ls = []
    min_test_acc_ls = []
    mean_test_acc_ls = []
    std_test_acc_ls = []
    box_plot_test_acc_dict = {}
    for _ in range(top_k):
        run_config = test_acc_sorted_rst_df.iloc[_]
        test_acc_rst_ls = random_baseline(run_config, runtimes=100, k=_)
        max_test_acc = max(test_acc_rst_ls)
        min_test_acc = min(test_acc_rst_ls)
        mean_test_acc = np.mean(test_acc_rst_ls)
        std_test_acc = np.std(test_acc_rst_ls)
        max_test_acc_ls.append(max_test_acc)
        min_test_acc_ls.append(min_test_acc)
        mean_test_acc_ls.append(mean_test_acc)
        std_test_acc_ls.append(std_test_acc)
        box_plot_test_acc_dict[_] = test_acc_rst_ls
        title = f"{dataset_name} random baseline {_+1}"
        plt.figure()
        sns.histplot(test_acc_rst_ls, kde=True)
        plt.title(title)
        plt.xlabel('test accuracy')
        plt.ylabel('count')
        plot_path = f"../img/random_baseline/{dataset_name}_undirected_random_baseline_{_+1}.png"
        plt.savefig(plot_path)
        plt.close()
    sns.boxplot(data=[box_plot_test_acc_dict[_] for _ in range(top_k)], width=0.5, showmeans=True)
    for _ in range(top_k):
        plt.scatter(_, test_acc_sorted_rst_df.iloc[_]['best_test_acc'], color='red', s=100, label='Sampling Acc')
    plt.title(f"{dataset_name} random baseline")
    plt.xlabel('Different Configurations')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plot_path = f"../img/random_baseline/{dataset_name}_undirected_random_baseline_boxplot.png"
    plt.savefig(plot_path)
    plt.close()
    test_acc_sorted_rst_df['max_test_acc'] = max_test_acc_ls
    test_acc_sorted_rst_df['min_test_acc'] = min_test_acc_ls
    test_acc_sorted_rst_df['mean_test_acc'] = mean_test_acc_ls
    test_acc_sorted_rst_df['std_test_acc'] = std_test_acc_ls
    csv_path = f"random_baseline/{dataset_name}_undirected_random_baseline.csv"
    test_acc_sorted_rst_df.to_csv(csv_path, index=False)
    
        