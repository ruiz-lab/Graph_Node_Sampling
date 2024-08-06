import os
import yaml
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data
import networkx as nx
import dgl
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *
from model import Modified_GCN, Modified_SAGE, Graph_Dataset
from train import validation, test
import seaborn as sns
import matplotlib.pyplot as plt

def train(config_dict, run_times=100):

    if torch.cuda.is_available():
        device = torch.device('cuda:1')

    if config_dict["dataset_name"] != 'OGB_MAG':
        data = dataset_name2dataset(config_dict["dataset_name"]).to(device)
    else:
        data = load_ogbn_mag().to(device)
    
    input_dim = data.num_features
    hidden_dim = config_dict["hidden_dim"]
    output_dim = data.num_classes
    model_type = config_dict["model_type"]
    activation_type = config_dict["activation_type"]
    num_hidden_layers = config_dict["num_hidden_layers"]

    if model_type == 'GCN':
        model = Modified_GCN(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)
    elif model_type == 'SAGE':
        model = Modified_SAGE(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)
    model = model.to(device)

    learning_rate = config_dict["learning_rate"]
    weight_decay = config_dict["weight_decay"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.NLLLoss()

    num_epochs = config_dict["num_epochs"]

    test_acc_ls = []
    for run in tqdm(range(run_times), desc=""):
        best_epoch_test_acc = 0
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                test_acc = test(model, data)
                if test_acc > best_epoch_test_acc:
                    best_epoch_test_acc = test_acc
        test_acc_ls.append(best_epoch_test_acc)
    
    return test_acc_ls


if __name__ == "__main__":
    trace_type_ls = ['Laplacian', 'Feature']
    exclusion_type_ls = ['Largest', 'Random']
    for trace_type in trace_type_ls:
        for exclusion_type in exclusion_type_ls:
            hyp_config_path = f"hyp_param_rst/top5_hyper_param_{trace_type}_{exclusion_type}.csv"
            config_df = pd.read_csv(hyp_config_path)
            mean_array = np.zeros(len(config_df))
            std_array = np.zeros(len(config_df))
            for config in range(len(config_df)):
                hidden_dim = config_df.iloc[config]["hidden_dim"]
                model_type = config_df.iloc[config]["model_type"]
                num_epochs = config_df.iloc[config]["num_epochs"]
                sample_rate = config_df.iloc[config]["sample_rate"]
                dataset_name = config_df.iloc[config]["dataset_name"]
                weight_decay = config_df.iloc[config]["weight_decay"]
                learning_rate = config_df.iloc[config]["learning_rate"]
                activation_type = config_df.iloc[config]["activation_type"]
                num_hidden_layers = config_df.iloc[config]["num_hidden_layers"]

                config_dict = {
                    "hidden_dim": hidden_dim,
                    "model_type": model_type,
                    "num_epochs": num_epochs,
                    "sample_rate": sample_rate,
                    "dataset_name": dataset_name,
                    "weight_decay": weight_decay,
                    "learning_rate": learning_rate,
                    "activation_type": activation_type,
                    "num_hidden_layers": num_hidden_layers,
                    "trace_type": trace_type,
                    "exclusion_type": exclusion_type
                }

                test_acc_ls = train(config_dict, run_times=100)
                test_acc_mean = np.mean(test_acc_ls)
                test_acc_std = np.std(test_acc_ls)
                mean_array[config] = test_acc_mean
                std_array[config] = test_acc_std

                plt.figure(figsize=(5,3), dpi=300)
                test_acc_hist = sns.histplot(test_acc_ls, kde=True)
                plt.xlabel("Test Accuracy")
                plt.title(f"{dataset_name}_{trace_type}_{exclusion_type}.png")
                values = list(config_dict.values())
                values = [str(v) for v in values]
                name = "_".join(values)
                plot_save_path = f"../img/stability_check/{name}.png"
                plt.savefig(plot_save_path)
                plt.close()

            config_df["mean"] = mean_array
            config_df["std"] = std_array
            save_path = f"stability_check/{trace_type}_{exclusion_type}.csv"
            config_df.to_csv(save_path, index=False)



