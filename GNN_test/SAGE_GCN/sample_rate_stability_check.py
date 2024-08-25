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
from tuning_stability_check import train

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
    dataset_name = config_dict["dataset_name"]

    sampler = Cached_Sampler(dataset_name, "Feature", "Largest")
    sample_rate = config_dict["sample_rate"]
    sampled_data = sampler.sample(data=data, sample_rate=sample_rate)

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
            output = model(sampled_data)
            loss = criterion(output[sampled_data.train_mask], sampled_data.y[sampled_data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                test_acc = test(model, sampled_data, device)
                if test_acc > best_epoch_test_acc:
                    best_epoch_test_acc = test_acc
if __name__ == "__main__":
    dataset_name_ls = ["CoraGraphDataset", "CiteSeerGraphDataset", "PubMedGraphDataset"]
    sample_rate_ls = [0.125, 0.25, 0.375]
    for dataset_name in dataset_name_ls:
        for sample_rate in sample_rate_ls:
            config_path = f"hyp_param_rst/top5_{dataset_name}_{sample_rate}_config.csv"
            config_df = pd.read_csv(config_path)
            mean_array = np.zeros(len(config_df))
            std_array = np.zeros(len(config_df))
            for config_idx in range(len(config_df)):
                crnt_config = config_df.iloc[config_idx]
                hidden_dim = config_df.iloc[config_idx]["hidden_dim"]
                model_type = config_df.iloc[config_idx]["model_type"]
                num_epochs = config_df.iloc[config_idx]["num_epochs"]
                sample_rate = config_df.iloc[config_idx]["sample_rate"]
                dataset_name = config_df.iloc[config_idx]["dataset_name"]
                weight_decay = config_df.iloc[config_idx]["weight_decay"]
                learning_rate = config_df.iloc[config_idx]["learning_rate"]
                activation_type = config_df.iloc[config_idx]["activation_type"]
                num_hidden_layers = config_df.iloc[config_idx]["num_hidden_layers"]

                config_dict = {
                    "hidden_dim": hidden_dim,
                    "model_type": model_type,
                    "num_epochs": num_epochs,
                    "sample_rate": sample_rate,
                    "dataset_name": dataset_name,
                    "weight_decay": weight_decay,
                    "learning_rate": learning_rate,
                    "activation_type": activation_type,
                    "num_hidden_layers": num_hidden_layers
                }