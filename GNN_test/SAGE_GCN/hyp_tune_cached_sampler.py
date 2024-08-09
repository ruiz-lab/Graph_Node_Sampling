import os
import yaml
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import networkx as nx
import dgl
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *
from model import Modified_GCN, Modified_SAGE, Graph_Loader 
from train import validation, test

dataset_name_ls = ['OGB_MAG']
trace_type_ls = ['Laplacian', 'Feature']
exclusion_type_ls = ['Largest', 'Random']
cached_sampler_dict = {}
for dataset_name in dataset_name_ls:
    for trace_type in trace_type_ls:
        for exclusion_type in exclusion_type_ls:
            key = f"{dataset_name}_{trace_type}_{exclusion_type}"
            cached_sampler_dict[key] = Cached_Sampler(dataset_name, trace_type, exclusion_type)

def train():
    wandb.init()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    config = wandb.config
    dataset_name = config.dataset_name
    model_type = config.model_type
    activation_type = config.activation_type
    num_hidden_layers = config.num_hidden_layers
    hidden_dim = config.hidden_dim

    optimizer_type = config.optimizer_type
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay

    trace_type = config.trace_type
    exclusion_type = config.exclusion_type
    sample_rate = config.sample_rate

    num_epochs = config.num_epochs
    batch_size = config.batch_size

    # load dataset from dataset_name
    if dataset_name != 'OGB_MAG':
        data = dataset_name2dataset(dataset_name).to(device)
    else:
        data = load_ogbn_mag().to(device)
    input_dim = data.num_features
    output_dim = data.y.max().item() + 1
    
    # initialize model
    if model_type == 'GCN':
        model = Modified_GCN(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)
    elif model_type == 'SAGE':
        model = Modified_SAGE(input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type)
    else:
        raise ValueError('Invalid model type')
    model = model.to(device)

    # initialize optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer type')

    # loss function
    criterion = nn.NLLLoss()

    # initialize sampler
    key = f"{dataset_name}_{trace_type}_{exclusion_type}"
    sampler = cached_sampler_dict[key]
    num_excluded = int(data.num_nodes * (1 - sample_rate))
    sub_data = sampler.sample(data, num_excluded)
    print("num_nodes", data.num_nodes, "num_edges", data.num_edges)

    # make dataset and dataloader
    train_dataloader = NeighborLoader(sub_data, num_neighbors=[-1], batch_size=batch_size, shuffle=True)


    # train model
    for epoch in range(num_epochs):
        for batch_data in tqdm(train_dataloader, desc=f"epoch {epoch+1}/{num_epochs}"):
            model.train()
            optimizer.zero_grad()
            out = model(x=batch_data.x, edge_index=batch_data.edge_index)
            train_loss = criterion(out, batch_data.y)
            train_loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            correct = pred.eq(batch_data.y)
            sub_train_acc = correct.sum().item() / batch_data.y.size(0)
            wandb.log({'sub_train_loss': train_loss.item(), 'sub_train_acc': sub_train_acc})

        if (epoch+1) % 5 == 0:
            valid_acc = validation(model, data)
            test_acc = test(model, data)    
            wandb.log({'valid_acc': valid_acc, 'test_acc': test_acc})
        
    return 


if __name__ == '__main__':
    with open('cache_sweep_config.yaml', 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])

    wandb.agent(sweep_id, function=train)

    wandb.finish()