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
import networkx as nx
import dgl
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import *
from model import Modified_GCN, Modified_SAGE

def validation(model, data):
    if data.val_mask.sum().item() == 0:
        return -1
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred.eq(data.y)
    valid_acc = correct[data.val_mask].sum().item() / data.val_mask.sum().item()
    return valid_acc

def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred.eq(data.y)
    test_acc = correct[data.test_mask].sum().item() / data.test_mask.sum().item()
    return test_acc

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
    exclude_largest = True if exclusion_type == 'Largest' else False
    sample_rate = config.sample_rate

    num_epochs = config.num_epochs

    # load dataset from dataset_name
    data = dataset_name2dataset(dataset_name).to(device)
    input_dim = data.num_features
    output_dim = data.num_classes
    
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
    sampler = A_Opt_Sampler(data, trace_type=trace_type)
    num_excluded = int(data.num_nodes * (1 - sample_rate))
    if trace_type == 'Laplacian':
        exclusion_idx, _ = sampler.Laplacian_Trace_Opt_Index(num_excluded=num_excluded, exclude_largest=exclude_largest)
    elif trace_type == 'Feature':
        exclusion_idx, _ = sampler.Feature_Trace_Opt_Index(num_excluded=num_excluded, exclude_largest=exclude_largest)
    else:
        raise ValueError('Invalid exclusion type')
    
    # generate sampled data
    sub_data = sampler.sample_from_idx(exclusion_idx, data)

    # train model
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(sub_data)
        train_loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
        train_loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct = pred.eq(sub_data.y)
        sub_train_acc = correct[sub_data.train_mask].sum().item() / sub_data.train_mask.sum().item()
        train_acc = correct[data.train_mask].sum().item() / data.train_mask.sum().item()
        wandb.log({'sub_train_loss': train_loss.item(), 'sub_train_acc': sub_train_acc, 'train_acc': train_acc})



        if (epoch+1) % 10 == 0:
            valid_acc = validation(model, data)
            test_acc = test(model, data)    
            wandb.log({'valid_acc': valid_acc, 'test_acc': test_acc})
        
    return 

if __name__ == '__main__':
    with open('sweep_config.yaml', 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, project=sweep_config['project'])

    wandb.agent(sweep_id, function=train)

    wandb.finish()