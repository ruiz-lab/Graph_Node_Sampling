import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid


class GConvModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GConvModel, self).__init__()
        self.conv1 = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
class Modified_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type):
        super(Modified_GCN, self).__init__()
        self.activation = nn.ReLU() if activation_type == 'Relu' else nn.Sigmoid()
        self.input_conv = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(pyg.nn.GCNConv(hidden_dim, hidden_dim))
        self.output_conv = pyg.nn.GCNConv(hidden_dim, output_dim)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.input_conv(x, edge_index))
        for layer in self.hidden_layers:
            x = self.activation(layer(x, edge_index))
        x = self.output_conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class Modified_SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type):
        super(Modified_SAGE, self).__init__()
        self.activation = nn.ReLU() if activation_type == 'Relu' else nn.Sigmoid()
        self.input_conv = pyg.nn.SAGEConv(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(pyg.nn.SAGEConv(hidden_dim, hidden_dim))
        self.output_conv = pyg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.input_conv(x, edge_index))
        for layer in self.hidden_layers:
            x = self.activation(layer(x, edge_index))
        x = self.output_conv(x, edge_index)
        return F.log_softmax(x, dim=1)