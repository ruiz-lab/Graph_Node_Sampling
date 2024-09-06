import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class GConvModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GConvModel, self).__init__()
        self.conv1 = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data=None, x=None, edge_index=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
class Modified_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type):
        super(Modified_GCN, self).__init__()
        self.activation = nn.ReLU() if activation_type == 'Relu' else nn.Sigmoid()
        self.layers = nn.ModuleList()
        if num_hidden_layers == 1:
            self.layers.append(pyg.nn.GCNConv(input_dim, output_dim))
        else:
            self.layers.append(pyg.nn.GCNConv(input_dim, hidden_dim))
            for _ in range(num_hidden_layers-2):
                self.layers.append(pyg.nn.GCNConv(hidden_dim, hidden_dim))
            self.layers.append(pyg.nn.GCNConv(hidden_dim, output_dim))
    def forward(self, data=None, x=None, edge_index=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
        out_nodes, in_nodes = edge_index
        undirected_edge_index = torch.cat([edge_index, torch.stack([in_nodes, out_nodes], dim=0)], dim=1)
        # print("x.shape", x.shape)
        # print("edge_index.shape", edge_index.shape)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, undirected_edge_index)) 
        x = self.layers[-1](x, undirected_edge_index)
        return F.log_softmax(x, dim=1)
    
class Modified_SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation_type):
        super(Modified_SAGE, self).__init__()
        self.activation = nn.ReLU() if activation_type == 'Relu' else nn.Sigmoid()
        self.layers = nn.ModuleList()
        if num_hidden_layers == 1:
            self.layers.append(pyg.nn.SAGEConv(input_dim, output_dim))
        else:
            self.layers.append(pyg.nn.SAGEConv(input_dim, hidden_dim))
            print("hidden_dim", hidden_dim)
            for _ in range(num_hidden_layers-2):
                self.layers.append(pyg.nn.SAGEConv(hidden_dim, hidden_dim))
            self.layers.append(pyg.nn.SAGEConv(hidden_dim, output_dim))

    def forward(self, data=None, x=None, edge_index=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
        out_nodes, in_nodes = edge_index
        undirected_edge_index = torch.cat([edge_index, torch.stack([in_nodes, out_nodes], dim=0)], dim=1)
        # print("x.shape", x.shape)
        # print("edge_index.shape", edge_index.shape)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, undirected_edge_index))
        x = self.layers[-1](x, undirected_edge_index)
        return F.log_softmax(x, dim=1)
    
class Graph_Dataset(Dataset):
    def __init__(self, data, mask) -> None:
        super(Dataset, self).__init__()
        self.out_edge_dict, self.in_edge_dict = self.graph2edge_dict(data, mask)
        self.x = data.x
        self.num_nodes = data.num_nodes

    def graph2edge_dict(self, data, mask):
        out_edge_dict = {} # {node: [out_edge]} node -> other_node
        in_edge_dict = {} # {node: [in_edge]} other_node -> node
        for node in range(data.num_nodes):
            out_edge_dict[node] = []
            in_edge_dict[node] = []
        for out_node, in_node in tqdm(data.edge_index.T, desc="graph2edge_dict", total=data.edge_index.shape[1]):
            out_node = out_node.item()
            in_node = in_node.item()
            if mask[out_node] or mask[in_node]:
                continue
            out_edge_dict[out_node].append(in_node)
            in_edge_dict[in_node].append(out_node)
        for node in range(data.num_nodes):
            out_edge_dict[node] = torch.tensor(out_edge_dict[node], dtype=torch.long)
            in_edge_dict[node] = torch.tensor(in_edge_dict[node], dtype=torch.long)
        return out_edge_dict, in_edge_dict

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, index):
        crnt_x = self.x[index]
        crnt_out_edge_index = torch.stack([torch.full_like(self.out_edge_dict[index], index), self.out_edge_dict[index]], dim=0)
        crnt_in_edge_index = torch.stack([self.in_edge_dict[index], torch.full_like(self.in_edge_dict[index], index)], dim=0)
        crnt_edge_index = torch.cat([crnt_out_edge_index, crnt_in_edge_index], dim=1)
        return Data(x=crnt_x, edge_index=crnt_edge_index, num_nodes=self.num_nodes, y=index) # here store y as index only for code to run without error

class Graph_Loader():
    def __init__(self, data:Data, batch_size:int, shuffle:bool, mode="train") -> None:
        self.data = data
        self.index = torch.randperm(data.num_nodes) if shuffle else torch.arange(data.num_nodes)
        self.batch_size = batch_size
        self.start_places = torch.arange(0, data.num_nodes, batch_size)
        self.end_places = torch.cat([self.start_places[1:], torch.tensor([data.num_nodes])])
        self.out_edge_dict, self.in_edge_dict = self.graph2edge_dict(data, data.train_mask if mode == "train" else data.val_mask if mode == "val" else data.test_mask)
        self.data_ls = []
        device = data.x.device
        for start, end in tqdm(zip(self.start_places, self.end_places), desc="Making Graph Loader", total=len(self.start_places)):
            self.data_ls.append((self.create_sub_data_from_index(self.index[start:end]).to(device), self.index[start:end]))

    def graph2edge_dict(self, data, mask):
        out_edge_dict = {} # {node: [out_edge]} node -> other_node
        in_edge_dict = {} # {node: [in_edge]} other_node -> node
        for node in range(data.num_nodes):
            out_edge_dict[node] = []
            in_edge_dict[node] = []
        for out_node, in_node in tqdm(data.edge_index.T, desc="graph2edge_dict", total=data.edge_index.shape[1]):
            out_node = out_node.item()
            in_node = in_node.item()
            if mask[out_node] or mask[in_node]:
                continue
            out_edge_dict[out_node].append(in_node)
            in_edge_dict[in_node].append(out_node)
        for node in range(data.num_nodes):
            out_edge_dict[node] = torch.tensor(out_edge_dict[node], dtype=torch.long)
            in_edge_dict[node] = torch.tensor(in_edge_dict[node], dtype=torch.long)
        return out_edge_dict, in_edge_dict
    
    def real_index2sub_index_dict(self, real_index):
        sub_index_dict = {}
        for sub_index, real_index in enumerate(real_index):
            sub_index_dict[real_index.item()] = sub_index
        return sub_index_dict

    def create_sub_data_from_index(self, node_ls):
        node_ls = node_ls.to(self.data.x.device)
        sub_x = torch.index_select(self.data.x, 0, node_ls)
        additional_nodes = []
        sub_edge_index = []
        for index in node_ls:
            index = index.item()

            crnt_out_edge_index = torch.stack([torch.full_like(self.out_edge_dict[index], index), self.out_edge_dict[index]], dim=0)
            for _ in self.out_edge_dict[index]:
                if _ not in node_ls and _ not in additional_nodes:
                    additional_nodes.append(_)

            crnt_in_edge_index = torch.stack([self.in_edge_dict[index], torch.full_like(self.in_edge_dict[index], index)], dim=0)
            for _ in self.in_edge_dict[index]:
                if _ not in node_ls and _ not in additional_nodes:
                    additional_nodes.append(_)

            crnt_edge_index = torch.cat([crnt_out_edge_index, crnt_in_edge_index], dim=1)
            sub_edge_index.append(crnt_edge_index)
        sub_edge_index = torch.cat(sub_edge_index, dim=1)

        additional_x = torch.index_select(self.data.x, 0, torch.tensor(additional_nodes, dtype=torch.long, device=node_ls.device))
        full_x = torch.cat([sub_x, additional_x])

        full_nodes_ls = torch.cat([node_ls, torch.tensor(additional_nodes, dtype=torch.long, device=node_ls.device)])
        real2sub_index_dict = self.real_index2sub_index_dict(full_nodes_ls)

        reindexed_edge_start = torch.tensor([real2sub_index_dict[edge_start.item()] for edge_start in sub_edge_index[0]])
        reindexed_edge_end = torch.tensor([real2sub_index_dict[edge_end.item()] for edge_end in sub_edge_index[1]])
        sub_edge_index = torch.stack([reindexed_edge_start, reindexed_edge_end], dim=0)

        return Data(x=full_x, edge_index=sub_edge_index)
        
    def __len__(self):
        return len(self.start_places)
    
    def __iter__(self):
        for crnt_data in self.data_ls:
            yield crnt_data

    def __getitem__(self, index):
        return self.data_ls[index]