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
from torch_geometric.datasets import Planetoid
from scipy.sparse.linalg import eigs, eigsh
import seaborn as sns
from tqdm import tqdm
from utils import *
from model import *




def train(n_epochs, model, criterion, optimizer, sub_data, data=None):
    if data is None:
        data = sub_data
    model.train()
    train_loss = []
    sampled_val_acc = []
    full_size_val_acc = []
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        out = model(sub_data)
        loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss {loss.item()}")

        train_loss.append(loss.item())
        
        model.eval()
        if epoch % 50 == 0:
            print("evaluating on sampled validation set")
        _, pred = model(sub_data).max(dim=1)
        correct = int(pred[sub_data.val_mask].eq(sub_data.y[sub_data.val_mask]).sum().item())
        acc = correct / int(data.val_mask.sum())
        if epoch % 50 == 0:
            print(f"Sampled Accuracy: {acc:.4f}")
        sampled_val_acc.append(acc)


        if epoch % 50 == 0:
            print("evaluating on full size validation set")
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        acc = correct / int(data.val_mask.sum())
        
        if epoch % 50 == 0:
            print(f"Full size Accuracy: {acc:.4f}")
        full_size_val_acc.append(acc)

        model.train()
    return model, train_loss, sampled_val_acc, full_size_val_acc

def evaluate(model, sub_data, data=None):
    if data is None:
        data = sub_data
    print("evaluating on sampled test set")
    model.eval()
    _, pred = model(sub_data).max(dim=1)
    correct = int(pred[sub_data.test_mask].eq(sub_data.y[sub_data.test_mask]).sum().item())
    sample_test_acc = correct / int(sub_data.test_mask.sum())
    print(f"Sampled Accuracy: {sample_test_acc:.4f}")

    print("evaluating on full size test set")
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    full_size_acc = correct / int(data.test_mask.sum())
    print(f"Full size Accuracy: {full_size_acc:.4f}")
    model.train()
    return sample_test_acc, full_size_acc

dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'TexasDataset', 'WisconsinDataset', 'CornellDataset', 'SquirrelDataset', 'ChameleonDataset']
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
pattern = re.compile(r'(Graph)?Dataset')
n_epochs = 200

for dataset_name in dataset_name_ls:
    dataset = dgl.data.__getattribute__(dataset_name)()
    data = convert_dgl_to_pyg(dataset)
    print(f"{pattern.sub('', dataset_name)}: {len(dataset)}")
    SyntheticSBM = pyg2SBM(data).to(device)
    print(f"Dataset: {dataset_name}\n Original:num_edges:{data.num_edges}\n Synthetic: num_nodes:{SBM_Sizes(SyntheticSBM)}, num_edges:{SyntheticSBM.num_edges}")
    model = GConvModel(SyntheticSBM.num_features, 16, SyntheticSBM.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    model, train_loss, sampled_val_acc, full_size_val_acc = train(n_epochs, model, criterion, optimizer, SyntheticSBM)
    sample_test_acc, full_size_acc = evaluate(model, SyntheticSBM)
    # plot train loss
    crnt_folder = f"img/SBM_performance/{pattern.sub('', dataset_name)}"
    if not os.path.exists(crnt_folder):
        os.makedirs(crnt_folder)
    plot_train_loss(train_loss, f"{crnt_folder}/{pattern.sub('', dataset_name)}_train_loss.png")
    plot_val_acc(sampled_val_acc, full_size_val_acc, title=f"{pattern.sub('', dataset_name)} test acc {full_size_acc}", save_path=f"{crnt_folder}/{pattern.sub('', dataset_name)}_val_acc.png")
    # plot_heatmap(generated_x, f"img/random_feature_heatmap/{pattern.sub('', dataset_name)}_cosine_similarity.png", metric="cos")
    # plot_heatmap(generated_x, f"img/random_feature_heatmap/{pattern.sub('', dataset_name)}_L2_Distance.png", metric="L2")

# for dataset_name in dataset_name_ls:
#     dataset = dgl.data.__getattribute__(dataset_name)()
#     data = convert_dgl_to_pyg(dataset)
#     best_sample_val_acc = []
#     best_full_size_val_acc = []
#     best_sample_test_acc = []
#     best_full_size_test_acc = []
#     sample_rate_ls = []
#     if dataset_name == 'Cora':
#         threshold_ls = np.linspace(0.3, 0.6, 5)
#     elif dataset_name == 'CiteSeer':
#         # threshold_ls = np.linspace(0.9, 0.99, 10)
#         threshold_ls = np.linspace(0.9, 0.99, 5)
#     elif dataset_name == 'PubMed':
#         # threshold_ls = np.linspace(0.015, 0.075, 20)
#         threshold_ls = np.linspace(0.015, 0.055, 5)
#     for threshold in threshold_ls:
#         sampler = GraphNodeSampler(data, threshold=threshold)
#         exclude_idx = sampler.get_idx_by_threshold()
#         # idx = sampler.get_idx_by_num_excluded()
#         sub_data = sampler.exclude_sample(exclude_idx)
#         sample_rate = 1- len(exclude_idx) / data.num_nodes
#         sample_rate_ls.append(sample_rate)
#         sub_data = sub_data.to(device)
#         data = data.to(device)
#         # train model
#         model = GConvModel(data.num_features, 16, dataset.num_classes).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#         criterion = nn.NLLLoss()
#         model, train_loss, sampled_val_acc, full_size_val_acc = train(n_epochs, model, criterion, optimizer, sub_data, data)
#         best_sample_val_acc.append(max(sampled_val_acc))
#         best_full_size_val_acc.append(max(full_size_val_acc))
#         # evaluate model
#         sample_test_acc, full_size_acc = evaluate(model, sub_data, data)
#         best_sample_test_acc.append(sample_test_acc)
#         best_full_size_test_acc.append(full_size_acc)
#     # plot validation acc
#     plt.figure()
#     plt.plot(threshold_ls, best_sample_val_acc, label="Sampled")
#     plt.plot(threshold_ls, best_full_size_val_acc, label="Full Size")
#     plt.xlabel("threshold")
#     plt.ylabel("Accuracy")
#     plt.title(f"{dataset_name} Validation Accuracy")
#     plt.legend()
#     plt.savefig(f"img/{dataset_name}_val_acc.png")
#     # plot test acc
#     plt.figure()
#     plt.plot(threshold_ls, best_sample_test_acc, label="Sampled")
#     plt.plot(threshold_ls, best_full_size_test_acc, label="Full Size")
#     plt.xlabel("threshold")
#     plt.ylabel("Accuracy")
#     plt.title(f"{dataset_name} Test Accuracy")
#     plt.legend()
#     plt.savefig(f"img/{dataset_name}_test_acc.png")

