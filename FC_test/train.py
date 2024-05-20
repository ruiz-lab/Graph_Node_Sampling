import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import dgl
import torch_geometric as pyg
import seaborn as sns
import matplotlib.pyplot as plt
import re
from torch.utils.data import DataLoader
from model import *
from utils import *
from tqdm import tqdm

def evaluate(model, criterion, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        eval_loss = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            eval_loss += loss.item()
            _, pred = out.max(dim=1)
            correct += int(pred.eq(y).sum().item())
            total += len(data.y)
    acc = correct / total
    avg_loss = eval_loss/len(test_loader)
    print(f"Evaluation Accuracy: {acc:.4f}, Evaluation Loss: {avg_loss:.4f}")
    return acc, avg_loss
    
def train(n_epochs, model, criterion, optimizer, train_loader, valid_loader, device):
    model.train()
    train_loss = []
    valid_timestamp = []
    valid_loss = []
    for epoch in tqdm(range(n_epochs), desc='Training'):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss.append(epoch_loss)
        if epoch % 20 == 0:
            valid_timestamp.append(epoch)
            print(f"epoch {epoch}, epoch avg loss {epoch_loss/len(train_loader)}, total avg loss {sum(train_loss)/(epoch+1)}")
            acc, avg_loss = evaluate(model, criterion, valid_loader, device)
            print(f"Validation Accuracy: {acc:.4f}, Validation Loss: {avg_loss:.4f}")
            valid_loss.append(avg_loss)
    return model, train_loss, valid_timestamp, valid_loss

dataset_name_ls = ['CoraGraphDataset', 'CiteseerGraphDataset', 'PubmedGraphDataset', 'TexasDataset', 'WisconsinDataset', 'CornellDataset', 'SquirrelDataset', 'ChameleonDataset']
# dataset_name_ls = ['CoraGraphDataset']
pattern = re.compile(r'(Graph)?Dataset')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 200

for dataset_name in dataset_name_ls:
    dataset = dgl.data.__getattribute__(dataset_name)()
    data = convert_dgl_to_pyg(dataset)
    train_mask, val_mask, test_mask = generate_mask(data.num_nodes)
    train_data = FC_Dataset(data, train_mask)
    valid_data = FC_Dataset(data, val_mask)
    test_data = FC_Dataset(data, test_mask)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model = FCNet(data.num_features, data.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model, train_loss, valid_timestamp, valid_loss = train(n_epochs, model, criterion, optimizer, train_loader, valid_loader, device)
    acc, avg_loss = evaluate(model, criterion, test_loader, device)
    sns.lineplot(x=valid_timestamp, y=valid_loss, label='Validation Loss')
    sns.lineplot(x=range(n_epochs), y=train_loss, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.ylim(0, 10)
    plt.title(f"{pattern.sub('', dataset_name)} Test Loss {avg_loss:.4f} Acc {acc:.4f}", fontsize=20)
    plt.savefig(f"img/{pattern.sub('', dataset_name)}_loss.png")
    plt.close()
