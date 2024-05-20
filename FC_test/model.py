import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch_geometric as pyg
import numpy as np
from torch_geometric.datasets import Planetoid


class FCNet(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(FCNet, self).__init__()
        self.embedding = nn.Linear(feature_dim, 128, bias=False)

        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x /= np.sqrt(x.shape[1])
        x = self.model(x)
        return x

