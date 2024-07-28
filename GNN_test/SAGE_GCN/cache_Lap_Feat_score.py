import sys
sys.path.append('..')

from utils import *
import torch
import torch_geometric as pyg

dataset_name_ls = ['OGB_MAG']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for dataset_name in dataset_name_ls:
    if dataset_name != 'OGB_MAG':
        data = dataset_name2dataset(dataset_name)
        A = pyg.utils.to_dense_adj(data.edge_index).squeeze(0)
        D = torch.diag(torch.sum(A, dim=1))
    else:
        data = load_ogbn_mag()
        A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]), (data.num_nodes, data.num_nodes))
        degrees = torch.sparse.sum(A, dim=1).to_dense()
        print(data.num_nodes, degrees.shape)
        D = torch.sparse_coo_tensor(torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)]), degrees, (data.num_nodes, data.num_nodes))
    
    L = D - A

    L = L.to(device)

    Lap_score = torch.sum(L**2, dim=1).detach().cpu()

    L = L.detach().cpu()

    X = data.x
    XXT = torch.mm(X, X.t())
    Feat_score = torch.sum(XXT**2, dim=1)

    with open(f"cache/{dataset_name}_Laplacian_score.pt", 'wb') as f:
        torch.save(Lap_score, f)
    with open(f"cache/{dataset_name}_Feature_score.pt", 'wb') as f:
        torch.save(Feat_score, f)
