import dgl.data
import torch
import torch_geometric
import torch_geometric as pyg
import numpy as np
import networkx as nx
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid
import re
import matplotlib.pyplot as plt
import seaborn as sns
import dgl
from torch_geometric.datasets import OGB_MAG


class GraphNodeSampler: # down sample graph according to leverage score, both threshold and number of excluded nodes are supported
    def __init__(self, data:torch_geometric.data.data.Data, threshold=0.2, num_excluded=100) -> None:
        self.data = data
        self.threshold = threshold
        self.num_excluded = num_excluded
        self.n = data.num_nodes
        self.leverage_score = self.compute_leverage_score()

    def compute_leverage_score(self, x=None):
        if x is None:
            x = self.data.x
        print("calculating hat matrix\n\n...\n")
        # normalized
        x = x / (torch.norm(x, dim=0) + 1e-6)
        H = x @ torch.linalg.pinv(x.T @ x) @ x.T
        leverage_score = torch.diag(H)
        print("leverage_score computed")
        return leverage_score

    def get_idx_by_threshold(self, threshold=None, exclude_low=True):
        if threshold is None:
            threshold = self.threshold
        if exclude_low:
            idx = torch.where(self.leverage_score < threshold)[0]
        else:
            idx = torch.where(self.leverage_score > threshold)[0]
        print(f"{len(idx)}, {len(idx) / self.n * 100:.2f}% nodes are excluded")
        return idx
    
    def get_idx_by_num_excluded(self, num_excluded=None):
        if num_excluded is None:
            num_excluded = self.num_excluded
        return torch.topk(self.leverage_score, num_excluded, largest=False)[1]
    
    def exclude_sample(self, idx, relabel=False):
        device = self.data.x.device
        idx = idx.to(device)
        full_size_idx = torch.ones(self.n, dtype=torch.bool, device=device)
        full_size_idx[idx] = False
        train_mask = self.data.train_mask & full_size_idx
        test_mask = self.data.test_mask & full_size_idx
        val_mask = self.data.val_mask & full_size_idx
        edge_index = pyg.utils.subgraph(idx, self.data.edge_index, relabel_nodes=relabel)[0]
        print(f"sampled training set size: {train_mask.sum()}, new test set size: {test_mask.sum()}, new val set size: {val_mask.sum()}")
        print(f"original: training set size: {self.data.train_mask.sum()}, original test set size: {self.data.test_mask.sum()}, original val set size: {self.data.val_mask.sum()}")
        print(f"percentage preserved: training set: {train_mask.sum() / self.data.train_mask.sum() * 100:.2f}%, test set: {test_mask.sum() / self.data.test_mask.sum() * 100:.2f}%, val set: {val_mask.sum() / self.data.val_mask.sum() * 100:.2f}%")
        if relabel:
            # print("num nodes remain:", sum(full_size_idx), flush=True)
            new_x = self.data.x[full_size_idx, :]
            new_y = self.data.y[full_size_idx]
            # print("new_x shape:", new_x.shape, flush=True)
        else:
            new_x = self.data.x
            new_y = self.data.y
        if relabel:
            return pyg.data.Data(x=new_x, edge_index=edge_index, y=new_y)
        return pyg.data.Data(x=new_x, edge_index=edge_index, y=new_y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
    
class A_Opt_Sampler:
    def __init__(self, data:torch_geometric.data.data.Data, num_excluded=100, trace_type=None):
        
        if trace_type == "Laplacian":
            A = pyg.utils.to_dense_adj(data.edge_index).squeeze()
            D = torch.diag(A.sum(dim=1))
            self.L = D - A
            self.L_squared_diag = torch.sum(self.L**2, dim=1)
        elif trace_type == "Feature":
            X = data.x
            self.XXT = X @ X.T
            self.XXT_squared_diag = torch.sum(self.XXT**2, dim=1)
        elif trace_type is None:
            A = pyg.utils.to_dense_adj(data.edge_index).squeeze()
            D = torch.diag(A.sum(dim=1))
            self.L = D - A
            self.L_squared_diag = torch.sum(self.L**2, dim=1)
            X = data.x
            self.XXT = X @ X.T
            self.XXT_squared_diag = torch.sum(self.XXT**2, dim=1)
        elif trace_type == "heritage":
            pass
        self.num_excluded = num_excluded
    
    def Laplacian_Trace_Opt_Index(self, num_excluded=None, exclude_largest=True):
        if num_excluded is None:
            num_excluded = self.num_excluded
        idx = torch.topk(self.L_squared_diag, num_excluded, largest=exclude_largest)[1]
        return idx, torch.sum(self.L_squared_diag[idx])
    
    def Feature_Trace_Opt_Index(self, num_excluded=None, exclude_largest=True):
        if num_excluded is None:
            num_excluded = self.num_excluded
        idx = torch.topk(self.XXT_squared_diag, num_excluded, largest=exclude_largest)[1]
        return idx, torch.sum(self.XXT_squared_diag[idx])
    
    def sample_from_idx(self, idx, data:torch_geometric.data.data.Data):
        device = data.x.device
        idx = idx.to(device)
        full_size_idx = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        full_size_idx[idx] = False
        train_mask = data.train_mask & full_size_idx
        test_mask = data.test_mask & full_size_idx
        val_mask = data.val_mask & full_size_idx
        edge_index = pyg.utils.subgraph(idx, data.edge_index, relabel_nodes=False)[0]
        return pyg.data.Data(x=data.x, edge_index=edge_index, y=data.y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

class RandomSampler:
    def __init__(self, num_excluded=100):
        self.num_excluded = num_excluded
    
    def Exclusion_Index(self, data:torch_geometric.data.data.Data, num_excluded=None):
        if num_excluded is None:
            num_excluded = self.num_excluded
        idx = torch.randperm(data.num_nodes)[:num_excluded]
        return idx

    def sample_from_idx(self, data:torch_geometric.data.data.Data, idx):
        device = data.x.device
        idx = idx.to(device)
        full_size_idx = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        full_size_idx[idx] = False
        train_mask = data.train_mask & full_size_idx
        test_mask = data.test_mask & full_size_idx
        val_mask = data.val_mask & full_size_idx
        edge_index = pyg.utils.subgraph(idx, data.edge_index, relabel_nodes=False)[0]
        return pyg.data.Data(x=data.x, edge_index=edge_index, y=data.y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

class Cached_Sampler:
    def __init__(self, dataset_name, trace_type, exclusion_type):
        self.dataset_name = dataset_name
        self.trace_type = trace_type
        self.exclusion_type = exclusion_type
        self.cache_path = f"cache/{self.dataset_name}_{self.trace_type}_score_descending_indices.pt"
        self.__load_cache(self.cache_path)

    def __load_cache(self, path):
        self.desending_indices = torch.load(path)

    def sample(self, data:torch_geometric.data.data.Data, num_excluded=100):
        if self.exclusion_type == "Largest":
            exclusion_indices = self.desending_indices[:num_excluded]
        elif self.exclusion_type == "Smallest":
            exclusion_indices = self.desending_indices[-num_excluded:]
        device = data.x.device
        exclusion_indices = exclusion_indices.to(device)
        full_size_idx = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        full_size_idx[exclusion_indices] = False
        train_mask = data.train_mask & full_size_idx
        test_mask = data.test_mask & full_size_idx
        val_mask = data.val_mask & full_size_idx
        edge_index = pyg.utils.subgraph(exclusion_indices, data.edge_index, relabel_nodes=False)[0]
        return pyg.data.Data(x=data.x, edge_index=edge_index, y=data.y, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)


def leverage_score(data:Data):
    X = data.x
    # norm = torch.linalg.vector_norm(X, ord=2, dim=1)
    # normed_X = X / torch.maximum(norm, torch.full_like(norm, 1e-6)).unsqueeze(1)
    U, _, _ = torch.svd(X)
    H = U * U
    leverage_score = H.sum(dim=1)
    return leverage_score

def convert_dgl_to_pyg(dgl_dataset):
    # Assuming the DGL dataset has only one graph for simplicity
    dgl_graph = dgl_dataset[0]

    # Convert edge indices from DGL to PyG format
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # Extract node features
    # Assuming 'feat' is the name of the node features in the DGL graph
    if 'feat' in dgl_graph.ndata:
        x = dgl_graph.ndata['feat']
    else:
        raise ValueError("Node features 'feat' not found in the DGL graph.")

    # Extract labels
    # Assuming 'label' is the name of the node labels in the DGL graph
    if 'label' in dgl_graph.ndata:
        y = dgl_graph.ndata['label']
    else:
        raise ValueError("Node labels 'label' not found in the DGL graph.")
    if len(dgl_graph.ndata["train_mask"].shape) > 1:
        train_mask = dgl_graph.ndata["train_mask"][:, 0]
        val_mask = dgl_graph.ndata["val_mask"][:, 0]
        test_mask = dgl_graph.ndata["test_mask"][:, 0]
    else:
        train_mask = dgl_graph.ndata["train_mask"]
        val_mask = dgl_graph.ndata["val_mask"]
        test_mask = dgl_graph.ndata["test_mask"]
    # Create the PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_classes=torch.unique(y).shape[0], num_features=x.shape[1])

    return pyg_data

def feature_homophily_score(data):
    X = data.x
    norm = torch.linalg.vector_norm(X, ord=2, dim=1)
    normed_X = X / torch.maximum(norm, torch.full_like(norm, 1e-6)).unsqueeze(1)
    A = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    H_score = torch.trace(A @ normed_X @ normed_X.T) / data.num_edges
    return H_score.item()

def Laplacian_feature_homophily_score(data):
    X = data.x
    norm = torch.linalg.vector_norm(X, ord=2, dim=1)
    normed_X = X / torch.maximum(norm, torch.full_like(norm, 1e-6)).unsqueeze(1)
    A = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    D = torch.diag(A.sum(dim=1))
    L = D - A
    H_score = torch.trace( -L @ normed_X @ normed_X.T) / data.num_edges
    return H_score.item()

def edge_homophily_score(data:Data):
    edge_index = data.edge_index
    if len(edge_index.shape) == 1 or edge_index.shape[1] == 0:
        return 0
    y = data.y
    y_y = y.unsqueeze(0) == y.unsqueeze(1)
    edge_homophily = torch.sum(y_y[edge_index[0], edge_index[1]]).item()
    return edge_homophily / edge_index.shape[1]

def categorical_homophily_score(adj, y):
    kronecker_delta = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    m = torch.sum(adj) / 2
    deg = adj.sum(dim=1)
    degree_term = (deg.unsqueeze(0) * deg.unsqueeze(1)) / (2 * m)
    return torch.sum((adj - degree_term) * kronecker_delta) / 2

def categorical_Q_homophily_score(adj, y):
    m = torch.sum(adj) / 2
    return categorical_homophily_score(adj, y) / m

def categorical_normalized_Q_homophily_score(adj, y):
    deg = adj.sum(dim=1)
    m = torch.sum(adj) / 2
    degree_term = (deg.unsqueeze(0) * deg.unsqueeze(1)) / (2 * m)
    kronecker_delta = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    Q_max = 2 * m - torch.sum(degree_term * kronecker_delta)
    Q = categorical_Q_homophily_score(adj, y)
    return Q / Q_max

def continuous_homophily_score(adj, x):
    m = torch.sum(adj) / 2
    deg = adj.sum(dim=1)
    degree_term = (deg.unsqueeze(0) * deg.unsqueeze(1)) / (2 * m)
    X_matrix = x.unsqueeze(0) * x.unsqueeze(1)
    cov = torch.sum((adj - degree_term) * X_matrix) / (2 * m)
    return cov

def continuous_r_homophily_score(adj, x):
    cov = continuous_homophily_score(adj, x)
    m = torch.sum(adj) / 2
    deg = adj.sum(dim=1)
    degree_term = (deg.unsqueeze(0) * deg.unsqueeze(1)) / (2 * m)
    X_matrix = x.unsqueeze(0) * x.unsqueeze(1)
    deg_diag = torch.diag(deg)
    var = torch.sum((deg_diag - degree_term) * X_matrix) / (2 * m)
    return cov / var

def random_SBM_graph(data):
    n = data.num_nodes
    k = data.y.unique().size(0)
    p_ls = torch.zeros(k)
    q_ls = torch.zeros(k)
    ramdom_edge_index = None
    for c in range(k):
        node_mask = data.y == c
        inclass_node_idx = torch.where(node_mask)[0]
        class_subgraph = pyg.utils.subgraph(node_mask, data.edge_index)[0]
        class_n = node_mask.sum()
        class_m = class_subgraph.size(1)
        p = 2 * class_m / (class_n * (class_n - 1))
        p_ls[c] = p
        in_class_edge_index = pyg.utils.erdos_renyi_graph(class_n, p).T
        for i, e in enumerate(in_class_edge_index):
            e[0] = inclass_node_idx[e[0]]
            e[1] = inclass_node_idx[e[1]]
            in_class_edge_index[i] = e
        if ramdom_edge_index is None:
            ramdom_edge_index = in_class_edge_index.T
        else:
            ramdom_edge_index = torch.cat((ramdom_edge_index, in_class_edge_index.T), dim=1)

        outside_node_idx = torch.where(~node_mask)[0]
        outside_class_n = n - class_n
        outside_class_m = data.edge_index.size(1) - class_m
        q = 2 * outside_class_m / (outside_class_n * (outside_class_n - 1))
        q_ls[c] = q
        A_in_out = torch.rand(class_n, outside_class_n) < q
        outside_edge_index = torch.vstack(torch.where(A_in_out)).T
        for i, e in enumerate(outside_edge_index):
            e[0] = inclass_node_idx[e[0]]
            e[1] = outside_node_idx[e[1]]
            outside_edge_index[i] = e
        ramdom_edge_index = torch.cat((ramdom_edge_index, outside_edge_index.T), dim=1)

    SBM_data = pyg.data.Data(x=data.x, edge_index=ramdom_edge_index, y=data.y)
    return SBM_data, p_ls, q_ls

def overall_random_graph(data):
    '''
    returns a standard random graph G(n, p)
    '''
    n = data.num_nodes
    m = data.edge_index.size(1)
    p = 2 * m / (n * (n - 1))
    random_A = torch.rand(n, n) < p
    random_edge_index = torch.vstack(torch.where(random_A))
    random_data = pyg.data.Data(x=data.x, edge_index=random_edge_index, y=data.y)
    return random_data

def plot_double_bars(names, list1, list2):
    """
    Plots a bar chart with two bars for each name, representing values from list1 and list2.

    Parameters:
    - names: List of names (strings).
    - list1: List of numbers corresponding to the first group.
    - list2: List of numbers corresponding to the second group.
    """
    # Check if the lengths of the lists match
    if not (len(names) == len(list1) == len(list2)):
        raise ValueError("All input lists must have the same length.")
    
    # Create an index for each tick position
    x = np.arange(len(names))
    
    # Bar width
    width = 0.35  
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, np.round(list1, 2), width, label='edge homophily')
    bars2 = ax.bar(x + width/2, np.round(list2, 2), width, label='feat homophily')
    
    # Adding some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Homophily Score')
    ax.set_title('Homophily Scores of Datasets')
    ax.set_xticks(x)
    pattern = re.compile(r'(Graph)?Dataset')
    ax.set_xticklabels([pattern.sub('', name) for name in names])
    ax.legend()
    
    # Attach a text label above each bar in *bars1* and *bars2*, displaying its height.
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.show()

def SBM_Prob_Matrix(data:Data):
    edge_index = data.edge_index
    y = data.y
    Counter_Matrix = torch.zeros(data.num_classes, data.num_classes)
    for (node_1, node_2) in edge_index.T:
        Counter_Matrix[y[node_1], y[node_2]] += 1
        Counter_Matrix[y[node_2], y[node_1]] += 1
    sizes = SBM_Sizes(data)
    Possible_Matrix = torch.zeros_like(Counter_Matrix)
    for c in range(data.num_classes):
        Possible_Matrix[c,c] = sizes[c] * (sizes[c] - 1)
        for c_ in range(c+1, data.num_classes):
            Possible_Matrix[c,c_] = sizes[c] * sizes[c_]
            Possible_Matrix[c_,c] = sizes[c] * sizes[c_]
    return Counter_Matrix / Possible_Matrix    

def SBM_Sizes(data:Data):
    y = data.y
    sizes = torch.zeros(data.num_classes, dtype=torch.int64)
    for c in range(data.num_classes):
        sizes[c] = (y == c).sum()
    return sizes

def SBM_Normal_Feature_Generator(data:Data):
    mean_feature = torch.zeros(data.num_classes, data.num_features)
    std_feature = torch.zeros(data.num_classes, data.num_features)
    y = data.y
    for c in range(data.num_classes):
        class_mask = y == c
        mean_feature[c] = data.x[class_mask].sum(dim=0)
        std_feature[c] = data.x[class_mask].std(dim=0)
    std_feature = torch.nan_to_num(std_feature, nan=0)
    generated_x = []
    sizes = SBM_Sizes(data)
    for c in range(data.num_classes):
        zero_mask = std_feature[c] == 0
        cov = torch.diag(std_feature[c] + zero_mask + 1)
        multi_normal = torch.distributions.MultivariateNormal(mean_feature[c], cov)
        crnt_x = multi_normal.sample((sizes[c].item(), ))
        crnt_x[:, zero_mask] = 0
        generated_x.append(crnt_x)
    generated_x = torch.cat(generated_x, dim=0)
    return generated_x

def generate_sbm_to_pyg(sizes, probs, x=None, seed=None, mask_source_data=None):
    """
    Generate a Stochastic Block Model (SBM) and convert it to PyTorch Geometric Data format.

    Args:
    sizes (list of int): Number of nodes in each block.
    probs (list of list of float): Probability matrix for edges within and between blocks.

    Returns:
    torch_geometric.data.Data: Graph data in PyTorch Geometric format.
    """
    # Generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Convert to edge index format expected by PyG
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    node_labels = torch.zeros(sum(sizes), dtype=torch.long)
    start = 0
    for idx, size in enumerate(sizes):
        node_labels[start:start+size] = idx
        start += size
    train_mask = None
    test_mask = None
    val_mask = None
    if mask_source_data is not None:
        train_mask = mask_source_data.train_mask
        test_mask = mask_source_data.test_mask
        val_mask = mask_source_data.val_mask
    # Create PyTorch Geometric Data object
    data = Data(x=x, y=node_labels, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, edge_index=edge_index, num_nodes=sum(sizes), num_edges=edge_index.size(1), num_classes=len(sizes))
    
    return data

def pyg2SBM(data:Data):
    Prob_Matrix = SBM_Prob_Matrix(data)
    sizes = SBM_Sizes(data)
    generated_x = SBM_Normal_Feature_Generator(data) 
    SyntheticSBM = generate_sbm_to_pyg(sizes, Prob_Matrix, generated_x, mask_source_data=data)
    return SyntheticSBM

def cosine_similarity_matrix(X):
    # Step 1: Normalize each row to have unit norm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device)
    norms = torch.linalg.norm(X, ord=2, dim=1)
    X_normalized = X / norms[:, None]
    cos = X_normalized @ X_normalized.T
    return cos.cpu()

def L2_distance_matrix(X):
    # X tensor
    # Step 1: Expand dimensions to enable broadcasting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # X = X.to(device)
    # X_expanded = X.unsqueeze(1)
    # X_transposed = X.unsqueeze(0)

    # Step 2: Compute the element-wise squared difference
    squared_diff = torch.square(X.unsqueeze(1) - X.unsqueeze(0))

    # Step 3: Sum along the feature dimension
    sum_squared_diff = torch.sum(squared_diff, dim=2)

    # Step 4: Take the square root to get the L2 distance
    L2_distance_matrix = torch.sqrt(sum_squared_diff)
    return L2_distance_matrix

def plot_heatmap(X, title, save_path, metric='cos'):
    if metric == 'cos':
        mat = cosine_similarity_matrix(X)
    elif metric == 'L2':
        mat = L2_distance_matrix(X)
    
    sns.heatmap(mat)
    plt.title(title, fontsize=20)
    plt.savefig(save_path)
    plt.close()

def plot_train_loss(train_loss, save_path):
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("loss", fontsize=16)
    plt.title("Train Loss", fontsize=20)
    plt.savefig(save_path)
    plt.close()

def plot_val_acc(sampled_val_acc, full_size_val_acc, title, save_path):
    plt.figure()
    plt.plot(sampled_val_acc, label="Sampled Validation")
    plt.plot(full_size_val_acc, label="Full Size Validation")
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0, 1)
    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def TFIDF2BOW_Feature(data:Data):
    x = data.x
    mask = x != 0
    new_x = torch.zeros_like(x)
    new_x[mask] = 1
    data.x = new_x
    return data

def subgraph_from_index(sample_idx, data):
    X = data.x
    new_x = X[sample_idx]
    edge_idx = pyg.utils.subgraph(sample_idx, data.edge_index, relabel_nodes=True)[0]
    sub_data = pyg.data.Data(x=new_x, y=data.y[sample_idx], edge_index=edge_idx)
    return sub_data

def dataset_name2dataset(dataset_name):
    dataset = dgl.data.__getattribute__(dataset_name)()
    data = convert_dgl_to_pyg(dataset)
    if dataset_name == "PubmedGraphDataset":
        data = TFIDF2BOW_Feature(data)
    return data

def load_ogbn_mag():
    dataset = OGB_MAG(root='../data/')
    data = dataset[0]
    
    paper_edge_index = data['paper', 'cites', 'paper'].edge_index
    
    paper_data = data['paper']
    x = paper_data.x
    y = paper_data.y
    train_mask = paper_data.train_mask
    val_mask = paper_data.val_mask
    test_mask = paper_data.test_mask

    paper_data = Data(x=x, edge_index=paper_edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return paper_data