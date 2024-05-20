import numpy as np
import numpy.linalg as la
import torch_geometric as pyg
import json
import time
import os
import argparse
import matplotlib.pyplot as plt
from datetime import timedelta
from utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--feature', type=int, default=0, help='whether to use feature matrix')
argparser.add_argument('--exclude_low', type=int, default=1, help='whether exclude nodes with low leverage score')
argparser.add_argument('--csv_dir', type=str, default='trace_rank_csv', help='directory to save csv file')
argparser = argparser.parse_args()

def get_sub_graph(data, threshold=0.2, exclude_low=True):
    sampler = GraphNodeSampler(data)
    idx = sampler.get_idx_by_threshold(threshold, exclude_low)
    sub_data = sampler.exclude_sample(idx, relabel=True)
    return sub_data

def calc_rank(data, epsilon=1e-10, feature=False):
    adj = pyg.utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].numpy()
    # print("calculating rank of adjacency matrix", flush=True)
    # print("shape of adjacency matrix:", adj.shape, flush=True)
    if not feature:
        U, S, V = la.svd(adj)
    else:
        # print(adj.shape, data.x.shape)
        U, S, V = la.svd(adj @ data.x.numpy() @ data.x.numpy().T)
    rank = np.sum(S > epsilon)
    # print(f"rank of adjacency matrix: {rank}", flush=True)
    return rank

def calc_trace(data, feature=False):
    adj = pyg.utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].numpy()
    # print("calculating trace of adjacency matrix", flush=True)
    if not feature:
        trace = np.trace(adj)
    else:
        trace = np.trace(adj @ data.x.numpy() @ data.x.numpy().T)
    # print(f"trace of adjacency matrix: {trace}", flush=True)
    trace /= data.num_nodes
    return trace

def plot_rst(full_size, sub_size, threshold_ls, dataset_name, metric_name):
    plt.plot(threshold_ls, sub_size, label='sub '+metric_name)
    plt.plot(threshold_ls, full_size, label='full size '+metric_name)
    plt.title(f'{dataset_name} {metric_name}')
    plt.xlabel('threshold')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'img/{dataset_name}_{metric_name}.png')
    plt.close()

def main(args):
    feature = bool(args.feature)
    exclude_low = args.exclude_low
    prefix = 'exclude_low_' if exclude_low else 'exclude_high_'
    print(prefix)
    dataset_name_ls = ['Cora', 'CiteSeer', 'PubMed']
    rst_dict = {}
    for dataset_name in dataset_name_ls:
        source = Planetoid(root='../data', name=dataset_name)
        data = source[0]

        if dataset_name == 'Cora':
            threshold_ls = np.linspace(0.4, 0.8, 5)
            # threshold_ls = np.array([0.4])
        elif dataset_name == 'CiteSeer':
            threshold_ls = np.linspace(0.9, 0.99, 5)
            # threshold_ls = np.array([0.95])
        elif dataset_name == 'PubMed':
            threshold_ls = np.array([0.02])
            threshold_ls = np.linspace(0.015, 0.055, 5)
        full_size_rank_ls = []
        sub_rank_ls = []
        full_size_trace_ls = []
        sub_trace_ls = []
        print(f"dataset: {dataset_name}", flush=True)
        for threshold in threshold_ls:
            print(f"threshold: {threshold}", flush=True)
            sub_data = get_sub_graph(data, threshold, exclude_low)  
            # print("subdata num nodes:", sub_data.num_nodes, flush=True) 
            start = time.time()
            full_size_rank = calc_rank(data, feature=feature)
            end = time.time()
            print(f"full size rank time: {timedelta(seconds=end-start)}\n", flush=True)
            start = time.time()
            full_size_trace = calc_trace(data, feature=feature)
            end = time.time()
            print(f"full size trace time: {timedelta(seconds=end-start)}\n", flush=True)
            start = time.time()
            sub_rank = calc_rank(sub_data, feature=feature)
            end = time.time()
            print(f"sub rank time: {timedelta(seconds=end-start)}\n", flush=True)
            start = time.time()
            sub_trace = calc_trace(sub_data, feature=feature)
            end = time.time()
            print(f"sub trace time: {timedelta(seconds=end-start)}\n", flush=True)
            full_size_rank_ls.append(full_size_rank)
            full_size_trace_ls.append(full_size_trace)
            sub_rank_ls.append(sub_rank)
            sub_trace_ls.append(sub_trace)
            print(f"full size rank: {full_size_rank}", flush=True)
            print(f"sub rank: {sub_rank}", flush=True)
            print(f"rank ratio: {sub_rank / full_size_rank:.4f}", flush=True)
            print('\n', flush=True)
            print(f"full size trace: {full_size_trace}", flush=True)
            print(f"sub trace: {sub_trace}", flush=True)
            if feature:
                print(f"trace ratio: {sub_trace / full_size_trace:.4f}", flush=True)
        feat_str = " feature" if feature else ""
        plot_rst(full_size_rank_ls, sub_rank_ls, threshold_ls, dataset_name, prefix+'rank'+feat_str)
        plot_rst(full_size_trace_ls, sub_trace_ls, threshold_ls, dataset_name, prefix+'trace'+feat_str)

        rst_dict[dataset_name] = {
            'threshold_ls': threshold_ls.tolist(),
            'sub_rank_ls': sub_rank_ls,
            'sub_trace_ls': sub_trace_ls,
            'full_size_rank': full_size_rank,
            'full_size_trace': full_size_trace,
            'num_nodes': int(data.num_nodes)
        }
    # prefix = 'exlude_low_'
    if not feature:
        csv_name = 'trace_rank.csv'
        json_name = 'trace_rank.json'
    else:
        csv_name = 'trace_rank_feature.csv'
        json_name = 'trace_rank_feature.json'
    
    with open(os.path.join(args.csv_dir, prefix+csv_name), 'w') as f:
        f.write('dataset,threshold,sub_rank,sub_trace,full_size_rank,full_size_trace, num_nodes\n')
        for dataset_name in dataset_name_ls:
            for i, threshold in enumerate(rst_dict[dataset_name]['threshold_ls']):
                f.write(f"{dataset_name},{threshold},{rst_dict[dataset_name]['sub_rank_ls'][i]},{rst_dict[dataset_name]['sub_trace_ls'][i]},{rst_dict[dataset_name]['full_size_rank']},{rst_dict[dataset_name]['full_size_trace']},{rst_dict[dataset_name]['num_nodes']}\n")

    with open(prefix+json_name, 'w') as f:
        json.dump(rst_dict, f)
    
    return rst_dict

if __name__ == '__main__':
    main(argparser)