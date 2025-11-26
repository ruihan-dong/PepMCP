import os
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from time import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error


def get_embedding(seq_id, seq):
    emb_path = '../data/ESMC-300M/'
    emb_file = os.path.join(emb_path, seq_id+'.pt')
    if os.path.exists(emb_file):
        emb = torch.load(emb_file)
    else:
        print(f'File Not Found: {seq_id}')
    return emb[1:-1].cpu()           # ankh needs torch.from_numpy(emb), esmc needs emb[1:-1].cpu()

def build_graph(seq, k_values=[2, 4]):
    n = len(seq)

    # source & target node index
    src, dst = [], []
    for k in k_values:
        for i in range(n - k):
            src.append(i)
            dst.append(i + k)
            
            src.append(i + k)  # reverse edge for undirected graph
            dst.append(i)
    
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    return g


class NodeLevelPeptideDataset(data.Dataset):
    """load train/val dataset"""
    
    def __init__(self, input_data, edge_k):
        self.input_data = input_data
        self.k = edge_k
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        data = self.input_data[idx]

        node_features = get_embedding(data['seq_id'], data['sequence'])
        label = torch.tensor(data['label'])
        split_mask = data['split_mask']

        # create graph (sequential edge)
        # seq_len = len(node_features)
        # src = torch.arange(seq_len - 1)  # source node index
        # dst = torch.arange(1, seq_len)   # target node index
        
        g = build_graph(data['sequence'], k_values=self.k)
        g = dgl.add_self_loop(g)
        g.ndata['h'] = node_features
        g.ndata['y'] = label
        g.ndata['split_mask'] = split_mask  # train/val/test split

        return g


class PredictDataset(data.Dataset):
    """ load predicting data (without split_mask)"""
    
    def __init__(self, input_data, edge_k):
        self.input_data = input_data
        self.k = edge_k
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        data = self.input_data[idx]
        
        node_features = get_embedding(data['seq_id'], data['sequence'])
        
        g = build_graph(data['sequence'], k_values=self.k)
        g = dgl.add_self_loop(g)
        g.ndata['h'] = node_features

        return g

# regression metrics
def evaluate(preds, labels):

    spearman = spearmanr(labels, preds)[0]
    pearson = pearsonr(labels, preds)[0]
    r2 = r2_score(labels, preds)
    rmse = mean_squared_error(labels, preds, squared = False) 
    # squared: If True returns MSE value, if False returns RMSE value.

    return spearman, pearson, r2, rmse
