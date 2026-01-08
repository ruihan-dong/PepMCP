import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from model import *
torch.manual_seed(1234)

# Hyperparameters
batch_size = 32
edge_k = [4]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path):
    df = pd.read_csv(path, header=None)
    seqs = df.iloc[:, 1].values.tolist()
    names = df.iloc[:, 0].values.tolist()

    output = []
    for i, seq in enumerate(seqs):
        seq_id = names[i]
        sample = {
            'seq_id': seq_id,
            'sequence': seq,
        }
        output.append(sample)
    return output


def predict(model, loader):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_graph in loader:

            batch_graph = batch_graph.to(device)
            
            pred_graph = model(batch_graph)
            predictions = pred_graph.ndata['pred']
            
            batch_sizes = batch_graph.batch_num_nodes().tolist()
            current_idx = 0
            
            for i, num_nodes in enumerate(batch_sizes):
                peptide_preds = predictions[current_idx:current_idx + num_nodes]
                peptide_preds_np = peptide_preds.cpu().numpy()
                
                all_predictions.append({
                    'len': num_nodes,
                    'preds': peptide_preds_np
                })

                current_idx += num_nodes
    
    return all_predictions


if __name__ == '__main__':

    best_model = ResidueGCN()
    best_model.load_state_dict(torch.load('model/Fold_1_res_best_model.pth'))
    best_model.to(device)
    
    pred_data = load_data('../data/test.csv')
    pred_dataset = PredictDataset(pred_data, edge_k)

    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, collate_fn=dgl.batch)

    preds = predict(best_model, pred_loader)

    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)

    # regression
    for pred, info in zip(preds, pred_data):
        fp = open(os.path.join(output_dir, info['seq_id'] + '_PepMCP.txt'), 'a')
        fp.write('%s %s %s\n' % ('#', 'AA', 'MCP'))
        for i in range(pred['len']):
            fp.write('%d %s %g\n' % (i + 1, info['sequence'][i], pred['preds'][i]))
        fp.close()	

    # classification mode
    output_labels = []
    for pred, info in zip(preds, pred_data):
        length = pred['len']
        values = pred['preds'][:length]
        n_mem = np.mean(values)

        if n_mem > 0.2:
            output_labels.append(1)
        else:
            output_labels.append(0)
    df_output = pd.DataFrame(output_labels, columns=['PepMCP'])
    df_output.to_csv(os.path.join(output_dir, 'output_labels.csv'), index=False)
        
