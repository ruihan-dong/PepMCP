import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

seed = 42
np.random.seed(seed)
    
# positive: MemAMPs
df_pos = pd.read_csv('../data/MemAMPs.csv')
pos_seqs = df_pos.seq.tolist()
pos_names = df_pos.ID.tolist()

pos_labels = []
for seq, name in zip(pos_seqs, pos_names):
    filename = 'MCP-Pro' + str(name[-3:]) + '-1us.txt'
    df = pd.read_csv('../data/MCP-txt/' + filename, sep='\t')
    values = df.ContactProb.values
    mcp_seq = ''.join(df.AA.tolist())
    assert seq == mcp_seq
    pos_labels.append(values)

# negative: non-AMPs filled with 0
neg_labels = []
df_neg = pd.read_csv('../data/pdb_sol_neg.txt', header=None)
neg_seqs = df_neg.iloc[:,1].values.tolist()
neg_names = df_neg.iloc[:,0].values.tolist()
for neg_seq in neg_seqs:
    neg_labels.append(np.zeros(len(neg_seq)))

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# residue-level train/val/test split
def res_split(output_dir, seqs, labels, names):
    os.makedirs(output_dir, exist_ok=True)

    outputs = [[], [], [], [], []]
    for i, (seq, label) in enumerate(zip(seqs, labels)):
        
        seq_id = names[i]
        seq_len = len(seq)
        
        # 0 for train, 1 for val, 2 for test
        split_mask = torch.full((seq_len,), -1, dtype=torch.long)
        node_indices = np.random.permutation(seq_len)
        test_size = max(1, int(0.2 * seq_len))
        
        split_mask[node_indices[:test_size]] = 2    # test

        for fold, (train_idx, val_idx) in enumerate(kfold.split(node_indices[test_size:])):

            split_mask[node_indices[test_size:][train_idx]] = 0   # train
            split_mask[node_indices[test_size:][val_idx]] = 1  # val
            
            data = {
                'seq_id': seq_id,
                'sequence': seq,
                'split_mask': split_mask,
                'label': label
            }
            outputs[fold].append(data)
        
    # save data
    for i, output in enumerate(outputs):
        output_path = os.path.join(output_dir, str(i+1) + '_sol_seed' + str(seed) + '.pt')
        torch.save(output, output_path)


# seq-level train/val/test split
def seq_split(output_dir, seqs, labels, names):
    os.makedirs(output_dir, exist_ok=True)

    seq_num = len(seqs)
    
    seq_mask = torch.full((seq_num,), -1, dtype=torch.long)
    node_indices = np.random.permutation(seq_num)
    
    test_size = max(1, int(0.2 * seq_num))

    # 0 for train, 1 for val, 2 for test
    seq_mask[node_indices[:test_size]] = 2

    for fold, (train_idx, val_idx) in enumerate(kfold.split(node_indices[test_size:])):

        seq_mask[node_indices[test_size:][train_idx]] = 0   # train
        seq_mask[node_indices[test_size:][val_idx]] = 1  # val

        output = []
        for i, (seq, label) in enumerate(zip(seqs, labels)):
            
            seq_id = names[i]
            seq_len = len(seq)
            
            split_mask = torch.full((seq_len,), seq_mask[i], dtype=torch.long)
            
            data = {
                'seq_id': seq_id,
                'sequence': seq,
                'split_mask': split_mask,
                'label': label
            }
            output.append(data)
        
        # save data
        output_path = os.path.join(output_dir, str(fold+1) + '_sol_seed' + str(seed) + '.pt')
        torch.save(output, output_path)


# res_split('../data/res_split/', pos_seqs+neg_seqs, pos_labels+neg_labels, pos_names+neg_names)
seq_split('../data/seq_split/', pos_seqs+neg_seqs, pos_labels+neg_labels, pos_names+neg_names)
# seq_split('../data/seq_split/', pos_seqs, pos_labels, pos_names)