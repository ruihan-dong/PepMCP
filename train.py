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
torch.cuda.manual_seed_all(1234)

# Hyperparameters
edge_k = [4]
learning_rate = 1e-4
batch_size = 4
epoches = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_masked_loss(pred, target, mask):

    masked_pred = pred[mask].to(torch.float32)
    masked_target = target[mask].to(torch.float32)
    
    if len(masked_pred) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(masked_pred, masked_target)

    return loss

def train_epoch(model, loader, optimizer):
    
    model.train()
    total_loss = 0.0
    num_nodes = 0
    
    for batch_graph in loader:
        batch_graph = batch_graph.to(device)

        train_mask = (batch_graph.ndata['split_mask'] == 0)

        pred = model(batch_graph).ndata['pred']
        
        loss = compute_masked_loss(
            pred, 
            batch_graph.ndata['y'], 
            train_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # accumulated loss
        total_loss += loss.item() * train_mask.sum().item()
        num_nodes += train_mask.sum().item()
    
    return total_loss / num_nodes

def validate_epoch(model, loader):
    
    model.eval()
    total_loss = 0.0
    num_nodes = 0

    all_preds, all_labels = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for batch_graph in loader:

            batch_graph = batch_graph.to(device)

            val_mask = (batch_graph.ndata['split_mask'] == 1)
            if not val_mask.any():
                continue
            
            pred = model(batch_graph).ndata['pred']
            
            loss = compute_masked_loss(
                pred,
                batch_graph.ndata['y'],
                val_mask
            )
            
            # accumulated loss
            total_loss += loss.item() * val_mask.sum().item()
            num_nodes += val_mask.sum().item()

            # metrics
            masked_pred = pred[val_mask]
            masked_target = batch_graph.ndata['y'][val_mask]

            all_preds = torch.cat((all_preds, masked_pred.cpu()), 0)
            all_labels = torch.cat((all_labels, masked_target.cpu()), 0)

    [val_spearman, val_pearson, val_r2, val_rmse] = evaluate(all_preds.numpy().flatten(), all_labels.numpy().flatten())
    print('Val Spearman {:.4f}, Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}'.format(val_spearman, val_pearson, val_r2, val_rmse))

    return total_loss / num_nodes


def train_val(model, train_loader, val_loader, optimizer, num_epochs=50, early_stop=10):
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate_epoch(model, val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (Best: {best_val_loss:.6f})')

        # early stopping
        if epochs_without_improvement >= early_stop:
            print(f"\nEarly stopping at {epoch+1} epochs")
            break
    
    return history, best_model


def test(model, loader):
    all_preds, all_labels = torch.Tensor(), torch.Tensor()
    model.eval()

    with torch.no_grad():
        for batch_graph in loader:

            batch_graph = batch_graph.to(device)
            test_mask = (batch_graph.ndata['split_mask'] == 2)
            if not test_mask.any():
                continue
            
            pred = model(batch_graph).ndata['pred']

            masked_pred = pred[test_mask]
            masked_target = batch_graph.ndata['y'][test_mask]

            all_preds = torch.cat((all_preds, masked_pred.cpu()), 0)
            all_labels = torch.cat((all_labels, masked_target.cpu()), 0)

    [Spearman, Pearson, R2, RMSE] = evaluate(all_preds.numpy().flatten(), all_labels.numpy().flatten())
    print('Test Spearman {:.4f}, Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}'.format(Spearman, Pearson, R2, RMSE))

    return [Spearman, Pearson, R2, RMSE]
    # results_test = pd.DataFrame(metrics, columns=['Spearman', 'Pearson', 'R2', 'RMSE'])
    # results_test.to_csv('test_results.csv')


if __name__ == '__main__':

    metrics = []
    for fold in range(5):
        print(f'\nFold: ', fold+1)
        train_data = torch.load('../data/res_split/' + str(fold+1) + '_sol_seed42.pt')  # seq_split, res_split
        train_dataset = NodeLevelPeptideDataset(train_data, edge_k)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dgl.batch)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=dgl.batch)
        test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=dgl.batch)
        
        model = ResidueGCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        history, best_model = train_val(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=epoches
        )
        torch.save(model.state_dict(), './model/Fold_' + str(fold+1) + '_res_best_model.pth')
        metrics.append(test(best_model, test_loader))
    
    [Spearman, Pearson, R2, RMSE] = np.mean(metrics, axis=0)
    [std1, std2, std3, std4] = np.std(metrics, axis=0)
    print('5-fold Test Results: Spearman {:.4f}, Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}'.format(Spearman, Pearson, R2, RMSE))
    print('            std:     Spearman {:.4f}, Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}'.format(std1, std2, std3, std4))

