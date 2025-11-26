# Peptide-specifc MCP predictor
# Ruihan Dong, 25-11-19

import torch
import torch.nn as nn
import dgl

class ResidueGCN(torch.nn.Module):
    
    def __init__(self, input_dim=960, hidden_dim=512, num_layers=3):
        super(ResidueGCN, self).__init__()
        
        # GCN
        self.layers = torch.nn.ModuleList([
            dgl.nn.SAGEConv(input_dim, hidden_dim, 'pool'),
            dgl.nn.SAGEConv(hidden_dim, hidden_dim, 'pool'),
            dgl.nn.SAGEConv(hidden_dim, 1, 'pool')
        ])

        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g):
        h = g.ndata['h'].float()
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if i < len(self.layers) - 1:
                h = self.drop(self.act(h))

        pred = self.sigmoid(h)  # [N_nodes, 1]
        
        g.ndata['pred'] = pred.squeeze()
        return g
    

'''model zoo
https://dgl.ac.cn/dgl_docs/api/python/nn-pytorch.html

GCN
dgl.nn.GraphConv(input_dim, hidden_dim),
            dgl.nn.GraphConv(hidden_dim, hidden_dim),
            dgl.nn.GraphConv(hidden_dim, 1)

GAT, num_heads
dgl.nn.GATConv(input_dim, hidden_dim, num_heads=1),
            dgl.nn.GATConv(hidden_dim, hidden_dim, num_heads=1),
            dgl.nn.GATConv(hidden_dim, 1, num_heads=1)

GraphSAGE, aggregator (mean, pool)
dgl.nn.SAGEConv(input_dim, hidden_dim, 'pool'),
            dgl.nn.SAGEConv(hidden_dim, hidden_dim, 'pool'),
            dgl.nn.SAGEConv(hidden_dim, 1, 'pool')

TAG, k
dgl.nn.TAGConv(input_dim, hidden_dim, k=4),
            dgl.nn.TAGConv(hidden_dim, hidden_dim, k=4),
            dgl.nn.TAGConv(hidden_dim, 1,  k=4)


'''