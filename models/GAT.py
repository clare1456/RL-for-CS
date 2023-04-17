'''
File: GAT.py
Project: models
File Created: Monday, 17th April 2023 2:55:17 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, node_feature_dim, column_feature_dim, hidden_dim=128, heads=8, dropout=0.2):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.node_linear = nn.Linear(node_feature_dim, hidden_dim)
        self.column_linear = nn.Linear(column_feature_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout) # node to column
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, dropout=dropout) # column to node
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, state):
        node_features = state["node_features"]
        column_features = state["column_features"]
        edges = state["edges"]
        # transfer to tensor if type is not tensor
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float)
        if not isinstance(column_features, torch.Tensor):
            column_features = torch.tensor(column_features, dtype=torch.float)
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges, dtype=torch.long)
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.conv1(embeddings, edges)) # node to column
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [1]))) # column to node
        logits = self.output(embeddings[:self.args.node_num]) # get node logits
        return logits.squeeze(1)

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')