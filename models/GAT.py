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
    def __init__(self, node_feature_dim, column_feature_dim, embed_dim, hidden_dim=128, heads=8, dropout=0.2, feature_gain=2):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.embed_dim = embed_dim
        self.feature_gain = feature_gain
        self.node_linear = nn.Linear(node_feature_dim*feature_gain, hidden_dim)
        self.column_linear = nn.Linear(column_feature_dim*feature_gain, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout) # node to column
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, dropout=dropout) # column to node
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), 
        )

    def forward(self, state):
        node_features = state["constraints_state"]
        column_features = state["columns_state"]
        edges = state["edges"]
        # transfer to tensor if type is not tensor
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float)
        if not isinstance(column_features, torch.Tensor):
            column_features = torch.tensor(column_features, dtype=torch.float)
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges, dtype=torch.long)
        # feature gain
        node_features = node_features.repeat(1, self.feature_gain)
        column_features = column_features.repeat(1, self.feature_gain)
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.conv1(embeddings, edges)) # node to column
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [1]))) # column to node
        logits = self.output(embeddings[-len(column_features):]) # get columns logits
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')