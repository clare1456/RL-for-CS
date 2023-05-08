'''
File: Net.py
Project: model
File Created: Monday, 8th May 2023 2:18:39 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Any, Dict, Optional, Sequence, Tuple, Union

class GAT4(nn.Module):
    def __init__(self, node_feature_dim, column_feature_dim, embed_dim, device, hidden_dim=256, heads=8, dropout=0.0):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.device = device
        self.embed_dim = embed_dim
        self.dual_feature_gain = 15
        self.feature_gain = 5
        self.node_linear = nn.Linear(self.dual_feature_gain+(node_feature_dim-1)*self.feature_gain, hidden_dim).to(device)
        self.column_linear = nn.Linear(self.dual_feature_gain+(column_feature_dim-1)*self.feature_gain, hidden_dim).to(device)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout).to(device) # node to column
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, dropout=dropout).to(device) # column to node
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim).to(device), 
        )
        # randomly init parameters
        # for param in self.parameters():
        #     torch.nn.init.normal_(param)

    def forward(self, state):
        node_features = state["constraints_state"]
        column_features = state["columns_state"]
        edges = state["edges"]
        # transfer to tensor if type is not tensor
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float).to(self.device)
        if not isinstance(column_features, torch.Tensor):
            column_features = torch.tensor(column_features, dtype=torch.float).to(self.device)
        if not isinstance(edges, torch.Tensor):
            edges = torch.tensor(edges, dtype=torch.long).to(self.device)
        # feature gain
        node_features = torch.cat(
            (node_features[:, :1].repeat(1, self.dual_feature_gain), node_features[:, 1:].repeat(1, self.feature_gain)),
            dim=1
        )
        column_features = torch.cat(
            (column_features[:, :1].repeat(1, self.dual_feature_gain), column_features[:, 1:].repeat(1, self.feature_gain)),
            dim=1
        )
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.conv1(embeddings, edges)) # column to node
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [0]))) # node to column
        logits = self.output(embeddings[-len(column_features):]) # get columns logits
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')

class Actor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_dim: int = 128,
        device: Union[str, int, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.input_dim = self.preprocess.embed_dim
        self.output_dim = 2
        self.hidden_dim = hidden_dim
        self.last = MLP(
            self.input_dim, 
            self.output_dim, 
            self.hidden_dim, 
            device=self.device
        )

    def forward(
        self,
        state: Union[np.ndarray, torch.Tensor],
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        embeddings = self.preprocess(state)
        logits = self.last(embeddings)
        probs = F.softmax(logits, dim=-1) 
        return probs

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device="cpu"):
        super().__init__()
        # 根据 hidden sizes 搭建神经网络
        self.device = device
        self.process = nn.Sequential(
            nn.Linear(input_dim, hidden_dim).to(device), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim).to(device), 
        )
        self.process.to(device=device)
    
    def forward(self, x):
        # 检查类型
        if isinstance(x, torch.Tensor) == False:
            x = torch.Tensor(x).to(self.device)
        # 输出结果
        return self.process(x)


