import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# origin version
class GAT(nn.Module):
    def __init__(self, node_feature_dim, column_feature_dim, embed_dim, device, hidden_dim=256, heads=8, dropout=0.0, feature_gain=2):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.device = device
        self.embed_dim = embed_dim
        self.feature_gain = feature_gain
        self.node_linear = nn.Linear(node_feature_dim*feature_gain, hidden_dim).to(device)
        self.column_linear = nn.Linear(column_feature_dim*feature_gain, hidden_dim).to(device)
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
        node_features = node_features.repeat(1, self.feature_gain)
        column_features = column_features.repeat(1, self.feature_gain)
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

# thick version
class GAT2(nn.Module):
    def __init__(self, node_feature_dim, column_feature_dim, embed_dim, device, hidden_dim=256, heads=8, dropout=0.0, feature_gain=2):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.device = device
        self.embed_dim = embed_dim
        self.feature_gain = feature_gain
        self.node_linear = nn.Linear(node_feature_dim*feature_gain, hidden_dim).to(device)
        self.column_linear = nn.Linear(column_feature_dim*feature_gain, hidden_dim).to(device)
        self.cat_linear = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout).to(device) # node to column
        self.conv1_linear = nn.Linear(hidden_dim*heads, hidden_dim).to(device)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout).to(device) # column to node
        self.conv2_linear = nn.Linear(hidden_dim*heads, hidden_dim).to(device)
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
        node_features = node_features.repeat(1, self.feature_gain)
        column_features = column_features.repeat(1, self.feature_gain)
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.cat_linear(embeddings))
        embeddings = F.relu(self.conv1(embeddings, edges)) # column to node
        embeddings = F.relu(self.conv1_linear(embeddings)) 
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [0]))) # node to column
        embeddings = F.relu(self.conv2_linear(embeddings)) 
        logits = self.output(embeddings[-len(column_features):]) # get columns logits
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')

# deep version
class GAT3(nn.Module):
    def __init__(self, node_feature_dim, column_feature_dim, embed_dim, device, hidden_dim=256, heads=8, dropout=0.0, feature_gain=2):
        super().__init__()
        self.name = "GAT"
        # build network 
        self.device = device
        self.embed_dim = embed_dim
        self.feature_gain = feature_gain
        self.node_linear = nn.Linear(node_feature_dim*feature_gain, hidden_dim).to(device)
        self.column_linear = nn.Linear(column_feature_dim*feature_gain, hidden_dim).to(device)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout).to(device) # node to column
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, dropout=dropout).to(device) # column to node
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout).to(device) # node to column
        self.conv4 = GATConv(hidden_dim*heads, hidden_dim, dropout=dropout).to(device) # column to node
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
        node_features = node_features.repeat(1, self.feature_gain)
        column_features = column_features.repeat(1, self.feature_gain)
        # network calculation 
        node_embeddings = F.relu(self.node_linear(node_features))
        column_embeddings = F.relu(self.column_linear(column_features))
        # embedding concat
        embeddings = torch.cat([node_embeddings, column_embeddings], dim=0)
        embeddings = F.relu(self.conv1(embeddings, edges)) # column to node
        embeddings = F.relu(self.conv2(embeddings, torch.flip(edges, [0]))) # node to column
        embeddings = F.relu(self.conv3(embeddings, edges)) # column to node
        embeddings = F.relu(self.conv4(embeddings, torch.flip(edges, [0]))) # node to column
        logits = self.output(embeddings[-len(column_features):]) # get columns logits
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path + self.name + '.pth')

    def load_model(self, path):
        torch.load(path + self.name + '.pth')

# strengthen dual
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


