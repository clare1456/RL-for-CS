import torch
import torch.nn as nn
import torch.nn.functional as F



class GAT_EFA(nn.Module):
    def __init__(self, nfeat, nedgef, nhid=64, nclass=2, dropout=0.1, 
                 alpha=0.2, nheads=8, noutheads=4, nlayer=2,
                 feature_gain=10,maxNodeNum=20):
        """Dense version of GAT.
            nfeat  N*F
            nedgef  N*N*F
        """
        super(GAT_EFA, self).__init__()
        self.dropout = dropout
        self.nlayer = nlayer
        self.nheads = nheads
        self.noutheads = noutheads
        self.feature_gain = feature_gain

        self.node_w1 = nn.Linear(2*feature_gain, nfeat//4)
        self.node_w2 = nn.Linear(1*feature_gain, nfeat//4)
        self.node_w3 = nn.Linear(2*feature_gain, nfeat//4)
        self.node_w4 = nn.Linear(1*feature_gain, nfeat//4)
        self.edge_w1 = nn.Linear(1*feature_gain, nedgef//2)
        self.edge_w2 = nn.Linear(1*feature_gain, nedgef//2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.node_embedding = nn.Linear(nfeat,nfeat)
        self.edge_embedding = nn.Linear(nedgef,nedgef)
        self.reduced_cost_w = nn.Linear(1*feature_gain, nedgef//2)
        self.distance_w = nn.Linear(1*feature_gain, nedgef//2)
        self.path_embedding = nn.Linear(maxNodeNum,1)
        self.path_w = nn.Linear(1*feature_gain, nedgef)
        self.column_embedding = nn.Linear(nedgef*2,nedgef)
        
        self.x_weight1 = nn.Linear(nhid,1)
        self.x_weight2 = nn.Linear(maxNodeNum,nedgef)
        
        self.attentions = [[GraphAttentionLayer_EFA(nfeat, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)]]

        for i, attention in enumerate(self.attentions[0]):
            self.add_module('attention_{}_{}'.format(0, i), attention)
        #attention layers
        for j in range(1, nlayer-1):
            self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha,lastact=True) for _ in range(nheads)])
            for i, attention in enumerate(self.attentions[j]):
                self.add_module('attention_{}_{}'.format(j, i), attention)
        #last attention layer
        self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha,lastact=False) for _ in range(noutheads)])
        for i, attention in enumerate(self.attentions[nlayer-1]):
            self.add_module('attention_{}_{}'.format(nlayer-1, i), attention)
        #output layer
        self.out_layer1 = nn.Linear(nhid*4, nhid*2)
        self.out_layer2 = nn.Linear(nhid*2, nclass)
  

    def forward(self,x_loc,x_tw,x_d,dismatrix,adj,x_dual,reduced_cost,distance,path_onehot):
        # x_c, x_tw, x_d, x_disMatrix, x_adj  = x_graph
        # dim,obj,distance,path,dualValue = x_column
        # graph features
        x_loc = x_loc.repeat(1,1,self.feature_gain) # 特征增强
        x_tw = x_tw.repeat(1,1,self.feature_gain)
        x_d = x_d.repeat(1,1,self.feature_gain)
        x_dual = x_dual.repeat(1,1,self.feature_gain)
        x_loc = self.relu(self.node_w1(x_loc))
        x_tw = self.relu(self.node_w3(x_tw))
        x_d = self.relu(self.node_w2(x_d))
        x_dual = self.relu(self.node_w4(x_dual))
        x = torch.cat((x_loc,x_d,x_tw,x_dual), dim=-1) 
        x_edge = dismatrix.unsqueeze(-1).repeat(1,1,1,self.feature_gain)
        x_adj = adj.unsqueeze(-1).repeat(1,1,1,self.feature_gain)
        edge_dis = self.relu(self.edge_w1(x_edge))
        edge_adj = self.relu(self.edge_w2(x_adj))
        edge_feats = torch.cat((edge_dis,edge_adj), dim=-1)
        # feature
        node_feats = self.relu(self.node_embedding(x))
        edge_feats = self.relu(self.edge_embedding(edge_feats))
        #input_layer
        x = torch.cat([att(node_feats, edge_feats, adj) for att in self.attentions[0]], dim=-1)
        x = self.dropout(x)
        #hidden layer
        for j in range(1, self.nlayer-1):
            mid = torch.cat([att(x, edge_feats, adj) for att in self.attentions[j]], dim=-1)
            x = mid + x  #residual connections
            x = self.dropout(x)
        
        #last hidden layer
        x = torch.mean(torch.stack([att(x, edge_feats, adj) for att in self.attentions[self.nlayer-1]]), 0)
        x = self.relu(x)  #h_i=δ(avg(∑ α_ij·Wh·h_j))
        x = self.dropout(x)

        x = self.relu(self.x_weight1(x)) 
        x = self.relu(self.x_weight2(x.squeeze(-1))) 
        x = self.dropout(x)
        #  column features
        x_reduced_cost = reduced_cost.repeat(1,self.feature_gain) 
        x_distance = distance.repeat(1,self.feature_gain) 
        x_path = self.relu(self.path_embedding(path_onehot))
        x_path = x_path.repeat(1,self.feature_gain)
        x_reduced_cost = self.relu(self.reduced_cost_w(x_reduced_cost))
        x_distance = self.relu(self.distance_w(x_distance))
        x_path = self.relu(self.path_w(x_path))
        x_column = torch.cat((x_reduced_cost,x_distance,x_path), dim=-1) 
        x_column = self.relu(self.column_embedding(x_column))
        x = torch.cat((x,x_column), dim=-1) 
        #output layer
        x = self.dropout(self.relu(self.out_layer1(x)))
        x = self.out_layer2(x)
        
        return self.softmax(x)


class GAT_EFA_Net(GAT_EFA):
    def __init__(self, nfeat, nedgef, embed_dim, nhid=64, nclass=2, dropout=0.1, 
                 alpha=0.2, nheads=8, noutheads=4, nlayer=2,
                 feature_gain=10,maxNodeNum=20):
        """Dense version of GAT.
            nfeat  N*F
            nedgef  N*N*F
        """
        super(GAT_EFA_Net, self).__init__(nfeat, nedgef, nhid, nclass, dropout,
                 alpha, nheads, noutheads, nlayer,
                 feature_gain,maxNodeNum)
        # redefine output layer
        self.out_layer = nn.Linear(nhid*4, embed_dim)
  
    def forward(self,x_loc,x_tw,x_d,dismatrix,adj,x_dual,reduced_cost,distance,path_onehot):
        # x_c, x_tw, x_d, x_disMatrix, x_adj  = x_graph
        # dim,obj,distance,path,dualValue = x_column
        # graph features
        x_loc = x_loc.repeat(1,1,self.feature_gain) # 特征增强
        x_tw = x_tw.repeat(1,1,self.feature_gain)
        x_d = x_d.repeat(1,1,self.feature_gain)
        x_dual = x_dual.repeat(1,1,self.feature_gain)
        x_loc = self.relu(self.node_w1(x_loc))
        x_tw = self.relu(self.node_w3(x_tw))
        x_d = self.relu(self.node_w2(x_d))
        x_dual = self.relu(self.node_w4(x_dual))
        x = torch.cat((x_loc,x_d,x_tw,x_dual), dim=-1) 
        x_edge = dismatrix.unsqueeze(-1).repeat(1,1,1,self.feature_gain)
        x_adj = adj.unsqueeze(-1).repeat(1,1,1,self.feature_gain)
        edge_dis = self.relu(self.edge_w1(x_edge))
        edge_adj = self.relu(self.edge_w2(x_adj))
        edge_feats = torch.cat((edge_dis,edge_adj), dim=-1)
        # feature
        node_feats = self.relu(self.node_embedding(x))
        edge_feats = self.relu(self.edge_embedding(edge_feats))
        #input_layer
        x = torch.cat([att(node_feats, edge_feats, adj) for att in self.attentions[0]], dim=-1)
        x = self.dropout(x)
        #hidden layer
        for j in range(1, self.nlayer-1):
            mid = torch.cat([att(x, edge_feats, adj) for att in self.attentions[j]], dim=-1)
            x = mid + x  #residual connections
            x = self.dropout(x)
        
        # last hidden layer
        x = torch.mean(torch.stack([att(x, edge_feats, adj) for att in self.attentions[self.nlayer-1]]), 0)
        x = self.relu(x)  #h_i=δ(avg(∑ α_ij·Wh·h_j))
        x = self.dropout(x)

        x = self.relu(self.x_weight1(x)) 
        x = self.relu(self.x_weight2(x.squeeze(-1))) 
        x = self.dropout(x)
        # column features
        x_reduced_cost = reduced_cost.repeat(1,self.feature_gain) 
        x_distance = distance.repeat(1,self.feature_gain) 
        x_path = self.relu(self.path_embedding(path_onehot))
        x_path = x_path.repeat(1,self.feature_gain)
        x_reduced_cost = self.relu(self.reduced_cost_w(x_reduced_cost))
        x_distance = self.relu(self.distance_w(x_distance))
        x_path = self.relu(self.path_w(x_path))
        x_column = torch.cat((x_reduced_cost,x_distance,x_path), dim=-1) 
        x_column = self.relu(self.column_embedding(x_column))
        x = torch.cat((x,x_column), dim=-1) 
        #output layer #! redefine output layer
        x = self.out_layer(x)
        
        return x


class GraphAttentionLayer_EFA(nn.Module):
    """
    GAT + EFA layer
    """

    def __init__(self, in_features, in_edge_features, out_features, dropout, alpha,lastact=False):
        #flowchart GAT
        super(GraphAttentionLayer_EFA, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.lastact = lastact
        # self.bn = torch.nn.BatchNorm1d(nodenum)
        self.activation = nn.LeakyReLU(self.alpha)
        
        self.wh = nn.Linear(in_features,out_features)
        self.wh1 = nn.Linear(in_features,out_features)
        self.wh2 = nn.Linear(in_features,out_features)
        
        self.ah = nn.Linear(out_features,1)
        self.wf = nn.Linear(in_edge_features,out_features)
        self.af = nn.Linear(out_features,1)
        # self.bf = nn.Linear(1,out_features)
        self.bf = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.bf.data, gain=1.414)

        

    def forward(self, input, edge_feat, adj):
        #compute h = input * W_h
        # h = torch.mm(input, self.Wh)  #input: num*in_size, W:in_size*out_size, h:num*out_size
        h = self.wh(input)
        bs,N,_ = h.size()  #=stroke_num
        #compute cij
        h1 = self.wh1(input)
        h2 = self.wh2(input)
        # h1 = torch.mm(input, self.Wh1)
        # h2 = torch.mm(input, self.Wh2)
        ah_input = h1.repeat(1,1, N).view(bs,N * N, -1) + h2.repeat(1,N, 1)      #W_h*h_i + W_h*H_j
        ah_input = ah_input.view(bs, N, -1, self.out_features)              #N*N*32
        # c = self.activation(torch.matmul(ah_input, self.ah).squeeze(2))  #N*N*32 · 32*1 = N*N*1-->N*N
        c = self.activation(self.ah(ah_input).squeeze(-1))  #N*N*32 · 32*1 = N*N*1-->N*N
        
        #compute c'ij
        # input_edge = edge_feat.unsqueeze(-1)            #N*N*1*19
        f = self.wf(edge_feat)               #N*N*1*32           #=W_f·f_ij
        f = f + self.bf                                #N*N*1*32+N*N*1*32  #=W_f·f_ij + b_f
        af_input = self.activation(f)                   #N*N*1*32           #=δ(W_f·f_ij + b_f)
        cp = self.af(af_input).squeeze(-1)     #N*N*1              #=a_f·δ(W_f·f_ij + b_f)
        cp = self.activation(cp)             #N*N                #=δ(a_f·δ(W_f·f_ij + b_f)))
        c = c + cp      #compute cij & c'ij
        
        #compute output = h * attention adj matrix
        zero_vec = -9e15*torch.ones_like(c)      #ones_like：返回大小与input相同的1张量
        attention = torch.where(adj>0, c, zero_vec)  
        attention = F.softmax(attention, dim=-1)  #α_ij
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #原有dropout
        h_prime = torch.matmul(attention, h)     #=∑ α_ij · Wh · h_j 
        
        # h_prime1 = self.bn(h_prime)
        
        mean_h = h_prime.mean(axis=(0,2)).unsqueeze(-1)
        std_h = h_prime.var(axis=(0,2),unbiased=False).unsqueeze(-1)
        h_prime = (h_prime - mean_h)/((1e-5 + std_h).sqrt())
        
        if self.lastact == True:
            return self.activation(h_prime)  #=δ(∑ α_ij·Wh·h_j)
        else:
            return h_prime  #=∑ α_ij·Wh·h_j
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ ==  '__main__':
    import random
    nodeNum = 20
    batch = 8
    x_c = torch.randn(nodeNum*2*batch).reshape(batch,nodeNum,2)
    x_tw = torch.randn(nodeNum*2*batch).reshape(batch,nodeNum,2)
    x_d = torch.randn(nodeNum*batch).reshape(batch,nodeNum,1)
    x_dual = torch.randn(nodeNum*batch).reshape(batch,nodeNum,1)
    dismatrix = torch.randn(nodeNum**2*batch).reshape(batch,nodeNum, nodeNum)
    adj = torch.randn(nodeNum**2*batch).reshape(batch,nodeNum, -1)
    reduced_cost = torch.randn(1*batch).reshape(batch,1)
    distance = torch.randn(1*batch).reshape(batch,1)
    path_onehot = torch.randint(0,2,(nodeNum*batch,1),dtype=torch.float32).reshape(batch,nodeNum)

    model = GAT_EFA(nfeat=128, nedgef=128)
    res = model(x_c,x_tw,x_d,dismatrix,adj,x_dual,reduced_cost,distance,path_onehot)
    print(res)
    print(res.shape)
    

