import torch
from utils import *


def NS_select(model,graph,isGPU=False):
    sigmoid = torch.nn.Sigmoid()
    x_loc = min_max_norm(torch.Tensor(list(graph.location.values())[1:-1]))
    x_d = min_max_norm(torch.Tensor(list(graph.demand.values())[1:-1]).unsqueeze(1))
    x_tw = min_max_norm(torch.Tensor([[list(graph.readyTime.values())[i],list(graph.dueTime.values())[i]] for i in range(1,graph.nodeNum-1)]))
    stDistance = min_max_norm(torch.Tensor([graph.disMatrix[key][1:-1] for key in range(1,graph.nodeNum-1)]))  # normalize
    adj = modify_adj(torch.Tensor([graph.adj[i][1:-1] for i in range(1,graph.nodeNum-1)]))
    x_dual = min_max_norm(torch.Tensor(graph.dualValue[1:-1]).reshape(-1,1))
    if isGPU: # if cuda
        x_loc,x_d,x_tw,x_dual,stDistance,adj = x_loc.cuda(),x_d.cuda(),x_tw.cuda(),x_dual.cuda(),stDistance.cuda(),adj.cuda() 
    output = model(x_loc.unsqueeze(0),x_d.unsqueeze(0),x_tw.unsqueeze(0),x_dual.unsqueeze(0),stDistance.unsqueeze(0),adj.unsqueeze(0))
    output_class = sigmoid(output)
    output_class = output_class[0].cpu()
    preds = [i+1 for i,val in enumerate(output_class) if val>=0.5]
    return preds
      





