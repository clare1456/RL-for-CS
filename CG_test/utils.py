
import torch,json

def modify_adj(A):
    for i in range(A.size(1)):
        A[i,i] = -1
    return A

def min_max_norm(data:torch.tensor,dim=1): 
    m,n = data.shape
    if dim==1:
        for i in range(n):
            data[:, i] = (data[:, i] - min(data[:, i])) / (max(data[:, i]) - min(data[:, i]))
    else:
        for i in range(m):
            data[i, :] = (data[i, :] - min(data[i, :])) / (max(data[i, :]) - min(data[i, :]))
    return data







def is_optmal(pre_support_nodes:list,label:list):  
    for i in label:
        if i not in pre_support_nodes:
            return False
    return True


def model_test(model_path=r'save\model.pt',data_path=r'trainingdata\class_trainingdatapool\nclass=4\labeled_c50_0-100.json'):
    """
    模型 预测 测试函数
    return 
    """
    model = torch.load(model_path)
    model.cpu()
    with open(data_path, "r") as f:
        data = json.load(f)
        f.close()
    num_data = len(data.keys())
    acc_class3,acc_class32 =0,0
    for key in data.keys():
        g = data[key]
        x_c = min_max_norm(torch.Tensor(g["location"]))
        x_d = min_max_norm(torch.Tensor(g["demand"]).view(-1,1))
        stDisMatrix = torch.Tensor([g["stdisMatrix"][key] for key in g["stdisMatrix"].keys()])
        y_label = g["label_class"]
        support_nodes =  torch.Tensor(g["Path"][1:-1])
        adj = torch.Tensor(g["adj"])

        output = model(x_c, x_d, stDisMatrix,adj)   
        class_output = output.argmax(1)
        pre_support_nodes_class3 = [n for n in range(class_output.shape[0]) if class_output[n]==3]
        pre_support_nodes_class2 = [n for n in range(class_output.shape[0]) if class_output[n]==2]
        pre_support_nodes_class1 = [n for n in range(class_output.shape[0]) if class_output[n]==1]
        if is_optmal(pre_support_nodes_class3,support_nodes):
            acc_class3 += 1
        elif is_optmal(pre_support_nodes_class3+pre_support_nodes_class2,support_nodes):
            acc_class32 += 1
  
    return acc_class3,acc_class32,num_data
        


                                   

           
      
      
