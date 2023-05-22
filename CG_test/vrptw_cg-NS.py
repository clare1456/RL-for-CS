
from espprc_gp import ESPPRC_gp
from gurobipy import GRB,Model,Column
from GetData import *
import time,json
from solution import *
from vrptw_initSol import *
from tqdm import tqdm
import numpy as np
from SPP_accelerate import *
from model_efgat import *
from vrptw_cg import *

def min_max_norm(data:torch.tensor,dim=1): 
    m,n = data.shape
    if dim==1:
        for i in range(n):
            data[:, i] = (data[:, i] - min(data[:, i])) / (max(data[:, i]) - min(data[:, i]))
    else:
        for i in range(m):
            data[i, :] = (data[i, :] - min(data[i, :])) / (max(data[i, :]) - min(data[i, :]))
    return data

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
      
      
class VRPTW_CG_NS(VRPTW_CG):
    def __init__(self,
                 graph,
                 TimeLimit=2*60*60,
                 SPPTimeLimit=3*60,
                 SolCount=10,
                 Max_iters=20000,
                 isSave=False,
                 SPP_alg='gp',
                 initSol_alg='original',
                 filename='',
                 vehicleNum=50):
        super().__init__(graph, TimeLimit, SPPTimeLimit, SolCount, Max_iters, isSave, SPP_alg, initSol_alg, filename, vehicleNum)
        
        self.column_pool = []
        self.SPP_NS_model = torch.load('NSmodel\\model.pt').cpu() # column selection model
        self.NS_sate = True
        self.column_used_info = {}
        self.NS_stop_iter = 0
        

        
    def solve(self):  
        s_time = time.time()
        self.RMP,self.rmp_constraints = self.build_RMP_gp()
        init_solution_vrptw = self.get_vrptw_initsol(self.filename)
        self.iterColumns[self.iters] = []
        for init_cg in init_solution_vrptw.values():
            self.cg_count += 1 
            self.add_new_column(init_cg)
        self.initColumnNum = len(init_solution_vrptw)
        print("init columns number:",len(init_solution_vrptw))
        self.RMP.setParam("OutputFlag", 0)
        self.RMP.optimize()
        # set SPP
        SPP = self.set_SPP_alg(self.SPP_alg)
        SPP.isPrint = False
        SPP.graph.dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] 
        SPP.solve()
        while (SPP.solution.objVal+self.epsilon < 0 and 
               self.cg_count <= self.Max_iters and 
               time.time() - s_time < self.TimeLimit): 
            self.iters += 1
            self.IterDualValueList[self.iters] = SPP.graph.dualValue
            self.iterColumns[self.iters] = []
            for sol in SPP.solutionPool:
                if sol.objVal < 0:
                    self.cg_count += 1 
                    self.add_new_column(sol)
            # record info
            self.updateBound(SPP)
            self.cost_time_list.append(time.time()-s_time)
            self.RMP_lb_list.append(round(self.LB,2))
            self.RMP_ub_list.append(round(self.UB,2))
            self.columnUsedCountList.append(self.columnUsedCount)
            print(f"cg_Num : {len(self.column_pool)} , cg_Used:{self.columnUsedCount}, lb: {round(self.LB,2)} , ub: {round(self.UB,2)} , SPPval: {round(SPP.solution.objVal,2)} , time cost:{time.time()-s_time}")
        
            rmp_time = time.time()
            self.RMP.optimize() 
            self.RMP_timeCost.append(time.time()-rmp_time )
            # solve subproblem 
            SPP.graph.dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] # update dual_value
            
            if self.NS_sate:
                selected_nodes = NS_select(self.SPP_NS_model,SPP.graph)
                self.SPP_accelerate(SPP,selected_nodes)
                SPP.updateObj()
                spp_time = time.time()
                SPP.solve()
                self.SPP_timeCost.append(time.time()-spp_time )  
                self.SPP_accelerate(SPP,selected_nodes,way='recover')
                if SPP.solution.objVal >= -1:
                    SPP.updateObj()
                    spp_time = time.time()
                    SPP.solve()
                    self.SPP_timeCost.append(time.time()-spp_time )  
                    self.NS_sate = False
                    self.NS_stop_iter = self.iters
                    print("change to complete graph")
            else:
                SPP.updateObj()
                spp_time = time.time()
                SPP.solve()
                self.SPP_timeCost.append(time.time()-spp_time )  
           
        # solve with INTEGER VType
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.INTEGER)
        self.RMP.optimize()
        # result
        self.print_SolPool()
        e_time = time.time()
        print(f"time cost: {e_time-s_time}")
        res = {"cgNum":self.cg_count, 
               "NS_stop_iter":self.NS_stop_iter,
                "initColumnNum":self.initColumnNum,
                "objVal": self.RMP.objVal,
                "total_cost_time":e_time-s_time, 
                "RMP_timeCost":self.RMP_timeCost,
                "SPP_timeCost":self.SPP_timeCost,
                "columnUsedNum":self.columnUsedCountList,
                "bestColumns":self.bestColumnName,
                "IterColumns":self.iterColumns,
                "IterDualValue":self.IterDualValueList,
                "cost_time_list":self.cost_time_list,
                "RMP_lb_list":self.RMP_lb_list, 
                "RMP_ub_list":self.RMP_ub_list, 
                "SPP_objVal_list": self.SPP_objVal_list,
        }
        self.result = res  
              
    def SPP_accelerate(self,SPP,selected_nodes,way='reduce'):
        x_value = 0 if way in ['reduce'] else 1
        for nj in range(1,SPP.graph.nodeNum-1):
            if nj not in selected_nodes:
                for ni in range(SPP.graph.nodeNum-1):
                    if(ni != nj) and SPP.graph.arcs[ni,nj]==1:
                        SPP.x[(ni,nj)].UB = x_value
        SPP.SPP_model.update()
        

    def column_manage(self):
        new_column_pool = []
        for col in self.column_pool:
            if self.column_used_info[col]['used'] >= 0:
                new_column_pool.append(col)
        self.column_pool = new_column_pool[:]
        for var in self.RMP.getVars():
            if var.VarName not in self.column_pool:
                self.RMP.remove(var)
        self.RMP.update()
        
                  
    def updateBound(self,SPP):
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.INTEGER)  
        self.RMP.optimize()
        self.UB = self.RMP.objVal
        self.columnUsedCount = len([1 for var in self.RMP.getVars() if 1-var.x<1e-3])   
        for var in self.RMP.getVars():
            if var.VarName not in self.column_used_info.keys():
                self.column_used_info[var.VarName] = {"used":0,'usedCount':0}
            if 1-var.x < 1e-3:
                self.column_used_info[var.VarName]['used'] = 10
                self.column_used_info[var.VarName]['usedCount'] += 1
            else:
                self.column_used_info[var.VarName]['used'] -= 1
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.CONTINUOUS)  
        self.RMP.optimize()
        self.LB = max(self.RMP.objVal + min(self.columnUsedCount * SPP.solution.objVal,0), 0)
               
    
    def update_RMP_Constr(self, new_path:list, path_distance, column_name):
        # creat new column 
        col_coef = [1 if i+1 in new_path else 0 for i in range(self.graph.customerNum)]
        rmp_col = Column(col_coef, self.rmp_constraints) 
        self.cg_vars[column_name] = self.RMP.addVar(lb = 0.0, ub = 1, obj = path_distance, vtype = GRB.CONTINUOUS, column = rmp_col,name=column_name)
        self.column_pool.append(column_name)
        self.RMP.update() 


        
if __name__=='__main__':
    import os
    
    datapath = 'GH_instance_1-10hard'  
    save_path = 'vrptw_cg_NS'
    print('save_path:',save_path)
    for iter in range(10,20):
        for filename in tqdm(os.listdir(datapath)):
            savefile = str(iter)+ filename
            if savefile in os.listdir(save_path): continue
            if filename[:2] != 'RC': continue
            print(f'\n {savefile}')
            filepath = os.path.join(datapath,filename)
            with open(filepath,"r") as f:
                data = json.load(f)
            graph = Data()   
            graph.__dict__ = data
            graph.demand = {int(key):val for key,val in graph.demand.items()}
            graph.location = {int(key):val for key,val in graph.location.items()}
            graph.serviceTime = {int(key):val for key,val in graph.serviceTime.items()}
            graph.readyTime = {int(key):val for key,val in graph.readyTime.items()}
            graph.dueTime = {int(key):val for key,val in graph.dueTime.items()}
            graph.disMatrix = {int(key):val for key,val in graph.disMatrix.items()}
            graph.feasibleNodeSet = {int(key):val for key,val in graph.feasibleNodeSet.items()}   
            vrptw = VRPTW_CG_NS(graph,
                            TimeLimit=2*60*60,  
                            SPPTimeLimit=10*60,  
                            SPP_alg='gp', 
                            initSol_alg='simple',   
                            filename=filename,
                            SolCount=200,   
                            Max_iters=20000,
                            isSave=True,
                            vehicleNum=50)  
            vrptw.solve()
            with open(os.path.join(save_path,savefile),'w') as f:
                json.dump(vrptw.result, f)


