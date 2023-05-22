from vrptw_cg import *
from utils import *

def NS_select(model,graph,isGPU=False):
    sigmoid = torch.nn.Sigmoid()
    x_loc = min_max_norm(torch.Tensor(list(graph.location.values())[1:-1]))
    x_d = min_max_norm(torch.Tensor(list(graph.demand.values())[1:-1]).unsqueeze(1))
    x_tw = min_max_norm(torch.Tensor([[list(graph.readyTime.values())[i],list(graph.dueTime.values())[i]] for i in range(1,graph.nodeNum-1)]))
    stDistance = min_max_norm(torch.Tensor([graph.disMatrix[key][1:-1] for key in range(1,graph.nodeNum-1)]))  
    adj = modify_adj(torch.Tensor([graph.adj[i][1:-1] for i in range(1,graph.nodeNum-1)]))
    x_dual = min_max_norm(torch.Tensor(graph.dualValue[1:-1]).reshape(-1,1))
    if isGPU: 
        x_loc,x_d,x_tw,x_dual,stDistance,adj = x_loc.cuda(),x_d.cuda(),x_tw.cuda(),x_dual.cuda(),stDistance.cuda(),adj.cuda() 
    output = model(x_loc.unsqueeze(0),x_d.unsqueeze(0),x_tw.unsqueeze(0),x_dual.unsqueeze(0),stDistance.unsqueeze(0),adj.unsqueeze(0))
    output_class = sigmoid(output)
    output_class = output_class[0].cpu()
    preds = [i+1 for i,val in enumerate(output_class) if val>=0.6]
    return preds


class VRPTW_CG_CM(VRPTW_CG):
    def __init__(self, graph, model, max_min_info, TimeLimit=2 * 60 * 60, SPPTimeLimit=3 * 60, SolCount=10, Max_iters=20000, isSave=False, SPP_alg='gp', initSol_alg='original', filename='', vehicleNum=50):
        super().__init__(graph, TimeLimit, SPPTimeLimit, SolCount, Max_iters, isSave, SPP_alg, initSol_alg, filename, vehicleNum)
        self.set_column_manage_model(model, max_min_info) 
        self.min_select_num = 100
        self.column_pool = []
        self.SPP_NS_model = torch.load('NSmodel\\model.pt').cpu() 
        self.NS_sate = True
        self.column_used_info = {}
 
    def set_column_manage_model(self, model, max_min_info):
        self.model = model 
        self.max_min_info = max_min_info 
    
    def standardize_state(self, state):
        for column_state in state["columns_state"]:
            for fi in range(len(column_state)):
                column_state[fi] = (column_state[fi] - self.max_min_info["column_state_min"][fi]) / (self.max_min_info["column_state_max"][fi] - self.max_min_info["column_state_min"][fi])
        for constraint_state in state["constraints_state"]:
            for fi in range(len(constraint_state)):
                constraint_state[fi] = (constraint_state[fi] - self.max_min_info["constraint_state_min"][fi]) / (self.max_min_info["constraint_state_max"][fi] - self.max_min_info["constraint_state_min"][fi])


    def column_manage(self, dualValue, solutionPool):
        """ get state """
        routes = [sol.path for sol in solutionPool]
        columns_state = []
        for route in routes:
            dual_sum = 0
            dist_sum = 0
            demand_sum = 0
            visit_num = len(route)
            visited = [0] * self.graph.nodeNum
            for i in range(1, len(route)):
                dual_sum += dualValue[route[i]]
                dist_sum += self.graph.disMatrix[route[i-1]][route[i]]
                demand_sum += self.graph.demand[route[i]]
                visited[route[i]] = 1
            state = [dual_sum, dist_sum, demand_sum] 
            columns_state.append(state)
        constraints_state = []
        for ni in range(self.graph.nodeNum):
            dual_value = dualValue[ni]
            coor_x, coor_y = self.graph.location[ni]
            demand = self.graph.demand[ni]
            ready_time = self.graph.readyTime[ni]
            due_time = self.graph.dueTime[ni]
            service_time = self.graph.serviceTime[ni]
            state = [dual_value, coor_x, coor_y, demand, ready_time, due_time] 
            constraints_state.append(state)
        edges = [[], []]
        for ri in range(len(routes)):
            for ni in routes[ri][1:]:
                edges[0].append(ri+self.graph.nodeNum) 
                edges[1].append(ni) 
        state = {
            "columns_state" : np.array(columns_state), 
            "constraints_state" : np.array(constraints_state), 
            "edges" : np.array(edges), 
        }
        self.standardize_state(state)

        """ select columns """
        select_result = np.ones(len(solutionPool))
        select_result = self.model(state)[:, 1].detach().numpy()
        if sum(select_result) == 0:
            select_result[0] = 1
        select_solutions = []
        unselect_solutions = []
        for i in range(len(solutionPool)):
            if select_result[i] >= 0.5:
                select_solutions.append(solutionPool[i])
            else:
                unselect_solutions.append(solutionPool[i])
        if len(select_solutions) < self.min_select_num:
            """ update dualValues """
            for sol in select_solutions:
                if sol.objVal < 0:
                    self.cg_count += 1 
                    self.add_new_column(sol)
            self.RMP.optimize()
            dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] 
            dualValue = np.array(dualValue)
            """ select still negative columns """
            objs = np.zeros(len(unselect_solutions))
            for i in range(len(unselect_solutions)):
                dual = 0
                dist = 0
                route = unselect_solutions[i].path
                for j in range(1, len(route)):
                    dual += dualValue[route[j]]
                    dist += self.graph.disMatrix[route[j-1]][route[j]]
                objs[i] = dist - dual
            solutionPool = [sol for i, sol in enumerate(unselect_solutions) if objs[i] < 0 and i+len(select_solutions)<=self.min_select_num]
        else:
            solutionPool = select_solutions
        return solutionPool
   
   
   
    def solve(self):  # sourcery skip: extract-duplicate-method, low-code-quality
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
        SPP.graph.dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] # update dual_value
        SPP.solve()
        while (SPP.solution.objVal+self.epsilon < 0 and 
               self.cg_count <= self.Max_iters and 
               time.time() - s_time < self.TimeLimit): # stop condition:
            self.iters += 1
            self.IterDualValueList[self.iters] = SPP.graph.dualValue
            self.iterColumns[self.iters] = []
                
            SPP.solutionPool = self.column_manage(SPP.graph.dualValue, SPP.solutionPool)  # to be implemented

            for sol in SPP.solutionPool:
                if sol.objVal < 0:
                    self.cg_count += 1 # need modify
                    self.add_new_column(sol)
            # record info
          
            self.updateBound(SPP)
            # if len(self.column_pool)%10==0 and self.iters > 10:
            #     self.column_drop()
            self.cost_time_list.append(time.time()-s_time)
            self.RMP_lb_list.append(round(self.LB,2))
            self.RMP_ub_list.append(round(self.UB,2))
            self.columnUsedCountList.append(self.columnUsedCount)
            print(f"cg_Num : {len(self.column_pool)} , cg_Used:{self.columnUsedCount}, lb: {round(self.LB,2)} , ub: {round(self.UB,2)} , SPPval: {round(SPP.solution.objVal,2)} , time cost:{time.time()-s_time}")
            
            rmp_time = time.time()
            self.RMP.optimize() 
            self.RMP_timeCost.append(time.time()-rmp_time)  # 精确记录RMP时间
            # solve subproblem 
            SPP.graph.dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] # update dual_value
            
            if self.NS_sate and self.iters > 10:
                self.NS_sate = False
                print("change to complete graph")
                
            if self.NS_sate:
                selected_nodes = NS_select(self.SPP_NS_model,SPP.graph)
                self.SPP_accelerate(SPP,selected_nodes)
                SPP.updateObj()
                spp_time = time.time()
                SPP.solve()
                self.SPP_timeCost.append(time.time()-spp_time )  # 精确记录SPP时间
                self.SPP_accelerate(SPP,selected_nodes,way='recover')
                if SPP.solution.objVal >= -1e-1:
                    SPP.updateObj()
                    spp_time = time.time()
                    SPP.solve()
                    self.SPP_timeCost.append(time.time()-spp_time )  # 精确记录SPP时间
                    self.NS_sate = False
                    self.NS_stop_iter = self.iters
                    print("change to complete graph")
            else:
                SPP.updateObj()
                spp_time = time.time()
                SPP.solve()
                self.SPP_timeCost.append(time.time()-spp_time )  # 精确记录SPP时间
                
            # SPP.updateObj()
            # spp_time = time.time()
            # SPP.solve()
            # self.SPP_timeCost.append(time.time()-spp_time)  # 精确记录SPP时间
            
        # solve with INTEGER VType
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.INTEGER)
        self.RMP.optimize()
        # result
        self.print_SolPool()
        e_time = time.time()
        print(f"time cost: {e_time-s_time}")
        res = {"cgNum":self.cg_count, 
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
                # "columnSet":{key:sol.__dict__ for key,sol in self.solution.solSet.items()},
        }
        self.result = res  ## result

    def SPP_accelerate(self,SPP,selected_nodes,way='reduce'):
        
        x_value = 0 if way in ['reduce'] else 1
        for nj in range(1,SPP.graph.nodeNum-1):
            if nj not in selected_nodes:
                for ni in range(SPP.graph.nodeNum-1):
                    if(ni != nj) and SPP.graph.arcs[ni,nj]==1:
                        SPP.x[(ni,nj)].UB = x_value
        SPP.SPP_model.update()
    
    
    def column_drop(self):
        self.column_pool.sort(key=lambda x: (self.column_used_info[x]['used'],self.column_used_info[x]['usedCount']), reverse=True) # from larger to small
        self.column_pool = self.column_pool[:1000]
        new_column_pool = []
        for col in self.column_pool:
            if self.column_used_info[col]['used'] >= -20:
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
    import sys
    # sys.path.append("D:\Code\RL-for-CS\CG_test")
    import torch
    import Net as Net
    net = Net.GAT4(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=torch.device("cpu"))
    actor = Net.Actor(net)
    actor.load_state_dict(torch.load("CGmanager\\actor_standard_GAT4.pth", map_location=torch.device('cpu')))
    standard_file = json.load(open("CGmanager\\mini_batches_standard_60.json", "r"))
    max_min_info = standard_file["max_min_info"]
    #
    datapath = 'GH_instance_1-10hard'  
    save_path = 'vrptw_cg_boosted_R'
    print('save_path:',save_path)
    for iter in range(80,90):
        for filename in tqdm(os.listdir(datapath)):
            savefile = str(iter)+ filename
            if savefile in os.listdir(save_path): continue
            if filename != 'R1_2_10.json': continue
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
            vrptw = VRPTW_CG_CM(graph, actor, max_min_info,
                            TimeLimit=2*60*60,  # 总的求解时间限制，默认
                            SPPTimeLimit=10*60,  # 子问题最大求解时间，默认
                            SPP_alg='gp', # gp 代表子问题用GUROBI求解,默认
                            initSol_alg='simple',   # simple即最简单的初始解，默认
                            filename=filename,
                            SolCount=200,   # 每次加不超过N列
                            Max_iters=20000,
                            isSave=True,
                            vehicleNum=50)  # 可以暂时不管，用来计算LB，但我们修改了时间窗，与原始的GH车辆不一样了，暂时不用它
            vrptw.solve()
            with open(os.path.join(save_path,savefile),'w') as f:
                json.dump(vrptw.result, f)





