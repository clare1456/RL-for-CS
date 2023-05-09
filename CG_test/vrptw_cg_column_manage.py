'''
File: vrptw_cg_column_manage.py
Project: CG_test
File Created: Monday, 8th May 2023 2:24:05 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

from vrptw_cg import *

class VRPTW_CG_CM(VRPTW_CG):
    def __init__(self, graph, model, max_min_info, TimeLimit=2 * 60 * 60, SPPTimeLimit=3 * 60, SolCount=10, Max_iters=20000, isSave=False, SPP_alg='gp', initSol_alg='original', filename='', vehicleNum=50):
        super().__init__(graph, TimeLimit, SPPTimeLimit, SolCount, Max_iters, isSave, SPP_alg, initSol_alg, filename, vehicleNum)
        self.set_column_manage_model(model, max_min_info) 
        self.min_select_num = 100
 
    def set_column_manage_model(self, model, max_min_info):
        self.model = model # column selection model
        self.max_min_info = max_min_info # max and min info of columns
    
    def standardize_state(self, state):
        for column_state in state["columns_state"]:
            for fi in range(len(column_state)):
                column_state[fi] = (column_state[fi] - self.max_min_info["column_state_min"][fi]) / (self.max_min_info["column_state_max"][fi] - self.max_min_info["column_state_min"][fi])
        for constraint_state in state["constraints_state"]:
            for fi in range(len(constraint_state)):
                constraint_state[fi] = (constraint_state[fi] - self.max_min_info["constraint_state_min"][fi]) / (self.max_min_info["constraint_state_max"][fi] - self.max_min_info["constraint_state_min"][fi])

    def column_manage(self, dualValue, solutionPool):
        """ get state """
        # 1. get states of columns
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
            state = [dual_sum, dist_sum, demand_sum] # dim = (len(columns), 3)
            columns_state.append(state)
        # 2. get states of nodes
        constraints_state = []
        for ni in range(self.graph.nodeNum):
            dual_value = dualValue[ni]
            coor_x, coor_y = self.graph.location[ni]
            demand = self.graph.demand[ni]
            ready_time = self.graph.readyTime[ni]
            due_time = self.graph.dueTime[ni]
            service_time = self.graph.serviceTime[ni]
            state = [dual_value, coor_x, coor_y, demand, ready_time, due_time] # dim = (len(constraints), 6)
            constraints_state.append(state)
        # 3. get edges
        edges = [[], []]
        for ri in range(len(routes)):
            for ni in routes[ri][1:]:
                edges[0].append(ri+self.graph.nodeNum) # dim = (2, len(constraints) * len(columns))
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
                    self.cg_count += 1 # need modify
                    self.add_new_column(sol)
            self.RMP.optimize()
            dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] # update dual_value
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
            solutionPool = [sol for i, sol in enumerate(unselect_solutions) if objs[i] < 0]
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
            if self.isSave:
                self.updateBound(SPP)
                self.cost_time_list.append(time.time()-s_time)
                self.RMP_lb_list.append(round(self.LB,2))
                self.RMP_ub_list.append(round(self.UB,2))
                self.columnUsedCountList.append(self.columnUsedCount)
                print(f"cg_Num : {self.cg_count} , cg_Used:{self.columnUsedCount}, lb: {round(self.LB,2)} , ub: {round(self.UB,2)} , SPPval: {round(SPP.solution.objVal,2)} , time cost:{time.time()-s_time}")
            
            rmp_time = time.time()
            self.RMP.optimize() 
            self.RMP_timeCost += time.time()-rmp_time  # 精确记录RMP时间
            # solve subproblem 
            SPP.graph.dualValue = [0]+self.RMP.getAttr("Pi", self.RMP.getConstrs())+[0] # update dual_value
            SPP.updateObj()
            spp_time = time.time()
            SPP.solve()
            self.SPP_timeCost += time.time()-spp_time  # 精确记录SPP时间
            
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
                "columnSet":{key:sol.__dict__ for key,sol in self.solution.solSet.items()},
        }
        self.result = res  ## result

if __name__=='__main__':
    import sys
    sys.path.append("D:\Code\RL-for-CS\CG_test")
    import torch
    import model.Net as Net
    net = Net.GAT4(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=torch.device("cpu"))
    actor = Net.Actor(net)
    actor.load_state_dict(torch.load("model\\actor_standard_GAT4.pth", map_location=torch.device('cpu')))
    standard_file = json.load(open("model\\mini_batches_standard_60.json", "r"))
    max_min_info = standard_file["max_min_info"]
    #
    datapath = 'GH_instance_1-10hard'  
    save_path = 'output'
    print('save_path:',save_path)
    for filename in tqdm(os.listdir(datapath)):
        # if filename not in ['C1_2_1.json']: continue
        print(f'\n {filename}')
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
                         SolCount=100,   # 每次加不超过100列，<=100
                         Max_iters=20000,
                         isSave=True,
                         vehicleNum=50)  # 可以暂时不管，用来计算LB，但我们修改了时间窗，与原始的GH车辆不一样了，暂时不用它
        vrptw.solve()
        with open(os.path.join(save_path,filename),'w') as f:
            json.dump(vrptw.result, f)





