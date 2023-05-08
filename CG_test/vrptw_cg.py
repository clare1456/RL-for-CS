"""
Author: 71
LastEditTime: 2023-02-07 14:48:23
Description: column generation framework
"""

from espprc_gp import ESPPRC_gp
from espprc_ns import ESPPRC_NS
from gurobipy import GRB,Model,Column
from GetData import *
import time,json
from solution import *
from vrptw_initSol import *
from tqdm import tqdm
import numpy as np


class VRPTW_CG():
    def __init__(self,graph,
                 TimeLimit=2*60*60,
                 SPPTimeLimit=3*60,
                 SolCount=10,
                 Max_iters=20000,
                 isSave=False,
                 SPP_alg='gp',
                 initSol_alg='original',
                 filename='',
                 vehicleNum=50) :
        # result info
        self.solution = VRPSolution()
        self.RMP_lb_list = []
        self.RMP_ub_list = []
        self.SPP_objVal_list = []
        self.cg_count = 0
        self.UB = 0 # feasible solution
        self.LB = 0 # relaxation solution
        # parameters
        self.vehicleNum = vehicleNum
        self.epsilon = 1e-3
        self.Max_iters = Max_iters
        self.SolCount = SolCount
        self.init_cg_name = 'init_cg_'
        self.cg_name = 'cg_'
        self.filename = filename
        self.isSave = isSave
        self.TimeLimit = TimeLimit
        self.SPPTimeLimit = SPPTimeLimit
        self.graph = graph
        self.SPP_alg = SPP_alg
        self.initSol_alg = initSol_alg
        self.cg_vars = {}  # column management
        self.cost_time_list = []
        self.iterColumns = {}
        self.iters = 0
        self.initColumnNum = 0
        self.columnUsedCount = 0
        self.columnUsedCountList = []
        self.IterDualValueList = {}
        self.RMP_timeCost = 0
        self.SPP_timeCost = 0
        
    
    def set_SPP_alg(self,alg):
        if alg in ['gp','gurobi']:
            return ESPPRC_gp(self.graph,SolCount=self.SolCount,timeLimit=self.SPPTimeLimit)
        elif alg in ['ns']:
            return ESPPRC_NS(self.graph,SolCount=self.SolCount,timeLimit=self.SPPTimeLimit)


    def get_vrptw_initsol(self,filename):
        if self.initSol_alg in ['original','simple']:
            init_solution_vrptw = get_init_sol_original(self.graph,self.init_cg_name)
        elif self.initSol_alg in ['ns']:
            init_solution_vrptw = get_init_sol_ns(self.graph,self.init_cg_name,SolCount=1000)
        elif self.initSol_alg in ['nn','Nearest_Neighbor']:
            init_solution_vrptw = get_init_sol_Nearest_Neighbor(self.graph,self.init_cg_name)
        elif self.initSol_alg in ['best']:
            init_solution_vrptw = get_best_columns(self.graph,self.init_cg_name,filename=filename)
        return init_solution_vrptw
    
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
                
            for sol in SPP.solutionPool:
                if sol.objVal < 0:
                    self.cg_count += 1 # need modify
                    self.add_new_column(sol)

            self.column_manage()  # to be implemented

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
        
               
    def updateBound(self,SPP):
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.INTEGER)  
        self.RMP.optimize()
        self.UB = self.RMP.objVal
        self.columnUsedCount = len([1 for var in self.RMP.getVars() if 1-var.x<1e-3])   
        for var in self.RMP.getVars(): 
            var.setAttr("VType", GRB.CONTINUOUS)  
        self.RMP.optimize()
        self.LB = max(self.RMP.objVal + min(self.columnUsedCount * SPP.solution.objVal,0), 0)
      

    def column_manage(self):
        return 
              
    
    def print_SolPool(self):
        print('----------------print result----------------')
        self.solution.objVal = self.RMP.objVal
        self.solution.vehicleNum = 0
        self.bestColumnName = []
        for var in self.RMP.getVars():
            if(var.x==1):
                self.bestColumnName.append(var.VarName)
                self.solution.vehicleNum += 1
                print(f"{var.VarName},  path: {self.solution.solSet[var.VarName].path[1:-1]}, distance: {self.solution.solSet[var.VarName].distance}")
        print(f'ObjVal: {self.solution.objVal}')
        print(f'vehicleNum: {self.solution.vehicleNum}')
        print(f'columns: {len(self.cg_vars)}')
        print(f'itersNumber: {self.cg_count}')
      
    
    def update_RMP_Constr(self, new_path:list, path_distance, column_name):
        # creat new column 
        col_coef = [1 if i+1 in new_path else 0 for i in range(self.graph.customerNum)]
        rmp_col = Column(col_coef, self.rmp_constraints) 
        self.cg_vars[column_name] = self.RMP.addVar(lb = 0.0, ub = 1, obj = path_distance, vtype = GRB.CONTINUOUS, column = rmp_col,name=column_name)
        self.RMP.update() 
        # lp file
        # if self.isSave:
        #     self.RMP.write('RMP.lp')

    def build_RMP_gp(self):
        '''
        function: build a RMP of VRPTW on given graph with gurobi
        return RMP 
        '''        
        RMP = Model('VRPTW-RMP')
        customerNum = self.graph.customerNum 
        # set objective function 
        RMP.setAttr('ModelSense',GRB.MINIMIZE)
        # build constraint 
        rmp_constraints = [RMP.addConstr(RMP.addVar(lb=0, ub=0,name=f'0_{i}') >= 1,name=f"Constr_{i}") for i in range(1,customerNum+1)]
        return RMP,rmp_constraints
    
     
    def add_new_column(self,sol):
        new_path = sol.path 
        path_distance = sol.distance 
        column_name = self.cg_name + str(self.cg_count) # add new column to RMP, then update 
         # record info
        self.SPP_objVal_list.append(sol.objVal)
        self.solution.solSet[column_name] = sol 
        self.update_RMP_Constr(new_path,path_distance,column_name)
        self.iterColumns[self.iters].append(column_name)
        
if __name__=='__main__':
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
        vrptw = VRPTW_CG(graph,
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

