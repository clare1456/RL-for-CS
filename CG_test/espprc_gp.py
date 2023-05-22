import numpy as np
from gurobipy import GRB,Model,LinExpr
from GetData import *
from solution import SPPSolution


        
class ESPPRC_gp():
    def __init__(self,Graph,timeLimit=10*60,SolCount=10,isPrint=False) -> None:
        self.solution = SPPSolution()
        self.solutionPool = []
        
        self.NumIntVars = 0 
        
        self.graph = Graph
        self.time_limit = timeLimit 
        self.SolCount = SolCount
        self.isPrint = isPrint
        
        self.SPP_model = None
        self.x = None
        self.s = None
        self.model_init()
        
    def model_init(self):
        self.preprocess()
        self.SPP_model,self.x,self.s = self.build_SPP()
        self.SPP_model.setParam("TimeLimit", self.time_limit)
        self.SPP_model.setParam("OutputFlag", self.isPrint)
        self.NumIntVars = self.SPP_model.NumIntVars
        
        
    def solve(self):  
        self.updateObj()
        self.SPP_model.optimize()
        if self.SPP_model.Status== GRB.OPTIMAL or self.SPP_model.Status== GRB.TIME_LIMIT:
            
            self.solution.objVal = round(self.SPP_model.objVal,2)
            for nsol in range (self.SPP_model.SolCount):
                self.SPP_model.setParam(GRB.Param.SolutionNumber, nsol)
                try: 
                    solution = get_solution(self.graph,round(self.SPP_model.PoolObjVal,2),self.x)
                    self.solutionPool.append(solution)
                except Exception:
                    pass
        else:
            print('fail to solve espprc, check~~~')
    
    
    def updateObj(self):
        self.solution = SPPSolution()
        self.solutionPool = []
        new_obj = LinExpr(0)
        for i in range(self.graph.nodeNum):
            for j in range(self.graph.nodeNum):
                if(i != j) and self.graph.arcs[i,j]==1:
                    new_obj.addTerms(self.graph.disMatrix[i][j]-round(self.graph.dualValue[i],4), self.x[i,j])
        self.SPP_model.setObjective(new_obj, GRB.MINIMIZE)
        
        
    def preprocess(self):  
        
        Graph = self.graph
        arcs = {}
        for i in range(Graph.nodeNum):
            for j in range(Graph.nodeNum):
                if (i!=Graph.nodeNum-1) and (j!=0) and (i!=j) and (Graph.readyTime[i]+Graph.serviceTime[i]+Graph.disMatrix[i][j] < Graph.dueTime[j]):
                    arcs[i,j] = 1
                else:
                    arcs[i,j] = 0  
        self.graph.arcs = arcs  
            
            
    def build_SPP(self):    
        model = Model('ESPPRC')  
        Graph = self.graph
        BigM = 1e5
        x,s = {},{}  
        for i in range(Graph.nodeNum):
            s[i] = model.addVar(lb = Graph.readyTime[i]
                                , ub = Graph.dueTime[i] 
                                , vtype = GRB.CONTINUOUS
                                , name = 's_' + str(i))           
            for j in range(Graph.nodeNum):
                if (i != j) and Graph.arcs[i,j]==1:
                    x[i,j] = model.addVar(0,1,vtype= GRB.BINARY,name= 'x_' + str(i) + '_' + str(j) )

        model.update()
        
        obj = LinExpr(0)
        for i in range(Graph.nodeNum):
            for j in range(Graph.nodeNum):
                if(i != j) and Graph.arcs[i,j]==1:
                    obj.addTerms(Graph.disMatrix[i][j]-round(Graph.dualValue[i],4), x[i,j])
        model.setObjective(obj, GRB.MINIMIZE)

        
        for h in range(1, Graph.nodeNum - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in range(Graph.nodeNum-1):
                if (h != i) and Graph.arcs[i,h]==1:
                    expr1.addTerms(1, x[i,h])
            for j in range(1,Graph.nodeNum):
                if (h != j) and Graph.arcs[h,j]==1:
                    expr2.addTerms(1, x[h,j])
            model.addConstr(expr1 == expr2, name= 'flow_conservation_' + str(i) + str(j))
            expr1.clear()
            expr2.clear()

        
        lhs = LinExpr(0)
        for j in range(1,Graph.nodeNum-1):
            if Graph.arcs[0,j]==1:
                lhs.addTerms(1, x[0,j])
        model.addConstr(lhs == 1, name= 'depart_' + str(j))

        
        lhs = LinExpr(0)
        for i in range(Graph.nodeNum-1):
            if Graph.arcs[i,Graph.nodeNum-1]==1:
                lhs.addTerms(1, x[i,Graph.nodeNum-1])
        model.addConstr(lhs == 1, name= 'arrive_' + str(i))

        
        for i in range(Graph.nodeNum):
            for j in range(Graph.nodeNum):
                if (i != j) and Graph.arcs[i,j]==1:
                    model.addConstr(s[i]+Graph.serviceTime[i]+Graph.disMatrix[i][j]-s[j] <= BigM*(1-x[i,j]), name = 'time_window' + str(i) + str(j))

        
        lhs = LinExpr(0)
        for i in range(Graph.nodeNum-1):
            for j in range(Graph.nodeNum):
                if (i!=j) and Graph.arcs[i,j]==1:
                    lhs.addTerms(Graph.demand[j], x[i,j])
        model.addConstr(lhs <= Graph.capacity, name= 'capacity_limited' )
        model.setParam(GRB.Param.PoolSolutions, self.SolCount)
        
        model.setParam(GRB.Param.PoolSearchMode, 2)
        return model,x,s

       
def get_solution(graph,objVal,x):
    '''
    @description: get new path from SPP model
    '''    

    Sol = SPPSolution()
    Sol.objVal = objVal 
    var_dict = {key[0]:key[1] for key in x.keys() if x[key].Xn ==1}
    assert len(var_dict.keys()) == len(set(list(var_dict.keys()))) ,'Wrong!!! repeat node in the path. '
    path = [0] 
    traval_time = [0] 
    load = 0
    distance = 0
    for _ in range(len(var_dict.keys())):
        cur_node = path[-1]
        next_node = var_dict[cur_node]
        path.append(next_node)
        distance += graph.disMatrix[cur_node][next_node]
        traval_time.append(graph.disMatrix[cur_node][next_node]+graph.serviceTime[cur_node])
        load += graph.demand[next_node]
    Sol.path = path        
    Sol.travelTime = traval_time
    Sol.load = load        
    Sol.distance = round(distance,2)      
    return Sol


    

if __name__=='__main__':
    np.random.seed(1)
    customerNum = 100
    datapath = r"solomon100/r102.txt"
    print(f"customerNum:{customerNum} , datapath:{datapath}")
    Graph = Data()
    Graph.read_solomon(path=datapath,customerNum=customerNum)
    
    Graph.dualValue = np.random.randint(0, 50, customerNum + 2)
    SPP = ESPPRC_gp(Graph)
    SPP.solve()
    print(f'objVal : {SPP.solution.objVal}')
    print(f'path : {SPP.solution.path}')
    for sol in SPP.solutionPool:
        print(f"Obj:{sol.objVal} , path:{sol.path}")
 
