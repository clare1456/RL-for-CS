# applying gurobipy to solve VRPTW

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from time import time

import GraphTool
from ConstructiveHeuristics import *

class ModelHandler():
    def __init__(self, graph):
        self.graph = graph
        self.build_model()
        # self.build_3_model()

    def build_model(self):
        """
        VRPTW model in default
        """
        # building model
        model = gp.Model('VRPTW')

        nodeNum = self.graph.nodeNum
        points = list(range(nodeNum))
        A = [(i, j) for i in points for j in self.graph.feasibleNodeSet[i]]

        ## add variates
        x = model.addVars(A, vtype=GRB.BINARY, name="x")
        t = model.addVars(points, vtype=GRB.CONTINUOUS, name="t")
        q = model.addVars(points, vtype=GRB.CONTINUOUS, name="q")
        ## set objective
        model.modelSense = GRB.MINIMIZE
        model.setObjective(gp.quicksum(x[i, j] * self.graph.disMatrix[i, j] for i, j in A))
        ## set constraints
        ### 1. flow balance
        model.addConstrs(gp.quicksum(x[i, j] for j in self.graph.feasibleNodeSet[i] if j!=i)==1 for i in points[1:]) # depot not included
        model.addConstrs(gp.quicksum(x[i, j] for i in self.graph.availableNodeSet[j] if i!=j)==1 for j in points[1:]) # depot not included
        ### 2. avoid subring / self-loop
        model.addConstrs((x[i, j] == 1) >> (t[j] >= t[i] + self.graph.serviceTime[i] + self.graph.timeMatrix[i, j]) for i, j in A if j!=0)
        ### 3. time constraints
        model.addConstrs(t[i] >= self.graph.readyTime[i] for i in points)
        model.addConstrs(t[i] <= self.graph.dueTime[i] for i in points)
        ### 4. capacity constraints
        model.addConstrs((x[i, j] == 1) >> (q[j] >= q[i] + self.graph.demand[i]) for i, j in A if j!=0)
        model.addConstrs(q[i] <= self.graph.capacity for i in points)
        model.addConstrs(q[i] >= 0 for i in points)
        ### 5. vehicle number constraint
        model.addConstr(gp.quicksum(x[0, j] for j in self.graph.feasibleNodeSet[0]) <= self.graph.vehicleNum)

        # update model
        model.update()

        self.model = model
        return model

    def build_3_model(self):
        """
        三下标模型
        """
        # building model
        model = gp.Model('VRPTW_3')

        nodeNum = self.graph.nodeNum
        points = list(range(nodeNum))
        K = list(range(self.graph.vehicleNum))
        A = [(i, j, k) for i in points for j in self.graph.feasibleNodeSet[i] for k in K]
        t_set = [(i, k) for i in points for k in K]

        ## add variates
        x = model.addVars(A, vtype=GRB.BINARY, name="x")
        t = model.addVars(t_set, vtype=GRB.CONTINUOUS, name="t")
        q = model.addVars(t_set, vtype=GRB.CONTINUOUS, name="q")
        ## set objective
        model.modelSense = GRB.MINIMIZE
        model.setObjective(gp.quicksum(x[i, j, k] * self.graph.disMatrix[i, j] for i, j, k in A))
        ## set constraints
        ### 1. flow balance
        model.addConstrs(gp.quicksum(x[i, j, k] for i in self.graph.availableNodeSet[j]) == gp.quicksum(x[j, i, k] for i in self.graph.feasibleNodeSet[j]) for j in points for k in K)
        ### 2. pass each point
        model.addConstrs(gp.quicksum(x[i, j, k] for i in self.graph.availableNodeSet[j] for k in K) >= 1 for j in points) 
        ### 3. vehilcle number constraint
        model.addConstrs(gp.quicksum(x[0, j, k] for j in self.graph.feasibleNodeSet[0]) <= 1 for k in K)
        ### 4. time constraints
        model.addConstrs(t[i, k] >= self.graph.readyTime[i] for i in points for k in K)
        model.addConstrs(t[i, k] <= self.graph.dueTime[i] for i in points for k in K)
        model.addConstrs((x[i, j, k] == 1) >> (t[j, k] >= t[i, k] + self.graph.serviceTime[i] + self.graph.timeMatrix[i, j]) for i, j, k in A if j!=0)
        ### 5. capacity constraints
        model.addConstrs(gp.quicksum(self.graph.demand[i] * x[i, j, k] for i in points for j in self.graph.feasibleNodeSet[i]) <= self.graph.capacity for k in K)

        # update model
        model.update()

        self.model = model
        return model

    def build_RLMP_model(self, graph):
        """
        build RLMP of VRPTW 
        """
        RLMP = gp.Model()
        # init solution with Heuristics
        # heuristic = Solomon_Insertion(graph)
        # routes = heuristic.run()
        routes = [[0, i, 0] for i in range(1, graph.nodeNum)]
        routes_length = []
        routes_a = np.zeros((len(routes), graph.nodeNum))
        for ri, route in enumerate(routes):
            length = 0
            for pi in range(1, len(route)):
                length += graph.disMatrix[route[pi-1], route[pi]]
                routes_a[ri, route[pi]] = 1
            routes_length.append(length)
        # add init solution in RLMP
        ## add variables
        y_list = list(range(len(routes)))
        y = RLMP.addVars(y_list, vtype="C", name="y")
        ## set objective
        RLMP.setObjective(gp.quicksum(y[i] * routes_length[i] for i in range(len(routes))), GRB.MINIMIZE)
        ## set constraints
        RLMP.addConstrs(gp.quicksum(y[i] * routes_a[i, j] for i in range(len(routes))) >= 1 for j in range(graph.nodeNum))

        RLMP.setParam("OutputFlag", 0)
        RLMP._init_routes = routes # pass initial routes
        RLMP.update()
        return RLMP
    
    def build_SP_model(self, graph):
        """
        build SP for VRPTW 
        """
        SP = gp.Model()
        ## add variables
        points = list(range(graph.nodeNum))
        A_list = [(i, j) for i in points for j in graph.feasibleNodeSet[i]]
        x = SP.addVars(A_list, vtype="B", name="x")
        t = SP.addVars(points, vtype="C", name="t")
        ## set objective 
        SP.setObjective(gp.quicksum(x[i, j] * graph.disMatrix[i][j] \
            for i in points for j in graph.feasibleNodeSet[i]))
        ## set constraints
        ### 1. flow balance
        SP.addConstrs(gp.quicksum(x[i, j] for j in graph.feasibleNodeSet[i]) 
                      == gp.quicksum(x[j, i] for j in graph.availableNodeSet[i]) for i in points)
        SP.addConstr(gp.quicksum(x[i, 0] for i in graph.availableNodeSet[0]) == 1) # only one vehicle
        ### 2. capacity
        SP.addConstr(gp.quicksum(x[i, j] * graph.demand[i] for i,j in A_list) <= graph.capacity)
        ### 3. time window & sub-ring
        SP.addConstrs((x[i,j]==1) >> (t[j] >= t[i] + graph.serviceTime[i] + graph.timeMatrix[i, j]) for i, j in A_list if j!=0)
        SP.addConstrs(t[i] >= graph.readyTime[i] for i in points)
        SP.addConstrs(t[i] <= graph.dueTime[i] for i in points)

        # set model params
        SP.setParam("OutputFlag", 0)
        SP.update()
        return SP
  
    def get_routes(self, model=None):
        """
        get routes from model
        """
        if model is None:
            model = self.model
        # get the routes
        routes = []
        for j in self.graph.feasibleNodeSet[0]:
            if round(model.getVarByName(f"x[0,{j}]").X) == 1:
                route = [0]
                route.append(j)
                i = j
                while j != 0:
                    for j in self.graph.feasibleNodeSet[i]:
                        if round(model.getVarByName(f"x[{i},{j}]").X) == 1:
                            route.append(j)
                            i = j
                            break
                routes.append(route)
        return routes

    def get_obj(self, model=None):
        """
        get optimal objective value of model
        """
        if model is None:
            model = self.model
        return model.ObjVal

    def draw_routes(self, routes=None):
        """
        draw routes
        """
        if routes is None:
            routes = self.get_routes()
        self.graph.render(routes)

    def run(self):
        """
        solve model with gurobi solver 
        """
        # build model
        self.build_model()
        # tune the model
        # self.model.tune()
        # optimize the model
        self.model.optimize()
        # return routes
        if self.model.status == 2:
            return self.get_routes()
        else:
            print("Failed: Model is infeasible")
            return []

if __name__ == "__main__":
    # solve model with gurobi solver
    file_name = "pretrain\dataset\CGDataset\RC1_2_1.json"
    # 101\102-1s, 101_25-0.02s, 103-200s
    graph = GraphTool.Graph(file_name)
    alg = ModelHandler(graph)
    time1 = time.time()
    routes = alg.run()
    time2 = time.time()
    for ri in range(len(routes)):
        print("route {}: {}".format(ri, routes[ri]))
    info = {}
    graph.evaluate(routes, info=info)
    # graph.render(routes)
    print("optimal obj: {}\ntime consumption: {}".format(graph.evaluate(routes), time2-time1))

