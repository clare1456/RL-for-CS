

from solution import SPPSolution
from espprc_ns import *
import os,json


      
def getSolution(graph,path):
        """ return a solution object given a path, status is used to determine if feasible
        """
        solution = SPPSolution()
        solution.path = path[:]  # deepcopy
        if len(path)!=len(set(path)):  # repeat check
            solution.status=False
            return solution
        for from_node,to_node in zip(path[:-1],path[1:]):
            solution.objVal += graph.disMatrix[from_node][to_node] - graph.dualValue[from_node]
            solution.travelTime += graph.disMatrix[from_node][to_node] + graph.serviceTime[from_node]
            solution.travelTime = max(solution.travelTime, graph.readyTime[to_node])
            solution.travelTimeList.append(solution.travelTime) 
            solution.distance += graph.disMatrix[from_node][to_node]
            solution.load += graph.demand[to_node]
            solution.loadList.append(solution.load) 
            if (to_node not in graph.feasibleNodeSet[from_node] or 
                solution.travelTime > graph.dueTime[to_node] or
                solution.load > graph.capacity
                ):
                solution.status=False
                return solution
        solution.status = True
        solution.objVal = round(solution.objVal,2)
        return solution

# 如需要不同的初始解，在此依次添加
def get_init_sol_original(graph,varName):
    """
        a simple rule to get initial solution of VRPTW. For 100-nodes instance, path like 0-node-101
    """
    init_solution_vrptw = {}
    for i in range(1,graph.customerNum+1):
        var_name = varName + str(i)
        new_path = [0] + [i] + [graph.verticles[-1]]
        init_solution_vrptw[var_name] = getSolution(graph,new_path)
    return init_solution_vrptw


def get_init_sol_ns(graph,varName,SolCount=1000,TimeLimit=60):
    """
        Neighborhood Search to get initial solution of VRPTW 
    """
    init_solution_vrptw = get_init_sol_Nearest_Neighbor(graph,varName)
    graph.dualValue = [0] + np.random.randint(0,200,graph.nodeNum-2).tolist() +[0]
    SPP = ESPPRC_NS(graph,SolCount=SolCount,timeLimit=TimeLimit)
    SPP.solve()
    for sol in SPP.solutionPool:
        var_name = varName + str(len(init_solution_vrptw)+1)
        init_solution_vrptw[var_name] = sol
    return init_solution_vrptw


def get_init_sol_Nearest_Neighbor(graph,varName):
    """ Nearest Neighbor to get initial solution of VRPTW
    """
    init_solution_vrptw = {}
    start_node,end_node = graph.verticles[0],graph.verticles[-1]
    to_be_visited_nodeset = graph.verticles[1:-1]
    SolCount = 0
    while(to_be_visited_nodeset):
        new_path = [start_node]
        SolCount += 1
        while (new_path[-1]!=end_node): # terminate when reach destination
            from_node = new_path[-1]
            min_distance,to_node = 1e8,end_node
            for node in graph.feasibleNodeSet[from_node]:
                if node not in to_be_visited_nodeset or node in new_path : continue
                if graph.disMatrix[from_node][node] < min_distance:
                    to_node = node
                    min_distance = graph.disMatrix[from_node][to_node]
            new_path.append(to_node)
            if to_node!=end_node:
                to_be_visited_nodeset.remove(to_node)
        var_name = varName + str(SolCount)
        init_solution_vrptw[var_name] = getSolution(graph,new_path)
    return init_solution_vrptw
    

def get_init_sol_saving(graph,varName):
    pass


def get_init_sol_insertion(graph,varName):
    pass



def get_best_columns(graph,varName,datapath='..\\dataset_solved\\VRPTW_GH_solverd',filename='C1_2_1.json'):
    init_solution_vrptw = {}
    filepath = os.path.join(datapath,filename)
    data = json.load(open(filepath,'r'))
    for colname in data['bestColumns']:
        col = data['columnSet'][colname]
        var_name = varName + str(len(init_solution_vrptw)+1)
        init_solution_vrptw[var_name] = SPPSolution(path=col['path'],distance=col['distance'])
    return init_solution_vrptw
  
    
if __name__=='__main__':
    import json
    filepath = 'instance/R1_2_1.json'  
    print(filepath)
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
    # vrptw_initSol = get_init_sol_ns(graph,varName="init_cg_",SolCount=1000,TimeLimit=60)  
    # vrptw_initSol = get_init_sol_original(graph,varName="init_cg_")  
    vrptw_initSol = get_init_sol_Nearest_Neighbor(graph,varName="init_cg_")  
    print("cgNum:",len(vrptw_initSol))
    print("Total_distance:",sum(sol.distance for sol in vrptw_initSol.values()))
    for name,sol in vrptw_initSol.items():
        print(f"{name} : {sol.path}")
        
  
