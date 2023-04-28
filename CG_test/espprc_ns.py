
'''
Author: 71
LastEditTime: 2023-04-27 17:01:34
version: 
Description: neighborhood search for solving ESPPRC， 融合了遗传算法和邻域搜索
'''

from GetData import Data
import numpy as np
from solution import SPPSolution
import time 


        
class Route():
    def __init__(self):
        self.status = False
        self.objVal = 0
        self.path = []  
        self.travelTime = 0
        self.travelTimeList = [] 
        self.distance = 0 
        self.load = 0  
        self.loadList =[]


class ESPPRC_NS():
    def __init__(self,graph,timeLimit=10*60,SolCount=10,tabuLength=1,popSize=50,maxEndurance=500,isPrint=False) -> None:
        # result
        self.solution = SPPSolution()
        self.solutionPool = []
        # parameters
        self.graph = graph
        self.maxEndurance = maxEndurance # stop condition
        self.TimeLimit = timeLimit
        self.tabuLength = tabuLength  # 暂时用不到，可以想想怎么用起来
        self.SolCount = SolCount  # the number of solutions
        self.popSize = popSize  # population size
        self.State = 0  # init=0  loaded=1  optimal=2  INFEASIBLE=3  TIME_LIMIT=9   # 模仿gurobi，暂时用不到，之后统一接口，每次运算完检查后返回状态

        self.repaire_tabu_list = [] # 暂时用不到，可以想想怎么用起来
        self.iters = 0
        self.isPrint = isPrint
        
        
    def updateObj(self):
        """Update the state for new cycle in VRPTW"""
        self.solution = SPPSolution()
        self.solutionPool = []
        self.repaire_tabu_list = []
        self.iters = 0
    
    def stop_condition(self,start_time,endurance):
        if time.time()-start_time > self.TimeLimit:
            print("time limit exceeded...")
            self.State = 9
            return False
        elif endurance > self.maxEndurance: # 连续maxEndurance个解没有提高就结束
            return False
        else:
            return True


    def solve(self):
        self.updateObj()
        Candidates = self.get_init_solutions()
        bestSolution = self.get_best_solution(Candidates)
        start_time = time.time()
        endurance = 0  # 如果超过maxEndurance次最优解没有更新就停止
        while(self.stop_condition(start_time,endurance)):
            self.iters += 1 
            if self.isPrint and self.iters%10==0:
                print(f"iters:{self.iters} , endurance:{endurance} , obj:{bestSolution.objVal} , path:{bestSolution.path}")
            best_Candidates = self.select_pop(Candidates)  # population selection
            new_Candidates = []
            for target in best_Candidates:
                destroyed_path = self.randomDestory(target)
                newSol = self.randomRepaire(destroyed_path)  # repaire operator  
                if newSol.status:
                    new_Candidates.append(newSol)
            Candidates = best_Candidates + new_Candidates
            bestSolution_new = self.get_best_solution(Candidates)
            solutionPoolObjList = [sol.objVal for sol in self.solutionPool]
            for sol in Candidates: 
                if len(self.solutionPool) < self.SolCount and sol.objVal not in solutionPoolObjList:  # update solpool
                    self.solutionPool.append(sol)
                elif sol.objVal < self.solutionPool[0].objVal and sol.objVal not in solutionPoolObjList:
                    self.solutionPool.append(sol)
                    self.solutionPool.pop(0)    
            if bestSolution_new.objVal < bestSolution.objVal:  # minimize
                bestSolution = bestSolution_new
                endurance = 0
            else:
                endurance += 1
        self.solution = bestSolution
        # self.solutionPool = self.updateSolutionPool(Candidates)    
              
             
    def get_init_solutions(self):
        """  random policy to find a initial solution,
        """
        init_solutions = []
        for _ in range(self.popSize):
            path = [self.graph.verticles[0],self.graph.verticles[-1]]
            cur_solution = self.getSolution(path) 
            end_node = cur_solution.path[-1]
            ptr = 0 # pointer to 0
            while (cur_solution.path[ptr] != end_node):  # 不断的在路径里插入节点，直到插不下
                from_node,to_node = cur_solution.path[ptr],cur_solution.path[ptr+1]
                feasibleNodes = self.graph.feasibleNodeSet[from_node][:] # deepcopy
                np.random.shuffle(feasibleNodes)   # self.graph.feasibleNodeSet[from_node]表示 from_node可以到达的节点集合，事先根据时间窗计算出来了
                partial_path = None
                for insert_node in feasibleNodes:
                    if to_node in self.graph.feasibleNodeSet[insert_node]:
                        partial_path = [from_node,insert_node,to_node]
                        if (
                            partial_path not in self.repaire_tabu_list 
                            and insert_node not in cur_solution.path
                            ):  
                            new_path = cur_solution.path[:] # deepcopy
                            new_path.insert(new_path.index(from_node)+1,insert_node)
                            new_sol = self.getSolution(new_path) # 给定一个path,输出一个solution
                            if new_sol.status:
                                cur_solution = new_sol
                                self.repaire_tabu_list.append(partial_path)
                                if len(self.repaire_tabu_list) > self.tabuLength:
                                    self.repaire_tabu_list.pop(0)
                                break
                ptr += 1
            init_solutions.append(cur_solution)
        return init_solutions
        
        
    
    def select_pop(self,Candidates):
        '''
        function: select the best population for the exploration
        return best_Candidates
        '''      
        objlist = [sol.objVal for sol in Candidates]
        min_obj,max_obj = min(objlist),max(objlist)
        norm_objlist = [(val-min_obj)/(max_obj-min_obj) for val in objlist]  # normalize
        prob_obj = [np.exp(-obj) for obj in norm_objlist] # prob
        norm_prob_obj = [val/sum(prob_obj) for val in prob_obj] # norm
        targets_index = np.random.choice(len(Candidates), self.popSize, p=norm_prob_obj)
        return [Candidates[i] for i in targets_index] + [self.get_best_solution(Candidates)]  # keep elite
    
    
    def getSolution(self,path):
        """ return a solution object given a path, status is used to determine if feasible
        """
        solution = Route()
        solution.path = path[:]  # deepcopy
        if len(path)!=len(set(path)):  # repeat check
            solution.status=False
            return solution
        for from_node,to_node in zip(path[:-1],path[1:]):
            solution.objVal += self.graph.disMatrix[from_node][to_node] - self.graph.dualValue[from_node]
            solution.travelTime += self.graph.disMatrix[from_node][to_node] + self.graph.serviceTime[from_node]
            solution.travelTime = max(solution.travelTime, self.graph.readyTime[to_node])
            solution.travelTimeList.append(solution.travelTime) 
            solution.distance += self.graph.disMatrix[from_node][to_node]
            solution.load += self.graph.demand[to_node]
            solution.loadList.append(solution.load) 
            if (to_node not in self.graph.feasibleNodeSet[from_node] or 
                solution.travelTime > self.graph.dueTime[to_node] or
                solution.load > self.graph.capacity
                ):
                solution.status=False
                return solution
        solution.status = True
        solution.objVal = round(solution.objVal,2)
        return solution
    
        
    def get_best_solution(self,Candidates):
        best_sol = Candidates[0]
        for sol in Candidates:
            if sol.objVal < best_sol.objVal:
                best_sol = sol
        return best_sol
    
    
    def updateSolutionPool(self,Candidates):
        objList = [sol.objVal for sol in Candidates]
        index_objList = np.argsort(np.array(objList))  # from min to max
        return [Candidates[i] for i in index_objList[:self.SolCount]] 
                    

    def randomDestory(self,solution):  # sourcery skip: use-next
        remove_length = np.random.randint(0,len(solution.path)-1)
        path = solution.path[:]
        if len(solution.path) <= 2 or remove_length==0:
            return path
        rm_point = np.random.randint(1,len(solution.path)-remove_length)
        # destroy_step = min(random.randint(1,2),remove_length) # 破坏的路径最多两个节点，
        return path[:rm_point] + path[rm_point+1::]
        
            
    def randomRepaire(self,destroyed_path):    # sourcery skip: use-next
        """ neighborhood search to generate new solution, insert one node
        input: destroyed_path
        return solution
        """
        cur_solution = self.getSolution(destroyed_path) 
        end_node = cur_solution.path[-1]
        ptr = 0 # pointer to 0
        while (cur_solution.path[ptr] != end_node):  # 不断的在路径里插入节点，直到插不下
            from_node,to_node = cur_solution.path[ptr],cur_solution.path[ptr+1]
            feasibleNodes = self.graph.feasibleNodeSet[from_node][:] # deepcopy
            np.random.shuffle(feasibleNodes)   # self.graph.feasibleNodeSet[from_node]表示 from_node可以到达的节点集合，事先根据时间窗计算出来了
            partial_path = None
            for insert_node in feasibleNodes:
                if to_node in self.graph.feasibleNodeSet[insert_node]:
                    partial_path = [from_node,insert_node,to_node]
                    if (
                        partial_path not in self.repaire_tabu_list 
                        and insert_node not in cur_solution.path
                        and self.graph.dualValue[insert_node] >= (self.graph.disMatrix[from_node][insert_node] + self.graph.disMatrix[insert_node][to_node])
                        ):  # greedy insertion
                        new_path = cur_solution.path[:] # deepcopy
                        new_path.insert(new_path.index(from_node)+1,insert_node)
                        new_sol = self.getSolution(new_path) # 给定一个path,输出一个solution
                        if new_sol.status:
                            cur_solution = new_sol
                            self.repaire_tabu_list.append(partial_path)
                            if len(self.repaire_tabu_list) > self.tabuLength:
                                self.repaire_tabu_list.pop(0)
                            break
            ptr += 1
        return cur_solution
            
if __name__ == "__main__":
    import time
    from pyinstrument import Profiler

    profiler = Profiler()  # 这个用来看代码时间消耗的，优化代码
    profiler.start()
    
    print(time.ctime())
    np.random.seed(1)
    Graph = Data()
    customerNum = 100
    datapath = r"solomon100/r102.txt"
    print(f"customerNum:{customerNum} , datapath:{datapath}")
    Graph.read_solomon(path=datapath, customerNum=customerNum)
    Graph.dualValue = np.random.randint(0, 50, customerNum + 2) # dual value随机生成
    
    SPP = ESPPRC_NS(Graph,isPrint=True)
    SPP.solve()
    print(f"iters:{SPP.iters} , bestobjVal:{SPP.solution.objVal} , path:{SPP.solution.path}")
    for sol in SPP.solutionPool:
        print(f"Obj:{sol.objVal} , path:{sol.path}")
    
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))



