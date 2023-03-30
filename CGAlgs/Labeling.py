# labeling algorithm for solving VRPTW subModel
# author: Charles Lee
# date: 2022.09.28

import numpy as np
import matplotlib.pyplot as plt
import math
import time

import GraphTool
# import ModelHandler

class Label():
    def __init__(self, path, tabu, obj, q, t):
        self.path = path # current route
        self.tabu = tabu # can't visit point list
        self.obj = obj # objective value of label
        self.q = q # current load of route
        self.t = t # start time of node
    
    @staticmethod
    def if_dominate(l1, l2):
        """check if l1 dominates l2 or on contrary

        Args:
            l1 (Label): label one 
            l2 (Label): label two
        Return:
            res (int): 0 stands for non-dominate, 1 for l1 dominate l2, 2 for l2 dominate l1
        """
        dominate_num = 0 
        if l1.obj < l2.obj:
            dominate_num += 1
        if l1.q < l2.q:
            dominate_num += 1
        if l1.t < l2.t:
            dominate_num += 1
        
        # if dominate_num == 3 and set(l1.path).issubset(l2.path): 
        if dominate_num == 3 and Label.is_subset(l1.tabu, l2.tabu): 
        # if dominate_num == 3:
            return 1
        # elif dominate_num == 0 and set(l2.path).issubset(l1.path): 
        elif dominate_num == 0 and Label.is_subset(l2.tabu, l1.tabu): 
        # elif dominate_num == 0:
            return 2
        else:
            return 0

    @staticmethod
    def is_subset(set1, set2):
        if len(set1) > len(set2):
            return False
        for key, value in set1.items():
            if set2.get(key, 0) == 0:
                return False
        return True

class Labeling():
    def __init__(self, graph, select_num=None, early_stop=0, outputFlag=True):
        self.name = "Labeling"
        self.graph = graph
        self.select_num = select_num # routes number generated at one time
        self.early_stop = early_stop # stop if complete route num > 2 * select_num
        assert early_stop != 1 or select_num is not None, "select_num can't be None if set early_stop"
        self.EPS = 1e-5
        self.iterCnt = 0
        self.outputFlag = outputFlag

        # node information
        self.dualValue = np.zeros(self.graph.nodeNum) # initialize as 0
        
    # basic funtions
    def initialize(self):
        """
        initialize variables 
        """
        self.Q = [[] for i in range(self.graph.nodeNum)] # queue for each points, containing part-routes that ends in the point
        label0 = Label([0], {}, 0, 0, 0)
        self.total_label_num = 1 # record
        self.last_label_num = 1 # record
        self.total_dominant_num = 0 # record
        self.best_obj = np.inf # record
        self.timeRecord = 0 # record: for debug
        self.label_expand(label0)
        self.last_label_num -= 1

    def set_dual(self, Dual):
        self.dualValue = Dual
    
    def dominant_add(self, label, node):
        """
        add label to node, while checking dominance
        input:
            label (Label): label to add
            node (int): idx of the node
        update:
            self.Q (dict[int:List]): queue for each points
        """
        if self.early_stop and len(self.Q[0]) >= 2 * self.select_num:
            return 
        li = 0
        while li < len(self.Q[node]):
            labeli = self.Q[node][li]
            flag = Label.if_dominate(label, labeli)
            # if l1 dominates l2, pop(l2)
            if flag == 1:
                self.Q[node].pop(li)
                self.total_dominant_num += 1
                self.last_label_num -= 1
            # if l2 dominates l1, not add l1
            elif flag == 2:
                self.total_dominant_num += 1
                return 
            else:
                li += 1
        self.Q[node].append(label)
        self.total_label_num += 1
        self.last_label_num += 1
            
    # problem depends
    def label_expand(self, label):
        """
        expand each labels in the node
        input:
            label (Label): label to expand
        update:
            self.Q (dict[int:List]): queue of node 
        """
        node = label.path[-1] # node is the current point of label
        for next_node in self.graph.feasibleNodeSet[node]: # next_node: the next node
            if node == next_node: # avoid circulation
                continue
            if next_node in label.path[1:]: # avoid can't visit set
                continue
            q_ = label.q + self.graph.demand[next_node]
            t_arrive = label.t + self.graph.serviceTime[node] + self.graph.timeMatrix[node, next_node]
            if q_ > self.graph.capacity or t_arrive > self.graph.dueTime[next_node]: # check feasibility
                continue
            t_ = max(self.graph.readyTime[next_node], t_arrive)
            # the correlation formula
            obj_ = label.obj + self.graph.disMatrix[node, next_node] - self.dualValue[next_node] # cal objective value
            if next_node == 0: # when route complete 
                if obj_ >=0: # only save negative obj route 
                    continue
                if obj_ < self.best_obj: # record best obj
                    self.best_obj = obj_
            path_ = label.path.copy()
            path_.append(next_node)
            # tabu_ = label.tabu.copy()
            # tabu_[next_node] = 1
            # for pi in self.graph.infeasibleNodeSet[next_node]:
            #     tabu_[pi] = 1
            new_label = Label(path_, label.tabu, obj_, q_, t_)
            start  = time.time()
            self.dominant_add(new_label, next_node) # add node and check dominance
            self.timeRecord += time.time() - start
    
    # output function
    def rank_result(self):
        pareto_labels = self.Q[0] 
        pareto_labels.sort(key=lambda label:label.obj)
        routes = [label.path for label in pareto_labels]
        objs = [label.obj for label in pareto_labels]
        return routes, objs

    # display information
    def display_information(self):
        print("Iter {}: best_obj={}, last_label_num={}, total_label_num={}, dominant_num={}"
            .format(self.iterCnt, self.best_obj, self.last_label_num, self.total_label_num, self.total_dominant_num))

    # main function
    def run(self):
        self.initialize() # initialize variables
        # expand labels until empty
        while True:
            breakFlag = True
            for node in range(1, self.graph.nodeNum):
                if len(self.Q[node]) > 0:
                    breakFlag = False
                while self.Q[node]:
                    label = self.Q[node].pop()
                    self.label_expand(label)
                    self.last_label_num -= 1
            if breakFlag:
                break
            if self.outputFlag:
                self.display_information()
            self.iterCnt += 1
        self.routes, self.objs = self.rank_result()
        return self.routes[:self.select_num], self.objs[:self.select_num] 
       
if __name__ == "__main__":
    # get data, build alg
    file_name = "solomon_100/r101.txt"
    graph = GraphTool.Graph(file_name)
    alg = Labeling(graph=graph, select_num=10)
    # set Dual value
    # Dual = [0] * alg.nodeNum
    # Dual = np.arange(graph.nodeNum)
    np.random.seed(1)
    Dual = np.random.randint(0, 50, graph.nodeNum)
    alg.set_dual(Dual)
    # solve and show result
    routes, objs = alg.run()
    print("timeRecord = {}".format(alg.timeRecord))
    for ri, route in enumerate(routes):
        print("{} obj: {}, route: {}".format(ri+1, objs[ri], route))
    


