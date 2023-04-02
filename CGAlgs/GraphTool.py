# read data from dateset

import numpy as np
import matplotlib.pyplot as plt
import math
import bitarray

class Graph():
    def __init__(self, file_name, limit_node_num = None):
        self.read_data(file_name, limit_node_num) # read data from file
        self.preprocess_data() # preprocess data
    
    def read_data(self, file_name, limit_node_num):
        """
        read VRPTW data from dataset
        input: file_name
        output: problem object (including (int)vehicleNum, (int capacity, (numpy-array[25, 6])customers)
                ps:customers include x, y, demand, ready_time, due_time, service_time
        """
        with open(file_name) as file_object:
            lines = file_object.readlines()
        
        # load vehicle setting
        vehicle = list(map(int, lines[4].split()))
        vehicleNum, capacity = vehicle

        # load customers setting
        location = []
        demand = []
        readyTime = []
        dueTime = []
        serviceTime = []
        for line in lines[9:]:
            cust = list(map(int, line.split()))
            if cust == []:
                continue
            location.append(cust[1:3])
            demand.append(cust[3])
            readyTime.append(cust[4])
            dueTime.append(cust[5])
            serviceTime.append(cust[6])

        # save data
        self.vehicleNum = vehicleNum
        self.capacity = capacity
        self.location = np.array(location[:limit_node_num])
        self.demand = np.array(demand[:limit_node_num])
        self.readyTime = np.array(readyTime[:limit_node_num])
        self.dueTime = np.array(dueTime[:limit_node_num])
        self.serviceTime = np.array(serviceTime[:limit_node_num])
     
    def preprocess_data(self):
        self.nodeNum = len(self.location) # record nodeNum
        self.cal_disMatrix() # calculate distances between each points
        self.cal_feasibleNodeSet() # filter feasible arc according to time window

    def cal_disMatrix(self):
        """
        calculate distances between each points
        """
        self.disMatrix = np.zeros((self.nodeNum, self.nodeNum))
        for i in range(self.nodeNum):
            for j in range(self.nodeNum):
                # self.disMatrix[i, j] = sum((self.location[i] - self.location[j])**2)**(1/2)
                self.disMatrix[i, j] = (np.linalg.norm(self.location[i] - self.location[j]))
        self.timeMatrix = self.disMatrix.copy() # speed=1 in solomon

    def cal_feasibleNodeSet(self):
        """
        filter feasible arc according to time window
        """
        self.feasibleNodeSet = [[] for _ in range(self.nodeNum)]
        self.availableNodeSet = [[] for _ in range(self.nodeNum)]
        self.infeasibleNodeSet = [[] for _ in range(self.nodeNum)]
        self.infeasibleBitSet = [bitarray.bitarray(self.nodeNum) for _ in range(self.nodeNum)]
        for i in range(self.nodeNum):
            for j in range(self.nodeNum):
                if i == j:
                    continue
                if self.readyTime[i] + self.serviceTime[i] + self.timeMatrix[i, j] <= self.dueTime[j]:
                    self.feasibleNodeSet[i].append(j)
                    self.availableNodeSet[j].append(i)
                else:
                    self.infeasibleNodeSet[i].append(j)
                    self.infeasibleBitSet[i][j] = 1
       
    def evaluate(self, routes, show=False, info = {}):
        obj = 0
        visit_customer = np.zeros(self.nodeNum)
        loads_record = []
        times_record = []
        objs_record = []
        # check each routes
        for route in routes:
            # check capacity constraint
            # check time window / pass all customers
            loads = []
            times = []
            t = 0
            load = 0
            for i in range(1, len(route)):
                pi = route[i-1]
                pj = route[i]
                t_ = t + self.serviceTime[pi] + self.timeMatrix[pi, pj]
                if t_ > self.dueTime[pj]:
                    print("Infeasible Solution: break time window")
                    return np.inf 
                t = max(t_, self.readyTime[pj])
                times.append(t)
                load += self.demand[pj]
                loads.append(load)
                if load > self.capacity:
                    print("Infeasible Solution: break capacity constraint")
                    return np.inf
                visit_customer[pj] = 1
            loads_record.append(loads)
            times_record.append(times)
            # calculate objective value
            dist = sum(self.disMatrix[route[:-1], route[1:]])
            obj += dist
            objs_record.append(dist)
        if sum(visit_customer) < self.nodeNum:
            print("Infeasible Solution: haven't visit all points")
            return np.inf
        if show:
            print("Feasible Solution: obj = {}".format(obj))
        info["loads_record"] = loads_record
        info["times_record"] = times_record
        info["objs_record"] = objs_record
        return obj

    def render(self, routes=[]):
        plt.figure()
        plt.scatter(self.location[1:, 0], self.location[1:, 1])
        plt.scatter(self.location[0:1, 0], self.location[0:1, 1], s = 150, c = 'r', marker='*')
        for route in routes:
            plt.plot(self.location[route, 0], self.location[route, 1])
        plt.show()

class GraphForAugerat(Graph):
    def __init__(self, file_name):
        self.read_data(file_name) 
        self.preprocess_data()

    def read_data(self, file_name):
        """
        read VRPTW data from Augerat dataset
        input: file_name
        output: problem object (including (int)vehicleNum, (int) capacity, (numpy-array[25, 6])customers)
                ps:customers include x, y, demand
        """
        with open(file_name) as file_object:
            lines = file_object.readlines()
        
        # load vehicle setting
        vehicleNum = nodeNum = int(lines[3].split()[2])
        capacity = int(lines[5].split()[2])

        # load customers setting
        location = []
        demand = []
        for line in lines[7:7+nodeNum]:
            cust = list(map(int, line.split()))
            location.append(cust[1:3])
        for line in lines[8+nodeNum:8+2*nodeNum]:
            cust = list(map(int, line.split()))
            demand.append(cust[1])
        self.vehicleNum = vehicleNum
        self.nodeNum = len(location)
        self.capacity = capacity
        self.location = np.array(location)
        self.demand = np.array(demand)

    def preprocess_data(self):
        self.nodeNum = len(self.location)
        self.cal_disMatrix()

if __name__ == "__main__":
    file_name = "solomon_100/r101.txt"
    graph = Graph(file_name)
    # file_name = "Augerat/A-n32-k5.vrp"
    # graph = GraphForAugerat(file_name)

    routes = [
        [0, 2, 21, 73, 41, 56, 4, 0], 
        [0, 5, 83, 61, 85, 37, 93, 0], 
        [0, 14, 44, 38, 43, 13, 0], 
        [0, 27, 69, 76, 79, 3, 54, 24, 80, 0], 
        [0, 28, 12, 40, 53, 26, 0], 
        [0, 30, 51, 9, 66, 1, 0], 
        [0, 31, 88, 7, 10, 0], 
        [0, 33, 29, 78, 34, 35, 77, 0], 
        [0, 36, 47, 19, 8, 46, 17, 0], 
        [0, 39, 23, 67, 55, 25, 0], 
        [0, 45, 82, 18, 84, 60, 89, 0], 
        [0, 52, 6, 0], 
        [0, 59, 99, 94, 96, 0], 
        [0, 62, 11, 90, 20, 32, 70, 0], 
        [0, 63, 64, 49, 48, 0], 
        [0, 65, 71, 81, 50, 68, 0], 
        [0, 72, 75, 22, 74, 58, 0], 
        [0, 92, 42, 15, 87, 57, 97, 0], 
        [0, 95, 98, 16, 86, 91, 100, 0], 
    ]
    print("value: {}".format(graph.evaluate(routes)))
