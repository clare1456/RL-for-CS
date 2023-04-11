# read data from dateset

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import bitarray

class Graph():
    def __init__(self, file_name, limit_node_num = None):
        if file_name.endswith('.json'):
            self.read_data_from_json(file_name, limit_node_num)
        else:
            self.read_data(file_name, limit_node_num) # read data from file
        self.preprocess_data() # preprocess data
    
    def read_data(self, file_name, limit_node_num):
        """
        read VRPTW data from dataset
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
     
    def read_data_from_json(self, file_name, limit_node_num):
        data = json.load(open(file_name, 'r')) 
        self.vehicleNum = data['vehicleNum']
        self.capacity = data['capacity']
        self.location = np.array(list(data['location'].values()))[:limit_node_num]
        self.demand = np.array(list(data['demand'].values()))[:limit_node_num]
        self.readyTime = np.array(list(data['readyTime'].values()))[:limit_node_num]
        self.dueTime = np.array(list(data['dueTime'].values()))[:limit_node_num]
        self.serviceTime = np.array(list(data['serviceTime'].values()))[:limit_node_num]

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
        self.adjMatrix = np.zeros((self.nodeNum, self.nodeNum))
        init_bitarray = bitarray.bitarray(self.nodeNum)
        init_bitarray ^= init_bitarray
        self.infeasibleBitSet = [init_bitarray.copy() for _ in range(self.nodeNum)]
        for i in range(self.nodeNum):
            for j in range(self.nodeNum):
                if i == j:
                    continue
                if self.readyTime[i] + self.serviceTime[i] + self.timeMatrix[i, j] <= self.dueTime[j]:
                    self.feasibleNodeSet[i].append(j)
                    self.availableNodeSet[j].append(i)
                    self.adjMatrix[i][j] = 1
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

if __name__ == "__main__":
    # file_name = "solomon_100/r101.txt"
    # graph = Graph(file_name)

    file_name = "pretrain\dataset\CGDataset\C1_2_1.json"
    graph = Graph(file_name)
