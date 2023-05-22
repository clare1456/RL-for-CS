import math,re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import random


class Data():
    def __init__(self):
        self.location = []
        self.capacity = 200 
        self.demand = []
        self.serviceTime = []
        self.nodeNum = []
        

    def get_euclidean_distance_matrix(self):
        """Creates callback to return distance between locations."""
        distances = {}
        for from_counter, from_node in enumerate(self.location):
            distances[from_counter] = []
            temp_list = []
            for to_counter, to_node in enumerate(self.location):
                if from_counter == to_counter:
                    temp_list.append(0)
                    
                else:
                    
                    temp_list.append(round(
                    math.hypot((from_node[0] - to_node[0]),
                                (from_node[1] - to_node[1])),2)
                                )
            distances[from_counter] = temp_list
        self.disMatrix = distances

    def get_time_dismatrix(self):
        time_distances={}
        for i in range(len(self.dueTime)):
            time_dis = (np.array(self.dueTime)-np.array(self.dueTime[i])).tolist()
            
            for j in range(len(time_dis)):
                if time_dis[j]<0 or self.disMatrix[i][j]+self.dueTime[i]>self.dueTime[j]:
                    time_dis[j] = 100000
            time_distances[i] = time_dis
        self.timedisMatrix = time_distances


    def read_solomon(self,path,customerNum=100):
        '''Description: load solomon dataset'''        
        f = open(path, 'r')
        lines = f.readlines()
        self.location,self.demand,self.readyTime,self.dueTime,self.serviceTime=[],[],[],[],[]
        self.nodeNum = customerNum + 2
        self.customerNum = customerNum
        self.verticles = list(range(self.nodeNum))
        for count,line in enumerate(lines):
            count = count + 1
            if(count == 5):  
                line = line[:-1].strip() 
                str = re.split(r" +", line)
                self.vehicleNum = int(str[0])
                self.capacity = float(str[1])
            elif(count >= 10 and count <= 10 + customerNum):
                line = line[:-1]
                str = re.split(r" +", line)
                self.location.append((float(str[2]),float(str[3])))
                self.demand.append(float(str[4]))
                self.readyTime.append(float(str[5]))
                self.dueTime.append(float(str[6]))
                self.serviceTime.append(float(str[7]))
        self.location.append(self.location[0])
        self.demand.append(self.demand[0])
        self.readyTime.append(self.readyTime[0]) 
        self.dueTime.append(self.dueTime[0])
        self.serviceTime.append(self.serviceTime[0])
        self.update()
        
        self.location = {node:self.location[node] for node in self.verticles}
        self.demand = {node:self.demand[node] for node in self.verticles}
        self.readyTime = {node:self.readyTime[node] for node in self.verticles}
        self.dueTime = {node:self.dueTime[node] for node in self.verticles}
        self.serviceTime = {node:self.serviceTime[node] for node in self.verticles}
        
    def update(self):
        self.get_euclidean_distance_matrix()
        self.calAdj()


    def calAdj(self):
        self.adj = np.zeros((self.nodeNum,self.nodeNum)).tolist()
        self.feasibleNodeSet = {}
        for i in range(self.nodeNum):
            self.feasibleNodeSet[i] = []
            for j in range(self.nodeNum):
                if i==self.verticles[-1] or j==self.verticles[0]: 
                    self.adj[i][j] = 0
                    continue
                if i==self.verticles[0] and j==self.verticles[-1]:
                    self.adj[i][j] = 0
                    continue
                if i==j:
                    self.adj[i][j] = 0
                    continue
                if (self.readyTime[i]+self.serviceTime[i]+self.disMatrix[i][j] <= self.dueTime[j]):
                    self.adj[i][j] = 1
                    self.feasibleNodeSet[i].append(j)
                else:
                    self.adj[i][j] = 0 


    def plot_nodes(self):
        ''' function to plot locations'''
        self.location
        Graph = nx.DiGraph()
        nodes_name = [str(x) for x in list(range(len(self.location)))]
        Graph.add_nodes_from(nodes_name)
        pos_location = {nodes_name[i]:x for i,x in enumerate(self.location)}
        nodes_color_dict = ['r'] + ['gray'] * (len(self.location)-2) + ['r']
        nx.draw_networkx(Graph,pos_location,node_size=200,node_color=nodes_color_dict,labels=None)  
        plt.show()

    def plot_route(self,locations,route,edgecolor='k',showoff=True):
        ''' function to plot locations and route'''
        'e.g. [0,1,5,9,0]'
        Graph = nx.DiGraph()
        edge = []
        edges = []
        for i in route : 
            edge.append(i)
            if len(edge) == 2 :
                edges.append(tuple(edge))
                edge.pop(0)
        nodes_name = list(range(len(locations)))
        Graph.add_nodes_from(nodes_name)
        Graph.add_edges_from(edges)
        pos_location = {nodes_name[i] : x for i,x in enumerate(locations)}
        nodes_color_dict = ['r'] + ['gray'] * (len(locations)-2) + ['r']  
        nx.draw_networkx(Graph,pos_location,node_size=200,node_color=nodes_color_dict,edge_color=edgecolor,labels=None)
        if showoff:
            plt.show()


if __name__ == "__main__":
    
    nodes = Data()
    
    nodes.read_solomon(path='r101.txt',customerNum=30)
    
    nodes.plot_route(nodes.location,route=[0,1,2,4])


