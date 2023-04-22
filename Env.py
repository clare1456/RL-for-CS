'''
File: Env.py
Project: ML4CS
Description: Column Generation Environment
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
sys.path.append("CGAlgs")
from utils.baseImport import *
from CGAlgs import GraphTool, ColumnGeneration
import gymnasium as gym

class CGEnv(gym.Env):
    def __init__(self, args):
        self.instance = "initial"
        self.limit_node_num = args.limit_node_num
        self.max_step = args.max_step # max iteration in one episode
        self.instance = args.instance
        file_path = "problems\{}.txt".format(self.instance)
        self.graph = GraphTool.Graph(file_path, self.limit_node_num)
        self.alpha = 100 # reward rate of obj improvement
        # action_space, observation_space updates 
        self.step_cost = 1 # step punishment in reward

    def reset(self, instance=None):
        # reset Column Generation Algorithm
        if instance is not None and instance != self.instance:
            self.instance = instance
            file_path = "problems\{}.txt".format(self.instance)
            self.graph = GraphTool.Graph(file_path, self.limit_node_num)
        self.CGAlg = CGWithSelection(self.graph)
        # run column generation until column selection part
        CG_flag = self.CGAlg.column_generation_before_selection()
        assert CG_flag == -1, "ERROR: Column Generation finished in 0 step"
        # get state from alg
        state = self.CGAlg.get_column_selection_info()
        self.obj_init = self.CGAlg.RLMP_obj
        info = {}
        self.iter_cnt = 0
        return state, info

    def step(self, action: np.ndarray):
        action = np.clip(action, 0, 1)
        """ select columns and sove RLMP """
        obj_before = self.CGAlg.RLMP_obj
        self.CGAlg.select_columns(action)
        self.CGAlg.get_columns_and_add_into_RLMP()
        CG_flag = self.CGAlg.column_generation_before_selection()
        obj_after = self.CGAlg.RLMP_obj
        """ get state, reward, done, info """
        state = self.CGAlg.get_column_selection_info()
        info = {}
        reward = self.alpha * (obj_before - obj_after) / self.obj_init - self.step_cost
        done = 0
        self.iter_cnt += 1
        if CG_flag == 1 or self.iter_cnt >= self.max_step:
            done = 1
        return state, reward, done, info
    
    def get_final_RLMP_obj(self):
        return self.CGAlg.RLMP_obj

    def get_iter_times(self):
        return self.iter_cnt

    def render(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        

class CGWithSelection(ColumnGeneration.ColumnGenerationWithLabeling):
    def __init__(self, graph):
        super(CGWithSelection, self).__init__(graph)
        self.OutputFlag = False

    def column_generation_before_selection(self):
        # solve RLMP and get duals
        is_feasible = self.solve_RLMP_and_get_duals()
        if is_feasible != 1:
            return 0
        # solve SP
        self.solve_SP()
        # record information
        self.cg_iter_cnt += 1
        self.output_info()
        # break if can't improve anymore
        if self.SP_obj >= -self.EPS:
            return 1
        return -1 # -1 means continue column generation

    def get_column_selection_info(self):
        """ get states of routes (columns) """
        routes = self.labeling_routes
        columns_state = []
        for route in routes:
            dual_sum = 0
            dist_sum = 0
            visit_num = len(route)
            demand_sum = sum(self.graph.demand[route])
            visited = [0] * self.graph.nodeNum
            for i in range(1, len(route)):
                dual_sum += self.duals_of_RLMP[f"R{route[i]}"] 
                dist_sum += self.graph.disMatrix[route[i-1], route[i]]
                visited[route[i]] = 1
            state = [dual_sum, dist_sum, demand_sum] # dim = (len(columns), 3)
            columns_state.append(state)
        """ get states of nodes (constraints) """
        constraints_state = []
        for ni in range(self.graph.nodeNum):
            dual_value = self.duals_of_RLMP[f"R{ni}"]
            coor_x, coor_y = self.graph.location[ni]
            demand = self.graph.demand[ni]
            ready_time = self.graph.readyTime[ni]
            due_time = self.graph.dueTime[ni]
            service_time = self.graph.serviceTime[ni]
            state = [dual_value, coor_x, coor_y, demand, ready_time, due_time] # dim = (len(constraints), 6)
            constraints_state.append(state)
        """ get edge_index of columns and constraints """ 
        edge_index = [[], []]
        for ri in range(len(routes)):
            for ni in routes[ri][1:]:
                edge_index[0].append(ri+self.graph.nodeNum) # dim = (2, len(constraints) * len(columns))
                edge_index[1].append(ni) 
        info = {
            "columns_state" : np.array(columns_state), 
            "constraints_state" : np.array(constraints_state), 
            "edges" : np.array(edge_index), 
        }
        return info

    def select_columns(self, select_result=None):
        if len(self.labeling_routes) == 0:
            return
        if select_result is None:
            delete_idx = np.random.randint(len(self.labeling_routes))
            self.labeling_routes.pop(delete_idx)
        else:
            self.labeling_routes = [self.labeling_routes[i] for i in range(len(self.labeling_routes)) if select_result[i]] 


if __name__ == "__main__":
    class Args:
        instance = "R101"
        limit_node_num = 50
        max_step = 20

    args = Args()
    env = CGEnv(args)

    start_time = time.time()
    state, info = env.reset()
    iter_cnt = 0
    reward_list = []
    while True:
        # test: randomly delete a column
        col_num = len(state["columns_state"])
        action = np.ones(col_num)
        # choose_delete = np.random.randint(col_num)
        # action[choose_delete] = 0
        state, reward, done, info = env.step(action)
        reward_list.append(reward)
        # print("Iter {}: delete column {}, reward = {}".format(iter_cnt, choose_delete, reward))
        if done:
            break
        iter_cnt += 1
    time_cost = time.time() - start_time
    print("total_reward = {}, time_cost = {}".format(sum(reward_list), time_cost))
    plt.plot(range(len(reward_list)), reward_list)
    plt.show()

