'''
File: Env.py
Project: ML4CS
Description: Column Generation Environment
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''


import sys
sys.path.append("D:\Code\RL4CS\CGAlgs")
from utils.baseImport import *
from CGAlgs import GraphTool, ColumnGeneration
import gymnasium as gym


class CGEnv(gym.Env):
    def __init__(self, instance, limit_node_num=None):
        file_path = "problems\{}.txt".format(instance)
        self.graph = GraphTool.Graph(file_path, limit_node_num)
        self.CGAlg = CGWithSelection(self.graph)
        # action_space, observation_space updates 
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        self.action_space = gym.spaces.Box(0, 1, (1,), dtype=np.float64)
        self.step_cost = 10 # step punishment in reward

    def reset(self):
        # reset Column Generation Algorithm
        self.CGAlg.reset()
        # run column generation until column selection part
        CG_flag = self.CGAlg.column_generation_before_selection()
        assert CG_flag == -1, "ERROR: Column Generation finished in 0 step"
        # get state from alg
        info = self.CGAlg.get_column_selection_info()
        state = np.array(info["columns_state"], dtype=np.float64)
        # update action_space, observation_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, state.shape, dtype=np.float64)
        self.action_space = gym.spaces.Box(0, 1, (len(state),), dtype=np.float64)
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
        info = self.CGAlg.get_column_selection_info()
        state = info["columns_state"]
        reward = obj_before - obj_after - self.step_cost
        done = 0
        if CG_flag == 1:
            done = 1
        return state, reward, done, info
    
    def render(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        

class CGWithSelection(ColumnGeneration.ColumnGenerationWithLabeling):
    def reset(self):
        super().__init__(self.graph) # no need to build SP
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
            visited = [0] * self.graph.nodeNum
            for i in range(1, len(route)):
                dual_sum += self.duals_of_RLMP[f"R{route[i]}"] 
                dist_sum += self.graph.disMatrix[route[i-1], route[i]]
                visited[route[i]] = 1
            state = [dual_sum, dist_sum, visit_num] + visited # dim = (len(columns), 3 + nodeNum)
            columns_state.append(state)
        """ get states of nodes (constraints) """
        constraints_state = []
        for ni in range(self.graph.nodeNum):
            dual_value = self.duals_of_RLMP[f"R{ni}"]
            demand = self.graph.demand[ni]
            ready_time = self.graph.readyTime[ni]
            due_time = self.graph.dueTime[ni]
            service_time = self.graph.serviceTime[ni]
            state = [dual_value, demand, ready_time, due_time, service_time] # dim = (len(constraints), 5)
            constraints_state.append(state)
        """ get edge_index of columns and constraints """ 
        edge_index = [[], []]
        for ri in range(len(routes)):
            for ni in routes[ri][1:]:
                edge_index[0].append(ri) # dim = (2, len(columns) * len(constraints))
                edge_index[1].append(ni+len(routes)) # ! make idxs different from columns'
        info = {
            "columns_state" : np.array(columns_state), 
            "constraints_state" : np.array(constraints_state), 
            "edge_index" : np.array(edge_index)
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

    def column_generation(self):
        while True:
            CG_flag = self.column_generation_before_selection()
            if CG_flag != -1:
                break
            self.select_columns()
            self.get_columns_and_add_into_RLMP()
        return CG_flag


if __name__ == "__main__":
    instance = "R101"
    env = CGEnv(instance)

    state, info = env.reset()
    iter_cnt = 0
    reward_list = []
    while True:
        # test: randomly delete a column
        col_num = len(state)
        action = np.ones(col_num)
        choose_delete = np.random.randint(col_num)
        action[choose_delete] = 0
        state, reward, done, info = env.step(action)
        reward_list.append(reward)
        print("Iter {}: delete column {}, reward = {}".format(iter_cnt, choose_delete, reward))
        if done:
            break
        iter_cnt += 1
    plt.plot(range(len(reward_list)), reward_list)
    plt.show()

