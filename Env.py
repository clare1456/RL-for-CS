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
import json

class CGEnv(gym.Env):
    def __init__(self, args):
        self.instance = "initial"
        self.limit_node_num = args.limit_node_num
        self.max_step = args.max_step # max iteration in one episode
        self.min_select_num = args.min_select_num # min column num in one episode
        # self.instance = args.instance
        # file_path = "problems\{}.txt".format(self.instance)
        # self.graph = GraphTool.Graph(file_path, self.limit_node_num)
        self.alpha = 1000 # reward rate of obj improvement
        # action_space, observation_space updates 
        self.step_cost = 1 # step punishment in reward
        standard_file = json.load(open(args.standard_file, "r"))
        self.max_min_info = standard_file["max_min_info"]

    def reset(self, instance=None):
        # reset Column Generation Algorithm
        if instance is not None and instance != self.instance:
            self.instance = instance
            file_path = "problems\{}.json".format(self.instance)
            self.graph = GraphTool.Graph(file_path, self.limit_node_num)
        self.CGAlg = CGWithSelection(self.graph)
        # run column generation until column selection part
        CG_flag = self.CGAlg.column_generation_before_selection()
        assert CG_flag == -1, "ERROR: Column Generation finished in 0 step"
        # get state from alg
        state = self.CGAlg.get_column_selection_info()
        self.standardize_state(state) # state standardization
        self.obj_init = self.CGAlg.RLMP_obj
        info = {}
        self.iter_cnt = 0
        return state, info

    def standardize_state(self, state):
        for column_state in state["columns_state"]:
            for fi in range(len(column_state)):
                column_state[fi] = (column_state[fi] - self.max_min_info["column_state_min"][fi]) / (self.max_min_info["column_state_max"][fi] - self.max_min_info["column_state_min"][fi])
        for constraint_state in state["constraints_state"]:
            for fi in range(len(constraint_state)):
                constraint_state[fi] = (constraint_state[fi] - self.max_min_info["constraint_state_min"][fi]) / (self.max_min_info["constraint_state_max"][fi] - self.max_min_info["constraint_state_min"][fi])

    def reward_function1(self, iter_step):
        return 1 - (1 - 1e-3 * iter_step) ** 0.35
    
    def reward_function2(self, column_num):
        return max(0, (1 - (1.1 - 2e-4*column_num) ** 0.5))

    def step(self, action: np.ndarray, extra_flag: bool = True):
        """
        Args: 
            action: np.ndarray, shape = (route_num, )
            extra_flag: bool, whether to select columns with negative reduced cost 
        """
        action = np.clip(action, 0, 1)
        info = {"extra_route_num": 0}
        """ select columns predicted by RLMP and sove RLMP """
        obj_before = self.CGAlg.RLMP_obj
        unselected_routes = [route for ri, route in enumerate(self.CGAlg.labeling_routes) if action[ri] == 0] # 预存所有未选择路
        self.CGAlg.select_columns(action) # 根据action选择列
        self.CGAlg.get_columns_and_add_into_RLMP() # 将列加入RLMP
        if extra_flag:
            self.CGAlg.solve_RLMP_and_get_duals() # 更新对偶值
            """ select columns with negative rc and sove RLMP """
            reduced_costs = self.CGAlg.evaluate_routes(unselected_routes) # 更新未选择路的对偶值
            self.CGAlg.labeling_routes = [route for ri, route in enumerate(unselected_routes) if reduced_costs[ri] < 0] # 根据对偶值选择路
            info["extra_route_num"] = len(self.CGAlg.labeling_routes) # 记录额外选择的路的数量
            self.CGAlg.get_columns_and_add_into_RLMP() # 将列加入RLMP
        CG_flag = self.CGAlg.column_generation_before_selection() # 进行列生成
        obj_after = self.CGAlg.RLMP_obj
        """ get state, reward, done, info """
        state = self.CGAlg.get_column_selection_info()
        self.standardize_state(state) # state standardization
        # reward = self.alpha * (obj_before - obj_after) / self.obj_init - self.step_cost
        # reward = self.alpha * (obj_before - obj_after) / self.obj_init - self.step_cost * ((sum(action) + info["extra_route_num"]) / len(action))
        reward = self.alpha * (obj_before - obj_after) / obj_before - self.reward_function1(self.iter_cnt) - self.reward_function2(sum(action) + info["extra_route_num"])
        done = 0
        self.iter_cnt += 1
        if CG_flag == 1 or self.iter_cnt >= self.max_step:
            done = 1
            if CG_flag == 1:
                reward += 1000 # 结束大奖励
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
    
    def evaluate_routes(self, routes):
        # 计算路经过的点的reduced cost
        reduced_costs = []
        for route in routes:
            dual_sum = 0
            dist_sum = 0
            for i in range(1, len(route)):
                dual_sum += self.duals_of_RLMP[f"R{route[i]}"]
                dist_sum += self.graph.disMatrix[route[i-1]][route[i]]
            reduced_costs.append(dist_sum - dual_sum)
        return reduced_costs
         
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

    def select_columns(self, select_result):
        if len(self.labeling_routes) == 0:
            return
        self.labeling_routes = [self.labeling_routes[i] for i in range(len(self.labeling_routes)) if select_result[i]] 
        self.labeling_objs = [self.labeling_objs[i] for i in range(len(self.labeling_objs)) if select_result[i]]
        self.SP_obj = min(self.labeling_objs)

if __name__ == "__main__":
    from run import Args
    import Net
    import torch

    args = Args()
    env = CGEnv(args)

    net = Net.GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
    actor = Net.Actor(net)
    # actor.load_state_dict(torch.load("pretrain\\model_saved\\actor.pth", map_location=torch.device('cpu')))
    start_time = time.time()
    state, info = env.reset(args.instance)
    iter_cnt = 0
    reward_list = []
    while True:
        # test: randomly delete a column
        col_num = len(state["columns_state"])
        action = np.ones(col_num)
        probs = actor(state, info)
        # for i in range(len(action)):
        #     if np.random.rand() < probs[i][1]:
        #         action[i] = 0
        state, reward, done, info = env.step(action)
        reward_list.append(reward)
        print("Iter {}: delete column {}, reward = {}".format(iter_cnt, len(action)-sum(action), reward))
        if done:
            break
        iter_cnt += 1
    time_cost = time.time() - start_time
    print("final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
    plt.plot(range(len(reward_list)), reward_list)
    plt.show()

