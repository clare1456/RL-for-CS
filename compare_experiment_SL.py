'''
File: compare_experiment_SL.py
Project: RL-for-CS
File Created: Saturday, 22nd April 2023 9:01:25 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''
from Env import *
from pretrain.dataProcess import MILPSolver
from run import Args
import Net
import torch
import json
import os

args = Args()
args.instance = "C1_2_1"
args.max_step = 200

net = Net.GAT4(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
actor = Net.Actor(net)
actor.load_state_dict(torch.load("pretrain\\model_saved\\actor_standard_GAT4.pth", map_location=torch.device('cpu')))
# original column generation
def column_generation(model = "origin"):
    start_time = time.time()
    env = CGEnv(args)
    state, info = env.reset(args.instance)
    milp_solver = MILPSolver()
    vehicleNum = 50 if env.CGAlg.graph.nodeNum < 400 else 100 # set vehicleNum manually

    iter_cnt = 0
    reward_list = [0]
    ub = env.CGAlg.RLMP.ObjVal
    lb = max(env.CGAlg.RLMP.ObjVal + min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
    ub_lb_list = [[ub,lb]]
    RLMP_time_list = [env.CGAlg.RLMP_timeRecord]
    time_list = [time.time() - start_time]
    extra_flag = False if model in ["greedy", "origin"] else True
    
    while True:
        # test: randomly delete a column
        col_num = len(state["columns_state"])
        if model == "greedy":
            action = np.zeros(col_num)
            action[0] = 1
        elif model == "MILP":
            present_columns = list(env.CGAlg.column_pool.values())
            new_columns = env.CGAlg.labeling_routes.copy()
            for columns in [present_columns, new_columns]:
                for ci in range(len(columns)):
                    route = columns[ci]
                    columns[ci] = {}
                    dist = 0
                    onehot_path = [0] * env.CGAlg.graph.nodeNum
                    for i in range(len(route)-1):
                        onehot_path[route[i]] = 1
                        dist += env.CGAlg.graph.disMatrix[route[i]][route[i+1]]
                    columns[ci]["distance"] = dist
                    columns[ci]["onehot_path"] = onehot_path
            action = milp_solver.solve(present_columns, new_columns, env.CGAlg.graph.nodeNum)
        elif model == "origin":
            action = np.ones(col_num)
        elif model is not None:
            action = np.zeros(col_num)
            probs = actor(state, info)
            for i in range(len(action)):
                if probs[i][1] > 0.5:
                    action[i] = 1
        else:
            raise ValueError("model is None")
        if sum(action) == 0: # at least one column
            action[np.random.randint(0, col_num)] = 1
        state, reward, done, info = env.step(action, extra_flag)
        ub = env.CGAlg.RLMP.ObjVal
        lb = max(env.CGAlg.RLMP.ObjVal + min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
        reward_list.append(reward)
        ub_lb_list.append([ub, lb])
        RLMP_time_list.append(env.CGAlg.RLMP_timeRecord)
        time_list.append(time.time() - start_time)
        print("Iter {}: delete column {}, extra_route_num = {}, reward = {}, column_num = {}"
              .format(iter_cnt, len(action)-sum(action), reward, info["extra_route_num"], len(env.CGAlg.column_pool)))
        if done:
            break
        iter_cnt += 1
    time_cost = time.time() - start_time
    if model is None:
        print("origin: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
    else:
        print("model: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
    return ub_lb_list, RLMP_time_list, time_list
        
if not os.path.exists("outputs/data/compare_{}.json".format(args.instance)):
    # run column generation
    greedy_ub_lb_list, greedy_RLMP_time_list, greedy_time_list = column_generation("greedy")
    origin_ub_lb_list, origin_RLMP_time_list, origin_time_list = column_generation("origin")
    MILP_ub_lb_list, MILP_RLMP_time_list, MILP_time_list = column_generation("MILP")
    model_ub_lb_list, model_RLMP_time_list, model_time_list = column_generation(actor)
    # preprocess
    origin_ub_list = np.array(origin_ub_lb_list)[:,0]
    origin_lb_list = np.array(origin_ub_lb_list)[:,1]
    model_ub_list = np.array(model_ub_lb_list)[:,0]
    model_lb_list = np.array(model_ub_lb_list)[:,1]
    greedy_ub_list = np.array(greedy_ub_lb_list)[:,0]
    greedy_lb_list = np.array(greedy_ub_lb_list)[:,1]
    MILP_ub_list = np.array(MILP_ub_lb_list)[:,0]
    MILP_lb_list = np.array(MILP_ub_lb_list)[:,1]
    origin_iter_list = np.arange(len(origin_ub_list))
    model_iter_list = np.arange(len(model_ub_list))
    greedy_iter_list = np.arange(len(greedy_ub_list))
    MILP_iter_list = np.arange(len(MILP_ub_list))
    # save data
    data = {
        "origin_ub_list": list(origin_ub_list),
        "model_ub_list": list(model_ub_list),
        "greedy_ub_list": list(greedy_ub_list),
        "MILP_ub_list": list(MILP_ub_list),
        "origin_lb_list": list(origin_lb_list),
        "model_lb_list": list(model_lb_list),
        "greedy_lb_list": list(greedy_lb_list),
        "MILP_lb_list": list(MILP_lb_list),

        "origin_RLMP_time_list": list(origin_RLMP_time_list),
        "model_RLMP_time_list": list(model_RLMP_time_list),
        "greedy_RLMP_time_list": list(greedy_RLMP_time_list),
        "MILP_RLMP_time_list": list(MILP_RLMP_time_list),

        "origin_time_list": list(origin_time_list),
        "model_time_list": list(model_time_list),
        "greedy_time_list": list(greedy_time_list),
        "MILP_time_list": list(MILP_time_list),
    }
    path = "outputs/data/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "compare_{}.json".format(args.instance), "w") as f:
        json.dump(data, f)
else:
    # read data if already exists
    data = json.load(open("outputs/data/compare_{}.json".format(args.instance), "r"))
    origin_ub_list = np.array(data["origin_ub_list"])
    model_ub_list = np.array(data["model_ub_list"])
    greedy_ub_list = np.array(data["greedy_ub_list"])
    MILP_ub_list = np.array(data["MILP_ub_list"])
    origin_lb_list = np.array(data["origin_lb_list"])
    model_lb_list = np.array(data["model_lb_list"])
    greedy_lb_list = np.array(data["greedy_lb_list"])
    MILP_lb_list = np.array(data["MILP_lb_list"])
    origin_RLMP_time_list = np.array(data["origin_RLMP_time_list"])
    model_RLMP_time_list = np.array(data["model_RLMP_time_list"])
    greedy_RLMP_time_list = np.array(data["greedy_RLMP_time_list"])
    MILP_RLMP_time_list = np.array(data["MILP_RLMP_time_list"])
    origin_time_list = np.array(data["origin_time_list"])
    model_time_list = np.array(data["model_time_list"])
    greedy_time_list = np.array(data["greedy_time_list"])
    MILP_time_list = np.array(data["MILP_time_list"])
    origin_iter_list = np.arange(len(origin_ub_list))
    model_iter_list = np.arange(len(model_ub_list))
    greedy_iter_list = np.arange(len(greedy_ub_list))
    MILP_iter_list = np.arange(len(MILP_ub_list))
# plot graphs
# normalize
# max_origin_ub = max(origin_ub_list)
# min_origin_ub = min(origin_ub_list)
# max_origin_lb = max(origin_lb_list)
# min_origin_lb = min(origin_lb_list)
# origin_ub_list = (origin_ub_list - min_origin_ub) / (max_origin_ub - min_origin_ub)
# origin_lb_list = (origin_lb_list - min_origin_lb) / (max_origin_lb - min_origin_lb)
# model_ub_list = (model_ub_list - min_origin_ub) / (max_origin_ub - min_origin_ub)
# model_lb_list = (model_lb_list - min_origin_lb) / (max_origin_lb - min_origin_lb)
# greedy_ub_list = (greedy_ub_list - min_origin_ub) / (max_origin_ub - min_origin_ub)
# greedy_lb_list = (greedy_lb_list - min_origin_lb) / (max_origin_lb - min_origin_lb)
# MILP_ub_list = (MILP_ub_list - min_origin_ub) / (max_origin_ub - min_origin_ub)
# MILP_lb_list = (MILP_lb_list - min_origin_lb) / (max_origin_lb - min_origin_lb)
## 1. RMP time
plt.figure()
plt.plot(origin_RLMP_time_list, origin_ub_list, label="origin_ub")
plt.plot(model_RLMP_time_list, model_ub_list, label="model_ub")
plt.plot(greedy_RLMP_time_list, greedy_ub_list, label="greedy_ub")
plt.plot(MILP_RLMP_time_list, MILP_ub_list, label="MILP_ub")
plt.plot(origin_RLMP_time_list, origin_lb_list-1, label="origin_lb")
plt.plot(model_RLMP_time_list, model_lb_list-1, label="model_lb")
plt.plot(greedy_RLMP_time_list, greedy_lb_list-1, label="greedy_lb")
plt.plot(MILP_RLMP_time_list, MILP_lb_list-1, label="MILP_lb")
title = args.instance + " RMP time"
plt.title(title)
plt.xlabel("RMP time")
plt.ylabel("obj")
plt.legend()
plt.xlim(0, max(greedy_RLMP_time_list)) # 横坐标截断
plt.savefig('outputs/pictures/{}.png'.format(title))
## 2. total time
plt.figure()
plt.plot(origin_time_list, origin_ub_list, label="origin_ub")
plt.plot(model_time_list, model_ub_list, label="model_ub")
plt.plot(greedy_time_list, greedy_ub_list, label="greedy_ub")
plt.plot(MILP_time_list, MILP_ub_list, label="MILP_ub")
title = args.instance + " total time"
plt.title(title)
plt.xlabel("time")
plt.ylabel("obj")
plt.legend()
plt.savefig('outputs/pictures/{}.png'.format(title))
## 3. iter times
plt.figure()
plt.plot(origin_iter_list, origin_ub_list, label="origin_ub")
plt.plot(model_iter_list, model_ub_list, label="model_ub")
plt.plot(greedy_iter_list, greedy_ub_list, label="greedy_ub")
plt.plot(MILP_iter_list, MILP_ub_list, label="MILP_ub")
title = args.instance + " iter"
plt.title(title)
plt.xlabel("iter")
plt.ylabel("obj")
plt.legend()
plt.savefig('outputs/pictures/{}.png'.format(title))