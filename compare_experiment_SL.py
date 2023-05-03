'''
File: compare_experiment_SL.py
Project: RL-for-CS
File Created: Saturday, 22nd April 2023 9:01:25 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''
from Env import *
from run import Args
import Net
import torch

args = Args()
args.instancce = "C1_2_2"

net = Net.GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
actor = Net.Actor(net)
actor.load_state_dict(torch.load("pretrain\\model_saved\\actor_standard_complete.pth", map_location=torch.device('cpu')))
# original column generation
def column_generation(model = None):
    start_time = time.time()
    env = CGEnv(args)
    state, info = env.reset(args.instance)
    iter_cnt = 0
    reward_list = []
    ub_lb_list = []
    RLMP_time_list = []
    time_list = []
    
    vehicleNum = 50 if env.CGAlg.graph.nodeNum < 400 else 100 # set vehicleNum manually
    while True:
        # test: randomly delete a column
        col_num = len(state["columns_state"])
        action = np.ones(col_num)
        if model == "greedy":
            action[1:] = 0
        elif model is not None:
            probs = actor(state, info)
            for i in range(len(action)):
                if probs[i][0] > 0.5:
                    action[i] = 0
        if sum(action) == 0: # at least one column
            action[np.random.randint(0, col_num)] = 1
        state, reward, done, info = env.step(action)
        ub = env.CGAlg.RLMP.ObjVal
        lb = max(env.CGAlg.RLMP.ObjVal + min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
        reward_list.append(reward)
        ub_lb_list.append([ub, lb])
        RLMP_time_list.append(env.CGAlg.RLMP_timeRecord)
        time_list.append(time.time() - start_time)
        print("Iter {}: delete column {}, reward = {}".format(iter_cnt, len(action)-sum(action), reward))
        if done:
            break
        iter_cnt += 1
    time_cost = time.time() - start_time
    if model is None:
        print("origin: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
    else:
        print("model: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
    return ub_lb_list, RLMP_time_list, time_list
        

# run column generation
origin_ub_lb_list, origin_RLMP_time_list, origin_time_list = column_generation()
model_ub_lb_list, model_RLMP_time_list, model_time_list = column_generation(actor)
greedy_ub_lb_list, greedy_RLMP_time_list, greedy_time_list = column_generation("greedy")
# preprocess
origin_ub_list = np.array(origin_ub_lb_list)[:,0]
origin_ub_list = (origin_ub_list - min(origin_ub_list)) / (max(origin_ub_list) - min(origin_ub_list))
model_ub_list = np.array(model_ub_lb_list)[:,0]
model_ub_list = (model_ub_list - min(model_ub_list)) / (max(model_ub_list) - min(model_ub_list))
greedy_ub_list = np.array(greedy_ub_lb_list)[:,0]
greedy_ub_list = (greedy_ub_list - min(greedy_ub_list)) / (max(greedy_ub_list) - min(greedy_ub_list))
origin_iter_list = np.arange(len(origin_ub_list))
model_iter_list = np.arange(len(model_ub_list))
greedy_iter_list = np.arange(len(greedy_ub_list))
# plot graphs
## 1. RMP time
plt.figure()
plt.plot(origin_RLMP_time_list, origin_ub_list, label="origin_ub")
plt.plot(model_RLMP_time_list, model_ub_list, label="model_ub")
plt.plot(greedy_RLMP_time_list, greedy_ub_list, label="greedy_ub")
title = args.instance + " RMP time"
plt.title(title)
plt.xlabel("RMP time")
plt.ylabel("obj")
plt.legend()
plt.savefig('outputs/pictures/{}.png'.format(title))
## 2. total time
plt.figure()
plt.plot(origin_time_list, origin_ub_list, label="origin_ub")
plt.plot(model_time_list, model_ub_list, label="model_ub")
plt.plot(greedy_time_list, greedy_ub_list, label="greedy_ub")
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
title = args.instance + " iter"
plt.title(title)
plt.xlabel("iter")
plt.ylabel("obj")
plt.legend()
plt.savefig('outputs/pictures/{}.png'.format(title))