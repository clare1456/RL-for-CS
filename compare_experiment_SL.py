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
env = CGEnv(args)

net = Net.GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
actor = Net.Actor(net)
actor.load_state_dict(torch.load("pretrain\\model_saved\\actor_standard.pth", map_location=torch.device('cpu')))
start_time = time.time()

# original column generation
state, info = env.reset(args.instance)
iter_cnt = 0
reward_list = []
origin_ub_lb = []
vehicleNum = 50 if env.CGAlg.graph.nodeNum < 400 else 100 # set vehicleNum manually
while True:
    # test: randomly delete a column
    col_num = len(state["columns_state"])
    action = np.ones(col_num)
    probs = actor(state, info)
    # for i in range(len(action)):
    #     if np.random.rand() < probs[i][0]:
    #         action[i] = 0
    state, reward, done, info = env.step(action)
    RMP = env.CGAlg.solve_final_RMP()
    ub = RMP.ObjVal
    lb = max(RMP.ObjVal + min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
    reward_list.append(reward)
    origin_ub_lb.append([ub, lb])
    print("Iter {}: delete column {}, reward = {}".format(iter_cnt, len(action)-sum(action), reward))
    if done:
        break
    iter_cnt += 1
time_cost = time.time() - start_time
print("origin: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))

# model column generation
state, info = env.reset(args.instance)
iter_cnt = 0
reward_list = []
model_ub_lb = []
vehicleNum = 50 if env.CGAlg.graph.nodeNum < 400 else 100 # set vehicleNum manually
while True:
    # test: randomly delete a column
    col_num = len(state["columns_state"])
    action = np.ones(col_num)
    probs = actor(state, info)
    for i in range(len(action)):
        if np.random.rand() < probs[i][0]:
            action[i] = 0
    state, reward, done, info = env.step(action)
    RMP = env.CGAlg.solve_final_RMP()
    ub = RMP.ObjVal
    lb = max(RMP.ObjVal + min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
    reward_list.append(reward)
    model_ub_lb.append([ub, lb])
    print("Iter {}: delete column {}, reward = {}".format(iter_cnt, len(action)-sum(action), reward))
    if done:
        break
    iter_cnt += 1
time_cost = time.time() - start_time
print("model: final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))

plt.plot(range(len(origin_ub_lb)), np.array(origin_ub_lb)[:,0], label="origin_ub")
plt.plot(range(len(origin_ub_lb)), np.array(origin_ub_lb)[:,1], label="origin_lb")
plt.plot(range(len(model_ub_lb)), np.array(model_ub_lb)[:,0], label="model_ub")
plt.plot(range(len(model_ub_lb)), np.array(model_ub_lb)[:,1], label="model_lb")
plt.legend()
plt.show()