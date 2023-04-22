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
actor.load_state_dict(torch.load("pretrain\\model_saved\\actor.pth", map_location=torch.device('cpu')))
start_time = time.time()
state, info = env.reset(args.instance)
iter_cnt = 0
reward_list = []
model_ub_lb = []
vehicleNum = 50 if env.CGAlg.graph.nodeNum < 400 else 100 # set vehicleNum manually
while True:
    # test: randomly delete a column
    col_num = len(state["columns_state"])
    action = np.ones(col_num)
    # probs = actor(state, info)
    # for i in range(len(action)):
    #     if np.random.rand() < probs[i][1]:
    #         action[i] = 0
    state, reward, done, info = env.step(action)
    ub = env.CGAlg.RLMP_obj
    lb = max(env.CGAlg.RLMP_obj - min(vehicleNum * env.CGAlg.SP_obj, 0), 0)
    reward_list.append(reward)
    model_ub_lb.append([ub, lb])
    print("Iter {}: delete column {}, reward = {}".format(iter_cnt, len(action)-sum(action), reward))
    if done:
        break
    iter_cnt += 1
time_cost = time.time() - start_time
print("final_obj = {}, iter_cnt = {}, total_reward = {}, time_cost = {}".format(env.CGAlg.RLMP_obj, iter_cnt, sum(reward_list), time_cost))
plt.plot(range(len(reward_list)), reward_list)
plt.show()