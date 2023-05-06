'''
File: Trainer.py
Project: RL4CS
Description: Trainer of RL
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

from utils.baseImport import *
import collections
import random
import torch
import Env

class instanceGenerator:
    def __init__(self, args):
        # 样本生成模式 or 固定样本名称
        self.instance = args.instance 
        # 地图更新周期
        self.map_change_eps = args.map_change_eps 
        # curriculumn learning 从简单到复杂的训练顺序
        self.sequence = [
            f"C1_{i}_{j}" for i in [2,4] for j in range(1,5)
        ]
        # 随机模式下随机样本训练顺序
        if self.instance == "random": 
            np.random.shuffle(self.sequence)
        # 记录已经训练的样本数
        self.iter_cnt = 0
    
    def get(self):
        if self.instance == "sequence" or self.instance == "random":
            cur_instance = self.sequence[(self.iter_cnt // self.map_change_eps) % len(self.sequence)]
            self.iter_cnt += 1
            return cur_instance
        else:
            return self.instance
        

def trainOffPolicy(policy, args, res_queue, outputFlag=False, seed=0):
    """ 
    OffPolicy训练过程
    """
    if seed != 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    env = Env.CGEnv(args)
    instance_generator = instanceGenerator(args)
    buffer = ReplayBuffer(args.buffer_size, args.batch_size)
    critic_1_optim = torch.optim.Adam(policy.critic_1.parameters(), lr=args.critic_lr)
    critic_2_optim = torch.optim.Adam(policy.critic_2.parameters(), lr=args.critic_lr)
    actor_optim = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
    alpha_optim = torch.optim.Adam([policy.log_alpha], lr=args.alpha_lr)
    ep_rewards = [] # 记录所有回合奖励
    if outputFlag:
        print("\nTraining Begin!")
    step_cnt = 0
    for epi in range(args.train_eps):
        ep_reward = 0
        # reset environment
        instance = instance_generator.get()
        state, info = env.reset(instance)
        # interact until done
        while True:
            act = policy(state)
            next_state, rew, done, next_info = env.step(act)
            buffer.add(state, act, rew, next_state, done)
            ep_reward += rew
            if done:
                break
            state = next_state
            info = next_info
            # update policy
            if (step_cnt + 1) % args.update_steps == 0 and buffer.size() >= args.minimal_size:
                loss_info = policy.update(buffer, critic_1_optim, critic_2_optim, actor_optim, alpha_optim)
                # record loss information
                for key, value in loss_info.items():
                    res_queue.put({"tag" : key, "value" : value, "step" : epi+1})
            step_cnt += 1
        ep_rewards.append(ep_reward)
        # record result information
        res_queue.put({"tag" : "result/reward", "value" : ep_reward, "step" : epi+1})
        res_queue.put({"tag" : "result/finalObj", "value" : env.get_final_RLMP_obj(), "step" : epi+1})
        res_queue.put({"tag" : "result/iterTimes", "value" : env.get_iter_times(), "step" : epi+1})
        # output information
        if outputFlag and (epi + 1) % args.output_eps == 0:
            avg_reward = sum(ep_rewards[-args.output_eps:])/args.output_eps
            print("Episode {}/{}: avg_reward = {}".format(epi+1, args.train_eps, avg_reward))
    res_queue.put(None)
    if outputFlag:
        print("Training Finished!")
    return ep_rewards


def trainOnPolicy(policy, args, res_queue, outputFlag=False, seed=0):
    """ 
    OnPolicy训练过程
    """
    if seed != 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    env = Env.CGEnv(args)
    instance_generator = instanceGenerator(args)
    actor_optim = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(policy.critic.parameters(), lr=args.critic_lr)
    ep_rewards = [] # 记录所有回合奖励
    if outputFlag:
        print("\nTraining Begin!")
    step_cnt = 0
    for epi in range(args.train_eps):
        ep_reward = 0
        transition_dict = {"states" : [], "actions" : [], "rewards" : [], "next_states" : [], "dones" : []}
        # reset environment
        instance = instance_generator.get()
        state, info = env.reset(instance)
        # interact until done
        while True:
            act = policy(state)
            next_state, rew, done, next_info = env.step(act)
            transition_dict["states"].append(state)
            transition_dict["rewards"].append(rew)
            transition_dict["actions"].append(act)
            transition_dict["next_states"].append(next_state)
            transition_dict["dones"].append(done)
            ep_reward += rew
            if done:
                break
            state = next_state
            info = next_info
            step_cnt += 1
        ep_rewards.append(ep_reward)
        # update policy
        if (epi + 1) % args.update_eps == 0:
            loss_info = policy.update(transition_dict, actor_optim, critic_optim)
            for key, value in loss_info.items():
                res_queue.put({"tag" : key, "value" : value, "step" : epi+1})
        # record result information
        res_queue.put({"tag" : "result/reward", "value" : ep_reward, "step" : epi+1})
        res_queue.put({"tag" : "result/finalObj", "value" : env.get_final_RLMP_obj(), "step" : epi+1})
        res_queue.put({"tag" : "result/iterTimes", "value" : env.get_iter_times(), "step" : epi+1})
        # output information
        if outputFlag and (epi + 1) % args.output_eps == 0:
            avg_reward = sum(ep_rewards[-args.output_eps:])/args.output_eps
            print("Episode {}/{}: avg_reward = {}".format(epi+1, args.train_eps, avg_reward))
    res_queue.put(None)
    if outputFlag:
        print("Training Finished!")
    return ep_rewards


def test(policy, args, outputFlag=False):
    """
    测试过程
    """
    env = Env.CGEnv(args)
    ep_rewards = [] # 记录所有回合奖励
    if outputFlag:
        print("\nTesting Begin!")
    instance_generator = instanceGenerator(args)
    for epi in range(args.test_eps):
        ep_reward = 0
        # reset environment
        instance = instance_generator.get()
        state, info = env.reset(instance)
        # interact until done
        while True:
            act = policy(state)
            next_state, reward, done, info = env.step(act)
            ep_reward += reward
            if done:
                break
            state = next_state
        ep_rewards.append(ep_reward)
        # output information
        if outputFlag:
            print("Episode {}/{}: reward = {}".format(epi+1, args.test_eps, ep_reward))
    if outputFlag:
        print("Testing Finished!")
    return ep_rewards

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = collections.deque(maxlen=buffer_size) 
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self): 
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self): 
        return len(self.buffer)