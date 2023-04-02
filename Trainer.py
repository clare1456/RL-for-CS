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

def trainOffPolicy(policy, args, outputFlag=False):
    """ 
    训练过程
    """
    env = Env.CGEnv(args)
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
        obs, info = env.reset()
        # interact until done
        while True:
            act = policy(obs)
            next_obs, rew, done, next_info = env.step(act)
            buffer.add(obs, act, rew, next_obs, done)
            ep_reward += rew
            if done:
                break
            obs = next_obs
            info = next_info
            # update policy
            if (step_cnt + 1) % args.update_steps == 0 and buffer.size() >= args.minimal_size:
                policy.update(buffer, critic_1_optim, critic_2_optim, actor_optim, alpha_optim)
            step_cnt += 1
        ep_rewards.append(ep_reward)
        # output information
        if outputFlag and (epi + 1) % args.output_eps == 0:
            print("Episode {}/{}: avg_reward = {}"
                    .format(epi+1, args.train_eps, 
                            sum(ep_rewards[-args.output_eps:])/args.output_eps))
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
    for epi in range(args.test_eps):
        ep_reward = 0
        # reset environment
        obs, info = env.reset()
        # interact until done
        while True:
            act = policy(obs)
            next_obs, reward, done, info = env.step(act)
            ep_reward += reward
            if done:
                break
            obs = next_obs
        ep_rewards.append(ep_reward)
        # output information
        if outputFlag:
            print("Episode {}/{}: reward = {}".format(epi, args.test_eps, ep_reward))
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