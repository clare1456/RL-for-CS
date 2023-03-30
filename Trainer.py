'''
File: Trainer.py
Project: RL4CS
Description: Trainer of RL
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

from utils.baseImport import *

class Trainer():
    def __init__(self, env, policy, memory, args):
        self.env = env
        self.policy = policy
        self.memory = memory
    
    def train(self, args):
        """ 
        训练过程
        """
        ep_rewards = [] # 记录所有回合奖励
        print("\nTraining Begin!")
        for epi in range(args.train_eps):
            ep_reward = 0
            # reset environment
            obs, info = self.env.reset()
            # interact until done
            while True:
                act, prob, val = self.policy(obs)
                next_obs, rew, done, next_info = self.env.step(act)
                self.memory.push(obs, act, prob, val, rew, done)
                ep_reward += rew
                if done:
                    break
                obs = next_obs
                info = next_info
            ep_rewards.append(ep_reward)
            # update policy
            if (epi + 1) % args.update_eps == 0:
                self.policy.update()
            # output information
            if (epi + 1) % args.output_eps == 0:
                print("Episode {}/{}: avg_reward = {}"
                      .format(epi+1, args.train_eps, 
                              sum(ep_rewards[-args.output_eps:])/args.output_eps))
        print("Training Finished!")
        return ep_rewards
    
    def test(self, args):
        ep_rewards = [] # 记录所有回合奖励
        print("\nTesting Begin!")
        for epi in range(args.test_eps):
            ep_reward = 0
            # reset environment
            obs = self.env.reset()
            # interact until done
            while True:
                act, prob, val = self.policy(obs)
                next_obs, reward, done, info = self.env.step(act)
                if done:
                    break
                obs = next_obs
            ep_rewards.append(ep_reward)
            # output information
            if epi % args.output_eps == 0:
                print("Episode {}/{}: avg_reward = {}"
                      .format(epi, args.test_eps, 
                              sum(ep_rewards[-args.output_eps:])/args.output_eps))
        print("Testing Finished!")
        return ep_rewards
            
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return self.states, self.actions, self.probs,\
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
                
    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

