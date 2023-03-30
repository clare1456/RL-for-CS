'''
File: Policy.py
Project: RL4CS
Description: Policy of RL
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

from utils.baseImport import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, actor, critic, optim, memory, args):
        super().__init__()
        # objects
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.memory = memory
        # arguments
        self.gamma = args.gamma
        self.policy_clip = args.policy_clip
        self.n_epochs = args.n_epochs
        self.gae_lambda = args.gae_lambda
        self.device = args.device
        self.loss = 0
    
    def forward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        column_num = len(obs)
        # 计算网络输出
        prob = self.actor(obs)
        value = self.critic(obs).detach().numpy()
        # 处理输出结果
        act = np.zeros(column_num)
        for i in range(column_num):
            # 依概率选择1或0
            act[i] = np.random.random() < prob[i][1]
        log_prob = torch.log(prob).detach().numpy() 
        return act, log_prob, value

    def update(self):
        for epoch_i in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                for i in batch:
                    state = torch.tensor(state_arr[i], dtype=torch.float).to(self.device)
                    old_prob = torch.tensor(old_prob_arr[i]).to(self.device)
                    action = torch.tensor(action_arr[i]).to(self.device)
                    actor_prob = self.actor(state)
                    critic_value = self.critic(state)
                    critic_value = torch.squeeze(critic_value)
                    new_prob = torch.log(actor_prob) 
                    prob_ratio = new_prob.exp() / old_prob.exp()
                    weighted_prob = advantage[i] * prob_ratio
                    weighted_clipped_prob = torch.clamp(prob_ratio, 1-self.policy_clip,
                            1+self.policy_clip)*advantage[i]
                    actor_loss = -torch.min(weighted_prob, weighted_clipped_prob).mean()
                    res = advantage[i] + values[i]
                    critic_loss = (res-critic_value)**2
                    critic_loss = critic_loss.mean()
                    self.loss += actor_loss + 0.5*critic_loss
                self.optim.zero_grad()
                self.optim.step()
        self.memory.clear()  
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')
        torch.save(self.optim.state_dict(), path + 'optim.pth')
        
    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))
        self.optim.load_state_dict(torch.load(path + 'optim.pth'))
    



