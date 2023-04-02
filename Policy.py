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
import torch.nn.functional as F
import Net

class SACPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        # trainable objects
        net = Net.MHA(input_dim=3+args.limit_node_num, embed_dim=128, hidden_dim=256)
        self.actor = Net.Actor(net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_1 = Net.Critic(net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_2 = Net.Critic(net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1 = Net.Critic(net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_2 = Net.Critic(net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1.load_state_dict(state_dict=self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(state_dict=self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        # arguments
        self.target_entropy = args.target_entropy
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device

    def forward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        column_num = len(obs)
        # 计算网络输出
        prob = self.actor(obs)
        # 处理输出结果
        act = np.zeros(column_num)
        for i in range(column_num):
            # 依概率选择1或0
            act[i] = np.random.random() < prob[i][1]
        return act
    
    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, reward, next_state, done):
        next_prob = self.actor(next_state)
        next_log_prob = torch.log(next_prob + 1e-8)
        entropy = -torch.sum(next_prob * next_log_prob) 
        q1_value = self.target_critic_1(next_state)
        q2_value = self.target_critic_2(next_state)
        min_qvalue = torch.sum(next_prob * torch.min(q1_value, q2_value))
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = reward + self.gamma * next_value * (1 - done)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, buffer, critic_1_optim, critic_2_optim, actor_optim, alpha_optim):
        states, actions, rewards, next_states, dones = buffer.sample() 
        
        for i in range(buffer.batch_size):
            # critic loss
            td_target = self.calc_target(rewards[i], next_states[i], dones[i])
            critic_1_q_value = torch.sum(self.critic_1(states[i])[range(len(actions[i])), actions[i]]) 
            critic_1_loss = F.mse_loss(critic_1_q_value, td_target.detach())
            critic_1_optim.zero_grad()
            critic_1_loss.backward()
            critic_1_optim.step()
            critic_2_q_value = torch.sum(self.critic_2(states[i])[range(len(actions[i])), actions[i]]) 
            critic_2_loss = F.mse_loss(critic_2_q_value, td_target.detach())
            critic_2_optim.zero_grad()
            critic_2_loss.backward()
            critic_2_optim.step()
            # actor loss
            prob = self.actor(states[i])
            log_prob = torch.log(prob + 1e-8)
            entropy = -torch.sum(prob * log_prob) 
            q1_value = self.critic_1(states[i])
            q2_value = self.critic_2(states[i])
            min_qvalue = torch.sum(prob * torch.min(q1_value, q2_value)) 
            actor_loss = torch.mean( - self.log_alpha.exp() * log_prob * entropy - min_qvalue)
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            # alpha loss
            alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
        # soft update target network
        self.soft_update(self.critic_1, self.target_critic_1) 
        self.soft_update(self.critic_2, self.target_critic_2) 
        
    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic_1.state_dict(), path + 'critic_1.pth')
        torch.save(self.critic_2.state_dict(), path + 'critic_2.pth')
        torch.save(self.optim.state_dict(), path + 'optim.pth')
        
    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic_1.load_state_dict(torch.load(path + 'critic_1.pth'))
        self.critic_2.load_state_dict(torch.load(path + 'critic_2.pth'))
        self.optim.load_state_dict(torch.load(path + 'optim.pth'))
    



