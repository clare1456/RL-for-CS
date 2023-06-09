from utils.baseImport import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import Net
import os
import copy

class basePolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net_name = args.net
        if args.net == "MHA":
            self.net = Net.MHA(input_dim=3, embed_dim=128, hidden_dim=128, device=args.device)
        elif args.net == "GAT":
            self.net = Net.GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
        elif args.net == "GAT4":
            self.net = Net.GAT4(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
    
    def forward(self, state):
        column_num = len(state["columns_state"])
        prob = self.actor(state)
        act = np.zeros(column_num, dtype=np.int)
        for i in range(column_num):
            act[i] = np.random.random() < prob[i][1]
        if sum(act) == 0:
            act[0] = 1
        return act

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path + '{}.pth'.format(self.alg_name))
        
    def load_policy(self, path):
        self.load_state_dict(torch.load(path, map_location=self.args.device))

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.args.device))

class SACPolicy(basePolicy):
    def __init__(self, args):
        super(SACPolicy, self).__init__(args)
        self.alg_name = "{}_SACPolicy".format(args.policy)
        self.actor = Net.Actor(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_1 = Net.Critic(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_2 = Net.Critic(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1 = Net.Critic(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_2 = Net.Critic(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1.load_state_dict(state_dict=self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(state_dict=self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.target_entropy = args.target_entropy
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device

    def calc_target(self, reward, next_state, done):
        next_prob = self.actor(next_state)
        next_log_prob = torch.log(next_prob + 1e-8)
        entropy = -torch.mean(torch.sum(next_prob * next_log_prob, dim=-1)) 
        q1_value = self.target_critic_1(next_state)
        q2_value = self.target_critic_2(next_state)
        min_qvalue = torch.sum(torch.sum(next_prob * torch.min(q1_value, q2_value), dim=-1))
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
        loss = 0 
        actor_loss = 0
        critic_1_loss = 0
        critic_2_loss = 0
        alpha_loss = 0
        avg_qvalue = 0
        avg_prob = 0
        for i in range(buffer.batch_size):
            td_target = self.calc_target(rewards[i], next_states[i], dones[i])
            critic_1_q_value = torch.sum(self.critic_1(states[i])[range(len(actions[i])), actions[i]]) 
            critic_1_loss += F.mse_loss(critic_1_q_value, td_target.detach()) / buffer.batch_size
            critic_2_q_value = torch.sum(self.critic_2(states[i])[range(len(actions[i])), actions[i]]) 
            critic_2_loss += F.mse_loss(critic_2_q_value, td_target.detach()) / buffer.batch_size
            prob = self.actor(states[i])
            log_prob = torch.log(prob + 1e-8)
            entropy = -torch.mean(torch.sum(prob * log_prob, dim=-1))
            q1_value = self.critic_1(states[i])
            q2_value = self.critic_2(states[i])
            min_qvalue = torch.sum(prob * torch.min(q1_value, q2_value)) 
            actor_loss += (-self.log_alpha.exp() * entropy - min_qvalue) / buffer.batch_size
            alpha_loss += (entropy - self.target_entropy).detach() * self.log_alpha.exp()
            loss = (critic_1_loss + critic_2_loss + actor_loss + alpha_loss) / buffer.batch_size
            avg_qvalue += min_qvalue.detach().numpy()
            avg_prob += torch.mean(prob[:, 1]).detach().numpy()
        critic_1_optim.zero_grad()
        critic_1_loss.backward()
        critic_2_optim.zero_grad()
        critic_2_loss.backward()
        actor_optim.zero_grad()
        actor_loss.backward()
        alpha_optim.zero_grad()
        alpha_loss.backward()
        critic_1_optim.step()
        critic_2_optim.step()
        actor_optim.step()
        alpha_optim.step()
        avg_actor_loss = actor_loss.detach().numpy()
        avg_critic_loss = (critic_1_loss + critic_2_loss).detach().numpy()
        avg_alpha_loss = alpha_loss.detach().numpy()
        avg_loss = loss.detach().numpy()
        avg_qvalue /= buffer.batch_size
        avg_prob /= buffer.batch_size
        self.soft_update(self.critic_1, self.target_critic_1) 
        self.soft_update(self.critic_2, self.target_critic_2) 
        loss_info = {
            "loss/avg_loss": avg_loss,
            "loss/avg_actor_loss": avg_actor_loss,
            "loss/avg_critic_loss": avg_critic_loss,
            "loss/avg_alpha_loss": avg_alpha_loss,
            "output/avg_critic_qvalue": avg_qvalue,
            "output/avg_actor_prob": avg_prob,
        }
        return loss_info

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.args.device))
        self.critic_1.load_state_dict(torch.load(path, map_location=self.args.device))
        self.critic_2.load_state_dict(torch.load(path, map_location=self.args.device))
        self.target_critic_1.load_state_dict(torch.load(path, map_location=self.args.device))
        self.target_critic_2.load_state_dict(torch.load(path, map_location=self.args.device))

class PPOPolicy(basePolicy):
    def __init__(self, args):
        super(PPOPolicy, self).__init__(args)
        self.alg_name = "{}_PPOPolicy".format(args.policy)
        self.actor = Net.Actor(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.critic = Net.Critic(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.epochs = args.epochs
        self.eps= args.eps
        self.device = args.device
    
    def compute_advantage(self, gamma, lmbda, td_deltas):
        advantage_list = []
        advantage = 0.0
        for delta in td_deltas[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return advantage_list
    
    def get_critic_value(self, critic, state):
        return torch.sum(torch.max(critic(state), dim=1).values)

    def update(self, transition_dict, actor_optim, critic_optim):
        states = transition_dict["states"]
        actions = transition_dict["actions"]
        rewards = transition_dict["rewards"]
        next_states = transition_dict["next_states"]
        dones = transition_dict["dones"] 
        rewards = torch.FloatTensor(rewards).to(self.device)
        td_targets = []
        for i in range(len(states)):
            td_target = rewards[i] + self.gamma * self.get_critic_value(self.critic, next_states[i]) * (1 - dones[i])
            td_targets.append(td_target)
        td_deltas = []
        for i in range(len(states)):
            td_deltas.append(td_targets[i] - self.get_critic_value(self.critic, states[i]))
        advantages = self.compute_advantage(self.gamma, self.lmbda, td_deltas)
        old_log_probs = []
        for i in range(len(states)):
            prob = self.actor(states[i])
            old_log_probs.append(torch.log(prob + 1e-8).detach()[range(len(actions[i])), actions[i]])
        td_deltas = torch.FloatTensor(td_deltas).to(self.device)
        td_targets = torch.FloatTensor(td_targets).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        avg_loss = 0.0
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        avg_qvalue = 0.0
        avg_prob = 0.0
        for _ in range(self.epochs):
            actor_optim.zero_grad()
            critic_optim.zero_grad()
            actor_loss = 0.0
            critic_loss = 0.0
            qvalue_sum = 0.0
            prob_sum = 0.0
            for i in range(len(states)):
                prob = self.actor(states[i])
                log_prob = torch.log(prob + 1e-8)[range(len(actions[i])), actions[i]]
                ratio = torch.mean(torch.exp(log_prob - old_log_probs[i]))
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages[i] 
                actor_loss += -torch.min(surr1, surr2) 
                critic_output = self.get_critic_value(self.critic, states[i])
                critic_loss += F.mse_loss(critic_output, td_targets[i].detach()) 
                qvalue_sum += critic_output.detach().numpy()
                prob_sum += torch.mean(prob[:, 1]).detach().numpy()
            actor_loss /= len(states) 
            critic_loss /= len(states) 
            qvalue_sum /= len(states)
            prob_sum /= len(states)
            actor_loss.backward()
            critic_loss.backward()
            actor_optim.step()
            critic_optim.step() 
            avg_actor_loss += actor_loss.detach().numpy()
            avg_critic_loss += critic_loss.detach().numpy()
            avg_loss += avg_actor_loss + avg_critic_loss
            avg_qvalue += qvalue_sum
            avg_prob += prob_sum
        avg_actor_loss /= self.epochs
        avg_critic_loss /= self.epochs
        avg_loss /= self.epochs
        avg_qvalue /= self.epochs
        avg_prob /= self.epochs
        loss_info = {
            "loss/avg_loss": avg_loss,
            "loss/avg_actor_loss": avg_actor_loss,
            "loss/avg_critic_loss": avg_critic_loss,
            "output/avg_critic_qvalue": avg_qvalue,
            "output/avg_actor_prob": avg_prob,
        }
        return loss_info
            
    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.args.device))
        self.critic.load_state_dict(torch.load(path, map_location=self.args.device))

class SACPolicy_choose_one(SACPolicy):
    def __init__(self, args):
        super(SACPolicy, self).__init__(args)
        self.alg_name = "{}_SACPolicy_choose_one".format(args.policy)
        self.actor = Net.Actor_choose_one(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_1 = Net.Critic_choose_one(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.critic_2 = Net.Critic_choose_one(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1 = Net.Critic_choose_one(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_2 = Net.Critic_choose_one(self.net, hidden_dim=128, device=args.device).to(args.device)
        self.target_critic_1.load_state_dict(state_dict=self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(state_dict=self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.target_entropy = args.target_entropy
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device

    def calc_target(self, reward, next_state, done):
        next_prob = self.actor(next_state)
        next_log_prob = torch.log(next_prob + 1e-8)
        entropy = -torch.sum(next_prob * next_log_prob, dim=-1)
        q1_value = self.target_critic_1(next_state)
        q2_value = self.target_critic_2(next_state)
        min_qvalue = torch.sum(next_prob * torch.min(q1_value, q2_value), dim=-1)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = reward + self.gamma * next_value * (1 - done)
        return td_target

    def update(self, buffer, critic_1_optim, critic_2_optim, actor_optim, alpha_optim):
        states, actions, rewards, next_states, dones = buffer.sample() 
        avg_loss = 0 
        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_alpha_loss = 0
        avg_qvalue = 0
        avg_prob = 0
        for i in range(buffer.batch_size):
            td_target = self.calc_target(rewards[i], next_states[i], dones[i])
            critic_1_q_value = torch.sum(self.critic_1(states[i])[np.argwhere(actions[i] == 1)[0]]) 
            critic_1_loss = F.mse_loss(critic_1_q_value, td_target.detach()) / buffer.batch_size
            critic_1_optim.zero_grad()
            critic_1_loss.backward()
            critic_1_optim.step()
            critic_2_q_value = torch.sum(self.critic_2(states[i])[np.argwhere(actions[i] == 1)[0]]) 
            critic_2_loss = F.mse_loss(critic_2_q_value, td_target.detach()) / buffer.batch_size
            critic_2_optim.zero_grad()
            critic_2_loss.backward()
            critic_2_optim.step()
            prob = self.actor(states[i])
            log_prob = torch.log(prob + 1e-8)
            entropy = -torch.sum(prob * log_prob, dim=-1)
            q1_value = self.critic_1(states[i])
            q2_value = self.critic_2(states[i])
            min_qvalue = torch.sum(prob * torch.min(q1_value, q2_value), dim=-1)
            actor_loss = (-self.log_alpha.exp() * entropy - min_qvalue) / buffer.batch_size
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            alpha_loss = (entropy - self.target_entropy).detach() * self.log_alpha.exp() / buffer.batch_size
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            avg_actor_loss += actor_loss.detach().numpy()
            avg_critic_loss += (critic_1_loss + critic_2_loss).detach().numpy()
            avg_alpha_loss += alpha_loss.detach().numpy()
            avg_loss += avg_actor_loss + avg_critic_loss + avg_alpha_loss
            avg_qvalue += min_qvalue.detach().numpy()
            avg_prob += torch.mean(prob).detach().numpy()
        avg_actor_loss /= buffer.batch_size
        avg_critic_loss /= buffer.batch_size
        avg_alpha_loss /= buffer.batch_size
        avg_loss /= buffer.batch_size
        avg_qvalue /= buffer.batch_size
        avg_prob /= buffer.batch_size
        self.soft_update(self.critic_1, self.target_critic_1) 
        self.soft_update(self.critic_2, self.target_critic_2) 
        loss_info = {
            "loss/avg_loss": avg_loss,
            "loss/avg_actor_loss": avg_actor_loss,
            "loss/avg_critic_loss": avg_critic_loss,
            "loss/avg_alpha_loss": avg_alpha_loss,
            "output/avg_critic_qvalue": avg_qvalue,
            "output/avg_actor_prob": avg_prob,
        }
        return loss_info

    def forward(self, state):
        column_num = len(state["columns_state"])
        prob = self.actor(state)
        act = np.zeros(column_num, dtype=np.int)
        choose_one = np.random.choice(column_num, p=prob.detach().numpy())
        act[choose_one] = 1
        return act