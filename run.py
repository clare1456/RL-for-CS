'''
File: run.py
Project: RL4CS
Description: main function, to run train and test
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

from utils.baseImport import *
import Env
import Net
import Policy
import Trainer
import torch
import datetime
import os
import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer, OnpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

class Args:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name = "PPO"  # 算法名称
        self.instance = "R101" # 算例
        self.limit_node_num = 30 # 限制算例点的个数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10 # 随机种子，置0则不设置随机种子
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.batch_size = 5  # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.n_epochs = 4 # PPO每次update时学习的batch数
        self.learning_rate = 0.0003 # 学习率
        self.gae_lambda = 0.95 # PPO 学习参数
        self.policy_clip = 0.2 # PPO 学习参数
        self.hidden_dim = 256 # 隐藏层大小
        self.update_eps = 20 # 策略更新频率
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.output_eps = 20 # 输出信息频率
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.result_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.model_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        ################################################################################

if __name__ == "__main__":
    # 1. get args
    args = Args()
    # 2. check device, set logger
    file_name = "RL4CS_" + str(time.time())[-4:]
    writer = SummaryWriter(args.result_path+file_name)
    writer.add_text("args", str(args))
    # 3. build env
    env = Env.CGEnv(args.instance, args.limit_node_num)
    # 4. build net / optimizer / memory
    net = Net.MHA(input_dim=3+args.limit_node_num, embed_dim=128, hidden_dim=256)
    actor = Net.Actor(net, hidden_dim=128, device=args.device).to(args.device)
    critic = Net.Critic(net, hidden_dim=128, device=args.device).to(args.device)
    actor_critic = Net.ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.learning_rate)
    memory = Trainer.PPOMemory(args.batch_size)
    # 5. build Policy
    policy = Policy.PPOPolicy(actor, critic, optim, memory, args)
    # 6. build trainer
    trainer = Trainer.Trainer(env, policy, memory, args)
    # 7. train
    train_rewards = trainer.train(args)
    # 8. test
    test_rewards = trainer.test(args)
    policy.eval()
    writer.close()
    
    




