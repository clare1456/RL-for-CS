'''
File: run.py
Project: RL4CS
Description: main function, to run train and test
-----
Author: CharlesLee
Created Date: Tuesday March 7th 2023
'''

import sys
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
        self.algo_name = "SAC"  # 算法名称
        self.instance = "R101" # 算例
        self.limit_node_num = 30 # 限制算例点的个数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10 # 随机种子，置0则不设置随机种子
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 10 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.batch_size = 64  # 每次训练的batch大小
        self.buffer_size = 10000 # replay buffer的大小
        self.minimal_size = 500 # 开始训练的最少数据量
        self.hidden_dim = 256 # 隐藏层大小
        self.update_eps = 20 # 策略更新频率
        self.gamma = 0.98  # 强化学习中的折扣因子
        self.tau = 0.005 # SAC软更新参数
        self.target_entropy = -1 # SAC目标熵
        self.n_epochs = 4 # PPO每次update时学习的batch数
        self.actor_lr = 1e-3 # actor的学习率
        self.critic_lr = 1e-2 # critic的学习率
        self.alpha_lr = 1e-2 # alpha的学习率
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.output_eps = 20 # 输出信息频率
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.result_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.model_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/models/'  # 保存模型的路径
        ################################################################################

if __name__ == "__main__":
    # 1. get args
    args = Args()
    # 2. build global Policy
    policy = Policy.SACPolicy(args)
    # 3. train policy
    Trainer.trainOffPolicy(policy, args, outputFlag=True)
    # 4. test
    Trainer.test(policy, args, outputFlag=True)


    # 2. check device, set logger
    # file_name = "RL4CS_" + str(time.time())[-4:]
    # writer = SummaryWriter(args.result_path+file_name)
    # writer.add_text("args", str(args))
    # writer.close()
    
    




