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
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

class Args:
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.instance = "R101" # 算例
        self.limit_node_num = 50 # 限制算例点的个数
        self.max_step = 15 # CG最大迭代次数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10 # 随机种子，置0则不设置随机种子
        self.process_num = mp.cpu_count()//2  # 每次训练的进程数
        self.train_eps = 0 # 训练的回合数
        self.test_eps = 10 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.net = "MHA" # GAT / MHA 选择 embedding 网络
        self.policy = "PPO" # SAC / PPO 选择算法
        self.hidden_dim = 128 # 隐藏层大小
        self.gamma = 0.98  # 强化学习中的折扣因子
        self.actor_lr = 1e-3 # actor的学习率
        self.critic_lr = 1e-2 # critic的学习率
        # SAC 超参数
        self.batch_size = 5*self.max_step  # 每次训练的batch大小(SAC)
        self.buffer_size = 200*self.max_step # replay buffer的大小(SAC)
        self.minimal_size = 10*self.max_step # 开始训练的最少数据量(SAC)
        self.update_steps = self.max_step/2 # 策略更新频率(SAC)
        self.tau = 0.005 # 软更新参数(SAC)
        self.target_entropy = -1 # 目标熵(SAC)
        self.alpha_lr = 1e-3 # alpha的学习率(SAC)
        # PPO 超参数
        self.update_eps = 10 # 策略更新频率(PPO)
        self.lmbda = 0.95 # GAE中的lambda(PPO)
        self.epochs = 10 # 每次训练的epoch数(PPO)
        self.eps = 0.2 # 截断范围参数(PPO)
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.output_eps = 1 # 输出信息频率
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.load_path = "" # 读取模型的路径
        self.result_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.model_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/models/'  # 保存模型的路径
        ################################################################################

if __name__ == "__main__":
    # 1. get args
    args = Args()
    # 2. build global Policy
    if args.policy == "SAC":
        policy = Policy.SACPolicy(args)
    elif args.policy == "PPO":
        policy = Policy.PPOPolicy(args)
    if args.load_path:
        policy.load(args.load_path)
    policy.share_memory()
    # 3. train policy
    if args.train_eps > 0:
        # build writer
        writer = SummaryWriter(args.result_path + "event")
        writer.add_text("args", str(args.__dict__))
        # train model
        res_queue = mp.Queue()
        # Trainer.trainOffPolicy(policy, args, res_queue, outputFlag=True, seed=1) # for debug
        # Trainer.trainOnPolicy(policy, args, res_queue, outputFlag=True, seed=1) # for debug
        processes = []
        process_num = args.process_num
        for pi in range(process_num):
            if args.policy == "SAC":
                p = mp.Process(target=Trainer.trainOffPolicy, args=(policy, args, res_queue, False, pi+1))
            elif args.policy == "PPO":
                p = mp.Process(target=Trainer.trainOnPolicy, args=(policy, args, res_queue, False, pi+1))
            p.start()
            processes.append(p)
        rew_list = []
        loss_list = []
        while True:
            res = res_queue.get()
            if res is None:
                break
            else:
                if res["tag"] == "reward":
                    rew_list.append(res["value"])
                    writer.add_scalar(res["tag"], res["value"], len(rew_list))
                    print("Episode: {}, Reward: {}".format(len(rew_list), res["value"]))
                elif res["tag"] == "loss":
                    loss_list.append(res["value"])
                    writer.add_scalar(res["tag"], res["value"], len(loss_list))
        for p in processes:
            p.join()
        # 4. save the model
        policy.save(args.model_path)

    # 4. test
    Trainer.test(policy, args, outputFlag=True)

    # 2. check device, set logger
    # file_name = "RL4CS_" + str(time.time())[-4:]
    # writer.close()
    
    




