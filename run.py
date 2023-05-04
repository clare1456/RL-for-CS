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
        self.debug = 1 # 主线程运行而非单线程
        self.instance = "C1_2_2" # 算例 / 生成模式 random or sequence
        self.standard_file = "pretrain\dataset_processed\mini_batches_standard_60.json" # for state standardization
        self.map_change_eps = 2 # 地图更新周期, only for random / sequence
        self.limit_node_num = None # 限制算例点的个数
        self.max_step = 100 # CG最大迭代次数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 2 # 随机种子，置0则不设置随机种子
        self.process_num = mp.cpu_count() // 2 if not self.debug else 1 # 每次训练的进程数
        self.train_eps = 200 // self.process_num # 训练的回合数
        self.test_eps = 10 # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ####################################
        self.net = "GAT4" # GAT / MHA 选择 embedding 网络
        self.policy = "SAC" # SAC / PPO 选择算法
        self.gamma = 0.98  # 强化学习中的折扣因子
        self.actor_lr = 1e-4 # actor的学习率
        self.critic_lr = 1e-4 # critic的学习率
        # SAC 超参数
        self.batch_size = 5*self.max_step  # 每次训练的batch大小(SAC)
        self.buffer_size = 200*self.max_step # replay buffer的大小(SAC)
        self.minimal_size = 10*self.max_step # 开始训练的最少数据量(SAC)
        self.update_steps = self.max_step/2 # 策略更新频率(SAC)
        self.tau = 0.005 # 软更新参数(SAC)
        self.target_entropy = -1 # 目标熵(SAC)
        self.alpha_lr = 1e-3 # alpha的学习率(SAC)
        # PPO 超参数
        self.update_eps = 1 # 策略更新频率(PPO)
        self.lmbda = 0.95 # GAE中的lambda(PPO)
        self.epochs = 10 # 每次训练的epoch数(PPO)
        self.eps = 0.2 # 截断范围参数(PPO)
        ################################################################################
        
        ################################# 保存结果相关参数 ################################
        self.output_eps = 1 # 输出信息频率
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.load_policy_path = "" # 读取策略网络模型的路径
        self.load_net_path = ""#"pretrain\\model_saved\\net.pth" # 读取网络模型的路径
        self.load_actor_path = "pretrain\\model_saved\\actor_standard_GAT4.pth"# 读取actor网络模型到actor, critic
        self.result_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.model_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/models/'  # 保存模型的路径
        ################################################################################

if __name__ == "__main__":
    import Env
    # 1. get args
    args = Args()
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # 2. build global Policy
    if args.policy == "SAC":
        policy = Policy.SACPolicy(args)
    elif args.policy == "PPO":
        policy = Policy.PPOPolicy(args)
    if args.load_policy_path:
        policy.load_policy(args.load_policy_path)
    elif args.load_net_path:
        policy.load_net(args.load_net_path)
    elif args.load_actor_path:
        policy.load_actor(args.load_actor_path)
    # 3. train policy
    if args.train_eps > 0:
        # train model
        res_queue = mp.Queue()
        if args.debug == True:
            # debug 模式 (print)
            if args.policy == "SAC":
                Trainer.trainOffPolicy(policy, args, res_queue, outputFlag=True, seed=args.seed) # for debug
            elif args.policy == "PPO":
                Trainer.trainOnPolicy(policy, args, res_queue, outputFlag=True, seed=args.seed) # for debug
        else:
            # 多线程加速模式 (logger)
            writer = SummaryWriter(args.result_path + args.policy + "_" + args.net)
            writer.add_text("args", str(args.__dict__))
            policy.share_memory()
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
            terminate_process = 0
            while True:
                res = res_queue.get()
                if res is None:
                    terminate_process += 1
                    if terminate_process == args.process_num:
                        break
                else:
                    if res["tag"].startswith("result"):
                        if res["tag"] == "result/reward":
                            rew_list.append(res["value"])
                            print("Episode: {}, Reward: {}".format(len(rew_list), res["value"]))
                        writer.add_scalar(res["tag"], res["value"], len(rew_list))
                    elif res["tag"].startswith("loss") or res["tag"].startswith("output"):
                        if res["tag"] == "loss/avg_loss":
                            loss_list.append(res["value"])
                        writer.add_scalar(res["tag"], res["value"], len(loss_list))
            for p in processes:
                p.join()
            writer.close()
        # 4. save the model
        if not args.debug:
            policy.save(args.model_path)

    # 4. test
    Trainer.test(policy, args, outputFlag=True)

    
    




