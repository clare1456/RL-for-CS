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
        self.debug = 1 
        self.instance = "R1_2_1" 
        self.standard_file = "pretrain\dataset_processed\mini_batches_standard_60.json" 
        self.map_change_eps = 10 
        self.limit_node_num = 405 
        self.max_step = 100 
        self.min_select_num = 100 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.seed = 2 
        self.process_num = mp.cpu_count() // 2 if not self.debug else 1 
        self.train_eps = 20000 // self.process_num 
        self.test_eps = 10 
        
        self.net = "GAT4" 
        self.policy = "SAC" 
        self.gamma = 0.98  
        self.actor_lr = 1e-5 
        self.critic_lr = 1e-5 
        
        self.batch_size = self.limit_node_num  
        self.buffer_size = 200*self.limit_node_num 
        self.minimal_size = 10*self.limit_node_num 
        self.update_steps = self.limit_node_num/2 
        self.tau = 0.001 
        self.target_entropy = -1 
        self.alpha_lr = 1e-3 

        self.update_eps = 1 
        self.lmbda = 0.95 
        self.epochs = 10 
        self.eps = 0.2 
        
        self.output_eps = 1 
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) 
        self.load_policy_path = "" 
        self.load_net_path = "pretrain\\model_saved\\net_standard_new.pth"
        self.load_actor_path = ""
        self.result_path = self.curr_path+"\\outputs\\" + self.instance + \
            '\\'+self.curr_time+'\\results\\'  
        self.model_path = self.curr_path+"\\outputs\\" + self.instance + \
            '\\'+self.curr_time+'\\models\\'  

if __name__ == "__main__":
    import Env
    
    args = Args()
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
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
    
    if args.train_eps > 0:
        
        res_queue = mp.Queue()
        if args.debug == True:
            
            if args.policy == "SAC":
                Trainer.trainOffPolicy(policy, args, res_queue, outputFlag=True, seed=args.seed) 
            elif args.policy == "PPO":
                Trainer.trainOnPolicy(policy, args, res_queue, outputFlag=True, seed=args.seed) 
        else:
            
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
        
        if not args.debug:
            policy.save(args.model_path)

    
    Trainer.test(policy, args, outputFlag=True)

    
    




