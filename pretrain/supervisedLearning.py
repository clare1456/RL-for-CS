'''
File: supervisedLearning.py
Project: pretrain
File Created: Monday, 10th April 2023 10:03:22 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import time
import gurobipy as gp
from Net import Actor
from models.MHA import MHA
from models.model_efgat_v1 import GAT_EFA_Net

class SLActor(Actor):
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+"net.pth")


class SLDataLoader:
    def __init__(self, file_name, test_prop=0.1, seed=1):
        # set random seed
        self.seed = seed
        # build MILPSolver
        self.milp_solver = MILPSolver()
        # read data and process data
        data = self._read_data(file_name)
        # split data into train/test data
        self.train_data, self.test_data = self._split_data(data=data, test_prop=test_prop)
    
    def _read_data(self, file_name):
        """read column generation process data from json file and process 

        Args:
            file_name (string): file path in string form
        """
        origin_data = json.load(open(file_name, 'r')) 
        #todo, split data into mini-batch, and get nodeNum
        self.nodeNum
        self._get_labels(mini_batches)
        return self._minibatch2state(mini_batches)
    
    def _get_labels(self, mini_batches):
        """add labels to mini_batches with MILP

        Args:
            mini_batches (List[Dict]): iteration data for MILP, ["present_columns", "new_columns"], column: Dict["distance", "onehot_path"]
        """
        for mini_batch in mini_batches:
            labels = self.milp_solver(mini_batch["present_columns"], mini_batch["new_columns"])
            mini_batch["labels"] = labels

    def _minibatch2state(self, mini_batches):
        """transfer mini_batches to states 

        Args:
            mini_batches (List[Dict]): iteration data for MILP, ["present_columns", "new_columns"], column: Dict["distance", "onehot_path"]

        Returns:
            states (List[Dict]): ["columns_features", "constraints_feature"]
        """
        # todo, 根据师兄产生数据设计
        return states
    
    def _split_data(self, data, test_prop=0.1):
        """split data into train_data and test_data

        Args:
            data (List[Dict]): entire dataset
            test_prop (double): proportion of test dataset

        Returns:
            train_data (List[Dict]): train datas
            test_data (List[Dict]): test datas
        """
        # 1. randomly shuffle dataset
        np.random.seed(self.seed)
        np.random.shuffle(data)
        # 2. split data into train/test data
        test_size = round(len(data) * test_prop)
        train_data = data[test_size:]
        test_data = data[:test_size]
        return train_data, test_data

    def get_epochs(self, epochNum=1):
        """get data epochs, randomly shuffle train data in each epoch

        Args:
            epochNum (int, optional): number of epoch. Defaults to 1.

        Returns:
            epochs: data epochs
        """
        np.random.seed(self.seed)
        epochs = [] 
        for i in range(epochNum):
            random.shuffle(self.train_data) 
            epochs.append(self.train_data.copy()) 
        return epochs

    def get_test_data(self):
        return self.test_data

class MILPSolver:
    def __init__(self, epsilon1=0.001, epsilon2=0.1):
        # weight of minimize columnNum
        self.epsilon1 = epsilon1 # coef for present columns
        self.epsilon2 = epsilon2 # coef for new columns

    def solve(self, present_columns, new_columns):
        """ build model """
        # get data
        columns = present_columns + new_columns
        epsilons = ([self.epsilon1 for _ in range(len(present_columns))] 
                    + [self.epsilon2 for _ in range(len(new_columns))])
        nodeNum = len(columns["0"]["path"])
        columnNum = len(columns)
        # building model
        MILP = gp.Model()
        # add columns into MILP
        ## add variables
        theta_list = list(range(columnNum))
        theta = MILP.addVars(theta_list, vtype="C", name="theta")
        y = MILP.addVars(theta_list, vtype="I", name="y")
        ## set objective
        MILP.setObjective(gp.quicksum((theta[i] * columns[f"{i}"]["distance"] + y[i] * epsilons[i]) for i in range(columnNum)), GRB.MINIMIZE)
        ## set constraints
        MILP.addConstrs(gp.quicksum(theta[i] * columns[f"{i}"]["path"][f"{j}"] for i in range(columnNum)) == 1 for j in range(1, nodeNum))
        MILP.addConstrs(theta[i] <= y[i] for i in range(columnNum))
        ## set params
        MILP.setParam("OutputFlag", 0)

        """ solve model """
        MILP.optimize()
        labels = []
        for i in range(columnNum):
            labels.append(round(y[i].X))
        return labels
    

class SLTrainer:
    def __init__(self, file_name):
        # set params
        self.net = "GAT"
        self.epochNum = 10
        self.learning_rate = 1e-5
        self.test_prop = 0.1
        self.test_freq = 10
        self.seed = 1
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.result_path = self.curr_path+"/outputs/" + self.instance + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        # build data loader
        self.data_loader = SLDataLoader(file_name, test_prop=self.test_prop, seed=self.seed)
        self.nodeNum = self.data_loader.nodeNum
        # build model
        if self.net== "MHA":
            net = MHA(input_dim=3+self.nodeNum, embed_dim=128)
        elif self.net== "GAT":
            net = GAT_EFA_Net(nfeat=128, nedgef=128, embed_dim=128, nhid = 64, maxNodeNum=self.nodeNum)
        self.actor = SLActor(net)
        self.optim = torch.optim.Adam(self.actor.state_dict(), lr=self.learning_rate)
    
    def _log_params(self):
        self.logger.add_text("net", self.net)  
        self.logger.add_text("learning_rate", self.learning_rate)  
        self.logger.add_text("test_prop", self.test_prop)  
        self.logger.add_text("test_freq", self.test_freq)  
        self.logger.add_text("seed", self.seed)  
    
    def train(self):
        self.logger = SummaryWriter(self.result_path + "pretrain_event") 
        self._log_params()
        data_epochs = self.data_loader.get_epochs(self.epochNum) 
        iter_cnt = 0
        for epoch in range(self.epochNum):
            states = data_epochs[epoch]
            for state in states:
                output = self.actor(state)
                labels = state["labels"]
                # calculate mse loss #? use probability or selection to calculate loss
                loss = torch.nn.MSELoss(output, labels) 
                # optimizer step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # test 
                if iter_cnt % self.test_freq == 0:
                    mean_test_loss = self.test()
                    self.logger.add_scalar("loss/test_loss", mean_test_loss, iter_cnt)
                # record process
                self.logger.add_scalar("loss/train_loss", loss.detach().numpy(), iter_cnt)
                iter_cnt += 1
    
    def test(self):
        test_data = self.data_loader.get_test_data()
        loss_list = []
        for state in test_data:
            output = self.actor(state)
            labels = state["labels"]
            # calculate mse loss #? use probability or selection to calculate loss
            loss = torch.nn.MSELoss(output, labels).detach().numpy()
            loss_list.append(loss)
        return np.mean(loss_list)

if __name__ == "__main__":
    file_name = ""
    trainer = SLTrainer(file_name)
    start = time.time()
    trainer.train()
    time_cost = time.time() - start
    final_result = trainer.test()
    print("final_test_loss = {}, train_time_cost = {}".format(final_result, time_cost))

        



