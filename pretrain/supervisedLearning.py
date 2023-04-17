'''
File: supervisedLearning.py
Project: pretrain
File Created: Monday, 10th April 2023 10:03:22 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
sys.path.append("D:\Code\RL-for-CS")
import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import time
import datetime
from CGAlgs import GraphTool
from Net import Actor
from models.MHA import MHA
from models.model_efgat_v1 import GAT_EFA_Net
from models.GAT import GAT

class SLActor(Actor):
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+"net.pth")

   
class SLTrainer:
    def __init__(self, file_name):
        self.file_name = file_name
        # set params
        self.net = "GAT"
        self.epochNum = 5
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.test_prop = 0.1
        self.test_freq = 10
        self.weight_0 = 1
        self.weight_1 = 50
        self.seed = 1
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.result_path = self.curr_path+"/outputs/" + self.file_name + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build data 
        self.mini_batches = json.load(open("D:\Code\RL-for-CS\pretrain\dataset_processed\{}.json".format(file_name)))
        self.train_data, self.test_data = self.preprocess_data(self.mini_batches, self.test_prop)
        # build model
        if self.net == "MHA":
            net = MHA(input_dim=3, embed_dim=128, hidden_dim=128, device=self.device)
        elif self.net == "GAT":
            net = GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=128)
        self.actor = SLActor(net)
        self.optim = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
    
    def _log_params(self):
        self.logger.add_text("net", self.net)  
        self.logger.add_text("learning_rate", str(self.learning_rate))  
        self.logger.add_text("test_prop", str(self.test_prop))  
        self.logger.add_text("test_freq", str(self.test_freq))  
        self.logger.add_text("seed", str(self.seed))  
    
    def preprocess_data(self, data, test_prop):
        """preprocess and split data into train_data and test_data

        Args:
            data (List[Dict]): entire dataset
            test_prop (double): proportion of test dataset

        Returns:
            train_data (List[Dict]): train datas
            test_data (List[Dict]): test datas
        """
        # preprocess data
        for state in data:
            state["columns_state"] = state["columns_features"]
            state.pop("columns_features")
            state["constraints_state"] = state["constraints_features"]
            state.pop("constraints_features")
        # randomly shuffle dataset
        np.random.seed(self.seed)
        np.random.shuffle(data)
        # split data into train/test data
        test_size = round(len(data) * test_prop)
        train_data = data[test_size:]
        test_data = data[:test_size]
        return train_data, test_data

    def get_epochs(self, epochNum):
        """get data epochs, randomly shuffle train data in each epoch

        Args:
            epochNum (int, optional): number of epoch. Defaults to 1.

        Returns:
            epochs: data epochs
        """
        np.random.seed(self.seed)
        epochs = [] 
        for i in range(epochNum):
            np.random.shuffle(self.train_data) 
            epochs.append(self.train_data.copy()) 
        return epochs

    def train(self):
        self.logger = SummaryWriter(self.result_path + "pretrain_event") 
        self._log_params()
        data_epochs = self.get_epochs(self.epochNum) 
        iter_cnt = 0
        loss = 0.0
        self.optim.zero_grad()
        for epoch in range(self.epochNum):
            states = data_epochs[epoch]
            for state in states:
                output = self.actor(state)[:, 1]
                labels = torch.FloatTensor(state["labels"])
                # calculate mse loss #? use probability or selection to calculate loss
                weights = np.array([self.weight_0 if label == 0 else self.weight_1 for label in labels])
                loss += torch.mean(torch.FloatTensor(weights) * torch.pow(output - labels, 2))
                # optimizer step
                if iter_cnt % self.batch_size == 0:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    self.logger.add_scalar("loss/train_loss", loss.detach().numpy() / self.batch_size, iter_cnt)
                    loss = 0.0
                # test 
                if iter_cnt % self.test_freq == 0:
                    mean_test_loss, accuracy_1, accuracy_0, accuracy_weighted = self.test()
                    self.logger.add_scalar("loss/test_loss", mean_test_loss, iter_cnt)
                    self.logger.add_scalar("accuracy/accuracy_1", accuracy_1, iter_cnt)
                    self.logger.add_scalar("accuracy/accuracy_0", accuracy_0, iter_cnt)
                    self.logger.add_scalar("accuracy/accuracy_weighted", accuracy_weighted, iter_cnt)
                # record process
                iter_cnt += 1

    @torch.no_grad() 
    def test(self):
        loss_list = []
        # accuracy weight : 1 -> 10, 0 -> 1
        total_num_1 = 0
        total_num_0 = 0
        correct_num_1 = 0
        correct_num_0 = 0
        for state in self.test_data:
            output = self.actor(state)[:, 1]
            choices = np.array([1 if np.random.rand() < prob else 0 for prob in output])
            labels = torch.FloatTensor(state["labels"])
            # calculate mse loss #? use probability or selection to calculate loss
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(output, labels).detach().numpy()
            loss_list.append(loss)
            # record result
            for i in range(len(choices)):
                if labels[i] == 1:
                    if choices[i] == labels[i]:
                        correct_num_1 += 1
                    total_num_1 += 1
                else:
                    if choices[i] == labels[i]:
                        correct_num_0 += 1
                    total_num_0 += 1
        # calculate accuracy
        accuracy_1 = correct_num_1 / total_num_1
        accuracy_0 = correct_num_0 / total_num_0
        accuracy_weighted = (correct_num_1 * self.weight_1 + correct_num_0 * self.weight_0) / (total_num_1 * self.weight_1 + total_num_0 * self.weight_0)
        return np.mean(loss_list), accuracy_1, accuracy_0, accuracy_weighted

if __name__ == "__main__":
    file_name = "mini_batches_5"
    trainer = SLTrainer(file_name)
    start = time.time()
    trainer.train()
    time_cost = time.time() - start
    final_result = trainer.test()
    print("final_test_loss = {}, train_time_cost = {}".format(final_result, time_cost))

        



