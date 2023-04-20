'''
File: supervisedLearning.py
Project: pretrain
File Created: Monday, 10th April 2023 10:03:22 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
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

class Args:
    def __init__(self):
        self.save = 0
        self.file_name = "mini_batches_1"
        self.net = "GAT"
        self.epochNum = 20
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.test_prop = 0.1
        self.test_freq = 2
        self.weight_0 = 1
        self.weight_1 = 15
        self.seed = 1
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.result_path = self.curr_path+"/outputs/" + self.file_name + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SLActor(Actor):
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+"net.pth")

   
class SLTrainer:
    def __init__(self, args):
        # save args
        self.args = args
        # build data 
        self.mini_batches = json.load(open(args.curr_path + "\dataset_processed\{}.json".format(args.file_name)))
        self.train_data, self.test_data = self.preprocess_data(self.mini_batches, args.test_prop)
        # build model
        if args.net == "MHA":
            net = MHA(input_dim=3, embed_dim=128, hidden_dim=128, device=self.device)
        elif args.net == "GAT":
            net = GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=128)
        self.actor = SLActor(net)
        self.optim = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
    
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
        np.random.seed(self.args.seed)
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
        np.random.seed(args.seed)
        epochs = [] 
        for i in range(epochNum):
            np.random.shuffle(self.train_data) 
            epochs.append(self.train_data.copy()) 
        return epochs

    def cal_weighted_loss(self, output, labels):
        weights = np.array([self.args.weight_0 if label == 0 else self.args.weight_1 for label in labels])
        weighted_loss = torch.mean(torch.pow(torch.FloatTensor(weights) * (output - labels), 2))
        return weighted_loss

    def train(self):
        if self.args.save:
            self.logger = SummaryWriter(self.result_path + "pretrain_event") 
            self.logger.add_text("args", self.args.__dict__)
        data_epochs = self.get_epochs(self.args.epochNum) 
        iter_cnt = 0
        loss = 0.0
        self.optim.zero_grad()
        for epoch in range(self.args.epochNum):
            states = data_epochs[epoch]
            for state in states:
                output = self.actor(state)[:, 1]
                labels = torch.FloatTensor(state["labels"])
                # calculate weighted mse loss 
                loss += self.cal_weighted_loss(output, labels)
                # optimizer step
                if iter_cnt % self.args.batch_size == 0:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    avg_loss = loss.detach().numpy() / self.args.batch_size
                    if self.args.save:
                        self.logger.add_scalar("loss/train_loss", avg_loss, iter_cnt)
                    print("Iter {}: train_loss == {:.2f}".format(iter_cnt, avg_loss))
                    loss = 0.0
                    # test 
                    if (iter_cnt // self.args.batch_size) % self.args.test_freq == 0:
                        avg_test_loss, accuracy_1, accuracy_0, accuracy_weighted, predict_time = self.test()
                        if self.args.save:
                            self.logger.add_scalar("loss/test_loss", avg_test_loss, iter_cnt)
                            self.logger.add_scalar("accuracy/accuracy_1", accuracy_1, iter_cnt)
                            self.logger.add_scalar("accuracy/accuracy_0", accuracy_0, iter_cnt)
                            self.logger.add_scalar("accuracy/accuracy_weighted", accuracy_weighted, iter_cnt)
                            self.logger.add_scalar("output/predict_time", predict_time, iter_cnt)
                        print("Iter {}:                       test_loss == {:.2f}".format(iter_cnt, avg_test_loss))
                # record process
                iter_cnt += 1
        # final optimize
        if iter_cnt % self.args.batch_size > 0:
            loss.backward()
            self.optim.step()
            if self.args.save:
                self.logger.add_scalar("loss/train_loss", loss.detach().numpy() / (iter_cnt % self.batch_size), iter_cnt)

    @torch.no_grad() 
    def test(self):
        loss_list = []
        predict_time_list = []
        # accuracy weight 
        total_num_1 = 0
        total_num_0 = 0
        correct_num_1 = 0
        correct_num_0 = 0
        for state in self.test_data:
            # predict and record predict time
            time1 = time.time()
            output = self.actor(state)[:, 1]
            time2 = time.time()
            predict_time_list.append(time2-time1)
            choices = np.array([1 if np.random.rand() < prob else 0 for prob in output])
            labels = torch.FloatTensor(state["labels"])
            # calculate mse loss 
            loss = self.cal_weighted_loss(output, labels).detach().numpy()
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
        accuracy_weighted = ((correct_num_1 * self.args.weight_1 + correct_num_0 * self.args.weight_0) 
                             / (total_num_1 * self.args.weight_1 + total_num_0 * self.args.weight_0))
        return np.mean(loss_list), accuracy_1, accuracy_0, accuracy_weighted, np.mean(predict_time_list)

if __name__ == "__main__":
    args = Args()
    trainer = SLTrainer(args)
    start = time.time()
    trainer.train()
    time_cost = time.time() - start
    final_result = trainer.test()
    print("final_test_loss = {}, train_time_cost = {}".format(final_result, time_cost))

        



