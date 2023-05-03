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
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import time
import datetime
from CGAlgs import GraphTool
from Net import Actor
from models.MHA import MHA
from models.model_efgat_v1 import GAT_EFA_Net
from models.GAT import *

class Args:
    def __init__(self):
        self.save = 1
        self.file_name = "mini_batches_standard_60"
        self.net = "GAT"
        self.epochNum = 300
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.test_prop = 0.05
        self.weight_0 = 1
        self.weight_1 = 50
        self.seed = 1
        self.curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间 
        self.result_path = self.curr_path+"/outputs/" + self.file_name + \
            '/'+self.curr_time+'/results/'  # 保存结果的路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SLActor(Actor):
    def save_net(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.preprocess.state_dict(), path+"net.pth")
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path+"actor.pth")

   
class SLTrainer:
    def __init__(self, args):
        # save args
        self.args = args
        # build data 
        file = json.load(open(args.curr_path + "\dataset_processed\{}.json".format(args.file_name)))
        states = file["states"]
        max_min_info = file["max_min_info"]
        self.train_data, self.test_data = self.preprocess_data(states, args.test_prop)
        # build model
        if args.net == "GAT":
            net = GAT(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
        elif args.net == "GAT2":
            net = GAT2(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
        elif args.net == "GAT3":
            net = GAT3(node_feature_dim=6, column_feature_dim=3, embed_dim=256, device=args.device)
        self.actor = SLActor(net, device=args.device)
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
        pass 
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
        # calculate weighted cross entropy loss
        weights = torch.FloatTensor([self.args.weight_0 if label == 0 else self.args.weight_1 for label in labels]).to(self.args.device)
        weighted_loss = torch.sum(weights * -torch.log(abs(output - (1-labels))))
        weighted_loss /= sum(weights)
        return weighted_loss

    def train(self):
        if self.args.save:
            self.logger = SummaryWriter(self.args.result_path + "pretrain_event") 
            self.logger.add_text("args", str(self.args.__dict__))
        data_epochs = self.get_epochs(self.args.epochNum) 
        iter_cnt = 0
        loss = 0.0
        self.optim.zero_grad()
        for epoch in range(self.args.epochNum):
            states = data_epochs[epoch]
            for state in states:
                output = self.actor(state)[:, 1]
                labels = torch.FloatTensor(state["labels"]).to(self.args.device)
                # calculate weighted mse loss 
                loss += self.cal_weighted_loss(output, labels)
                # optimizer step
                if (iter_cnt + 1) % self.args.batch_size == 0:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    avg_loss = loss.cpu().detach().numpy() / self.args.batch_size
                    if self.args.save:
                        self.logger.add_scalar("loss/train_loss", avg_loss, iter_cnt)
                    loss = 0.0
                    # test 
                    avg_test_loss, accuracy_1, accuracy_0, accuracy_weighted, predict_time, accuracy, precision, recall  = self.test()
                    if self.args.save:
                        self.logger.add_scalar("loss/test_loss", avg_test_loss, iter_cnt)
                        self.logger.add_scalar("accuracy/accuracy_1", accuracy_1, iter_cnt)
                        self.logger.add_scalar("accuracy/accuracy_0", accuracy_0, iter_cnt)
                        self.logger.add_scalar("accuracy/accuracy_weighted", accuracy_weighted, iter_cnt)
                        self.logger.add_scalar("accuracy/accuracy", accuracy, iter_cnt)
                        self.logger.add_scalar("accuracy/precision", precision, iter_cnt)
                        self.logger.add_scalar("accuracy/recall", recall, iter_cnt)
                        self.logger.add_scalar("output/predict_time", predict_time, iter_cnt)
                    print("Iter {}/{}: train_loss = {:.2f}, test_loss = {:.2f}".format(iter_cnt+1, self.args.epochNum*len(self.train_data), avg_loss, avg_test_loss))
                # record process
                iter_cnt += 1
            # save model each epoch
            if self.args.save:
                self.actor.save(self.args.result_path)
                self.actor.save_net(self.args.result_path)
        # save model
        if self.args.save:
            self.logger.close()

    @torch.no_grad() 
    def test(self):
        loss_list = []
        predict_time_list = []
        total_choices = []
        total_labels = []
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
            choices = [1 if np.random.rand() < prob else 0 for prob in output]
            labels = torch.FloatTensor(state["labels"]).to(self.args.device)
            total_choices += choices
            total_labels += state["labels"]
            # calculate mse loss 
            loss = self.cal_weighted_loss(output, labels).cpu().detach().numpy()
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
        accuracy = torchmetrics.Accuracy(task="binary")(torch.tensor(total_choices), torch.tensor(total_labels)).cpu().detach().numpy()
        precision = torchmetrics.Precision(task="binary")(torch.tensor(total_choices), torch.tensor(total_labels)).cpu().detach().numpy()
        recall = torchmetrics.Recall(task="binary")(torch.tensor(total_choices), torch.tensor(total_labels)).cpu().detach().numpy()
        return np.mean(loss_list), accuracy_1, accuracy_0, accuracy_weighted, np.mean(predict_time_list), accuracy, precision, recall

if __name__ == "__main__":
    args = Args()
    trainer = SLTrainer(args)
    start = time.time()
    trainer.train()
    time_cost = time.time() - start
    final_result = trainer.test()
    print("final_test_loss = {}, train_time_cost = {}".format(final_result, time_cost))

        



