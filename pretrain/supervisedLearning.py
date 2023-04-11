'''
File: supervisedLearning.py
Project: pretrain
File Created: Monday, 10th April 2023 10:03:22 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from Net import Actor
from models.MHA import MHA
from models.model_efgat_v1 import GAT_EFA_Net

class SLActor(Actor):
    def save_net(self, path):
        torch.save(self.net.state_dict(), path+"net.pth")

class SLDataset:
    def __init__(self, file_name):
        self.read_data(file_name)
    
    def read_data(self, file_name):
        self.origin_data = json.load(open(file_name, 'r')) 


