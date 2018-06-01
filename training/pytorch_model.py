import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import time

# 4 layer model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=1000) 
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=1000, out_features=512) 
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=512, out_features=200) 
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(in_features=200, out_features=64)
        self.relu4 = nn.LeakyReLU()
        
        self.output_actions = nn.Linear(in_features=64, out_features=4)
        self.output_parameters = nn.Linear(in_features=64, out_features=6)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        
        out_actions = self.output_actions(out)
        out_parameters = self.output_parameters(out)
        return out_actions, out_parameters


# 5 layer model 
class NeuralNet2(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet2, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=1000) 
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=1000, out_features=750) 
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=750, out_features=512) 
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(in_features=512, out_features=200) 
        self.relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(in_features=200, out_features=64)
        self.relu5 = nn.LeakyReLU()
        
        self.output_actions = nn.Linear(in_features=64, out_features=4)
        self.output_parameters = nn.Linear(in_features=64, out_features=6)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        
        out_actions = self.output_actions(out)
        out_parameters = self.output_parameters(out)
        return out_actions, out_parameters
