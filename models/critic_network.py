import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config.hyperparameters import CRITIC_NETWORK

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        FC1_DIMS = CRITIC_NETWORK['FC1_DIMS']
        FC2_DIMS = CRITIC_NETWORK['FC2_DIMS']

        self.fc1 = nn.Linear(input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.v = nn.Linear(FC2_DIMS, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        v = self.v(x)
        return v

