import torch
import torch.nn as nn
import numpy as np
from config.hyperparameters import ACTOR_NETWORK

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, action_std_init=0.5):
        super(ActorNetwork, self).__init__()
        FC1_DIMS = ACTOR_NETWORK['FC1_DIMS']
        FC2_DIMS = ACTOR_NETWORK['FC2_DIMS']

        self.fc1 = nn.Linear(input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc_mu = nn.Linear(FC2_DIMS, n_actions)

        self.log_std = nn.Parameter(torch.ones(1, n_actions) * np.log(action_std_init))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1.0)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def get_log_prob(self, states, actions):
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return log_probs

    def get_entropy(self, states):
        dist = self.get_dist(states)
        return dist.entropy().mean()

