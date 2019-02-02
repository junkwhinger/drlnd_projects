import torch
import torch.nn as nn
import numpy as np

def hidden_init(layer):
    """
    The weight initialization method used by the authors of DDPG paper
    > The other layers were initialized from uniform distributions [-1/np.sqrt(f), 1/np.sqrt(f)]
    > where f is the fan-in of the layer.
    """
    fan_in = layer.weight.data.size(0)
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor Model"""

    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.tanh_out = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.tanh_out(x)

        return x


class Critic(nn.Module):
    """Critic Network"""

    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.relu1(x)

        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        return x