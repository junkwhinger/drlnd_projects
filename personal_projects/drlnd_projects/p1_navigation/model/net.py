import torch
import torch.nn as nn
import torch.nn.functional as F

class dqn_network(nn.Module):

    def __init__(self, params):
        super(dqn_network, self).__init__()

        self.state_size = params.state_size
        self.action_size = params.action_size

        self.is_duelingNetwork = params.is_duelingNetwork
        self.network_layers = [self.state_size] + params.network_layers

        tmp = []
        for layer_idx in range(len(self.network_layers) - 1):
            tmp.append(nn.Linear(self.network_layers[layer_idx], self.network_layers[layer_idx+1]))
            tmp.append(nn.ReLU())

        self.network_front = nn.Sequential(*tmp)

        self.network_back = nn.ModuleList()
        if self.is_duelingNetwork:

            self.network_back.append(nn.Linear(self.network_layers[-1], 1)) #state

        self.network_back.append(nn.Linear(self.network_layers[-1], self.action_size))


    def forward(self, x):

        x = self.network_front(x)

        if self.is_duelingNetwork:
            v_x = self.network_back[0](x)
            a_x = self.network_back[1](x)
            average_operator = (1 / self.action_size) * a_x
            x = v_x + (a_x - average_operator)
        else:
            x = self.network_back[0](x)
        return x


