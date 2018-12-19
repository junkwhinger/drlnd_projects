import random

import numpy as np
import torch
import torch.optim as optim

class DQNAgent():

    def __init__(self, params, local_network, target_network, replayBuffer):
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.use_per = params.use_per

        self.update_every = params.update_every
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.tau = params.tau
        self.device = params.device

        # initialize Q-Network
        self.qnetwork_local = local_network
        self.qnetwork_target = target_network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params.lr)

        # replay memory
        self.memory = replayBuffer

        # initialize time step
        self.t_step = 0


    def step(self, state, action, reward, next_state, done, completion):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(completion)
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        # single state to state tensor (batch size = 1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # set eval mode for local QN
        self.qnetwork_local.eval()

        # predict state value with local QN
        with torch.no_grad():  # no need to save the gradient value
            action_values = self.qnetwork_local(state)

        # set the mode of local QN back to train
        self.qnetwork_local.train()

        # e-greedy action selection
        # return greedy action if prob > eps
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())

        # return random action if prob <= eps
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        if self.use_per:
            states, actions, rewards, next_states, dones, sampled_indicies, weights = experiences

            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            p_eps = 1e-5
            td_error = Q_targets - Q_expected
            new_priorities = td_error.abs().detach().numpy() + p_eps

            loss = (td_error.pow(2) * weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.memory.update_priorities(sampled_indicies, new_priorities)
            self.optimizer.step()

        else:

            states, actions, rewards, next_states, dones = experiences

            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            loss_function = torch.nn.MSELoss()

            loss = loss_function(Q_expected, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # Q_target = TAU * local_model + (1-TAU) * target_model

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



class DDQNAgent(DQNAgent):
    def __init__(self, params, local_network, target_network, replayBuffer):
        super(DDQNAgent, self).__init__(params, local_network, target_network, replayBuffer)

    def learn(self, experiences, gamma):

        if self.use_per:
            states, actions, rewards, next_states, dones, sampled_indicies, weights = experiences

            best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            p_eps = 1e-5
            td_error = Q_targets - Q_expected
            new_priorities = td_error.abs().detach().numpy() + p_eps

            loss = (td_error.pow(2) * weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.memory.update_priorities(sampled_indicies, new_priorities)
            self.optimizer.step()

        else:

            states, actions, rewards, next_states, dones = experiences

            best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
            Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            loss_function = torch.nn.MSELoss()

            loss = loss_function(Q_expected, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

