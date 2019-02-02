import random
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from utils import OrnsteinUhlenbeckProcess, ReplayBuffer

class DDPGAgent():
    def __init__(self, random_seed, device, n_agents, state_size, action_size,
                 buffer_size, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay, checkpoint_folder="ckpt"):

        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        self.seed = random.seed(random_seed)
        self.DEVICE = device
        self.n_agents = n_agents

        self.state_size = state_size
        self.action_size = action_size

        # hyper-paramters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.checkpoint_folder = checkpoint_folder

        # local and target actors and critics
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_local_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        self.critic_local = Critic(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_local_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)

        # Ornstein-Uhlenbeck noise process
        self.noise = OrnsteinUhlenbeckProcess((n_agents, action_size), random_seed)

        # Replay Buffer
        self.memory = ReplayBuffer(random_seed, device, action_size, self.buffer_size, self.batch_size)

    def step(self, state, action, reward, next_state, done):
        """
        1. append experience to the replay memory
        2. take random samples from the replay memory
        3. learn from those samples
        """

        # 1
        for i in range(self.n_agents):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

        # 2
        if len(self.memory) > self.batch_size:
            samples_drawn = self.memory.sample()

            # 3
            self.learn(samples_drawn)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            ohnoise = self.noise.sample()
            action += ohnoise
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset_states()

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # update critic
        # 1. use actor_target to predict the next actions for the next states
        predicted_next_actions = self.actor_target(next_states)

        # 2. use critic_target to estimate the value of the predicted next actions
        estimated_values_of_predicted_next_actions = self.critic_target(next_states, predicted_next_actions)

        # 3. calculate Q targets with the immediate reward and discounted value of #2
        # if the next state is done, discounted #2 is 0
        Q_targets = rewards + (self.gamma * estimated_values_of_predicted_next_actions * (1 - dones))

        # 4. use critic_local to estimate Q expected from states and actions of the sample experience
        Q_expected = self.critic_local(states, actions)

        # 5. calculate MSE loss between Q expected and Q targets
        # Q_targets based on critic_target, Q_expected based on critic_local
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # 6. minimise #5 using optimizer and backward to update critic_local
        # that is learning the value of current state and action
        self.critic_local_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_local_optimizer.step()

        # update actor
        # 1 use actor_local to predict the next action without Ornstein-Uhlenbeck noise
        # this shows actor_local's current progress
        predicted_actions_wo_noise = self.actor_local(states)

        # 2 use critic_local to estimate the value of actor_local's predicted actions
        # take mean of the values of (states, actions)
        estimated_value_of_predicted_actions = self.critic_local(states, predicted_actions_wo_noise).mean()

        # 3 loss is the minus of #2 as we use gradient descient to maximize the estimated value
        actor_loss = -estimated_value_of_predicted_actions

        # 4 minimize #3 to update actor_local
        self.actor_local_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_local_optimizer.step()

        # update target network using soft update
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_nn, target_nn):
        for target_parameter, local_parameter in zip(target_nn.parameters(), local_nn.parameters()):
            target_parameter.data.copy_(self.tau * local_parameter.data + (1.0 - self.tau) * target_parameter.data)

    def save_checkpoints(self, suffix):
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        torch.save(self.actor_local.state_dict(), self.checkpoint_folder + '/actor_{}.pth'.format(str(suffix)))
        torch.save(self.critic_local.state_dict(), self.checkpoint_folder + '/critic_{}.pth'.format(str(suffix)))