#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from VanillaDQN import VanillaQNET

class DoubleDQNAgent():
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        optimizer,
        device,
        model_savefile="Double_Doom.pth",
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,

        
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.device=device

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:   
            print("Initializing New Model")
            self.q_net = VanillaQNET(action_size).to(self.device)
            self.target_net = VanillaQNET(action_size).to(self.device)
    
        if optimizer =='SGD':
            self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)
        elif optimizer == 'Adam':
            self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)


    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        
        
        
        

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(self.device)
            # compute the best actions in the next state using Q-network
            next_actions = torch.argmax(self.q_net(next_states), dim=1).unsqueeze(-1)
            # estimate the Q-values of the next state-action pairs using the target network
            next_state_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            next_state_values = next_state_values[not_dones]

        # compute the TD-targets using the Double DQN update rule

        q_targets = rewards.copy()
        q_targets[not_dones] = np.add(q_targets[not_dones], self.discount * next_state_values.cpu())
        q_targets = torch.from_numpy(q_targets).float().to(self.device)

        # compute the predicted Q-values for the state-action pairs
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(self.device)
        action_values = self.q_net(states)[idx].float().to(self.device)

        # compute the TD-errors and update the Q-network
        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        # update the target network
        self.update_target_net()

        # decrease the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        

