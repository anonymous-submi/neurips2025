# prerequisites
import copy
import glob
import sys
import os
import time
from collections import deque

import gym

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BehaviorCloning:
    def __init__(self, policy,  device, batch_size=None, lr=None, annotated_dataset=None,
         num_batches=np.float('inf'), training_data_split=None, envs=None, ensemble_size=None, use_log_loss=False):
        super(BehaviorCloning, self).__init__()

        self.actor_critic = policy

        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.device = device
        self.lr = lr
        self.use_log_loss = use_log_loss
        self.batch_size = batch_size

        datasets = annotated_dataset.load_demo_data(training_data_split, batch_size, ensemble_size)
        self.trdata = datasets['trdata']
        self.tedata = datasets['tedata']

        self.num_batches = num_batches
        self.action_space = envs.action_space

    def update(self, update=True, data_loader_type=None):
        if data_loader_type == 'train':
            data_loaders = self.trdata
        elif data_loader_type == 'test':
            data_loaders = self.tedata
        else:
            raise Exception("Unknown Data loader specified")

        total_loss = 0
        if not data_loaders:
            return 0
        for batch_idx, data_loaders_batch in enumerate(zip(*data_loaders), 1):
            states = torch.stack([batch[0] for batch in data_loaders_batch]).to(self.device)
            expert_actions = torch.stack([batch[1] for batch in data_loaders_batch]).to(self.device)

            self.optimizer.zero_grad()

            pred_actions = self.actor_critic(states) 

            if isinstance(self.action_space, gym.spaces.Box):

                expert_actions = torch.clamp(expert_actions.float(), self.action_space.low[0],self.action_space.high[0])

                if self.use_log_loss:
                    mean, log_std = pred_actions  # model returns tuple now
                    log_std = torch.clamp(log_std, min=-20, max=2)
                    std = log_std.exp()
                    var = std ** 2
                    log_probs = 0.5 * (((expert_actions - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
                    loss = log_probs.sum(dim=-1).mean()
                else:
                    pred_actions = torch.clamp(pred_actions, self.action_space.low[0], self.action_space.high[0])
                    # expert_actions = torch.clamp(expert_actions, self.action_space.low[0], self.action_space.high[0])
                    loss = F.mse_loss(pred_actions, expert_actions)


                # loss = F.mse_loss(pred_actions, expert_actions)
            elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
                raise NotImplementedError
                loss = F.cross_entropy(pred_actions, expert_actions.flatten().long())
            elif self.action_space.__class__.__name__ == "MultiBinary":
                raise NotImplementedError
                loss = torch.binary_cross_entropy_with_logits(pred_actions, expert_actions).mean()

            if update:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if batch_idx >= self.num_batches:
                break

        return (total_loss / batch_idx)

    def reset(self):
        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)


