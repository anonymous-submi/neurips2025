import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import random

class InstantDataset:
    def __init__(self, obs, acs, env_name, num_trajs, seed, ensemble_shuffle_type,
                 additional_obs=None, additional_acs=None, additional_sample_budget=None):
        self.obs = obs
        self.acs = acs
        self.env_name = env_name
        self.num_trajs = num_trajs
        self.seed = seed
        self.ensemble_shuffle_type = ensemble_shuffle_type

    def load_demo_data(self, training_data_split, batch_size, ensemble_size):
        obs = torch.from_numpy(self.obs)
        acs = torch.from_numpy(self.acs)

        perm = torch.randperm(obs.size(0))
        obs = obs[perm]
        acs = acs[perm]

        n_total = obs.size(0)
        # Reserve 1-training_data_split of the data for testing (validation)
        n_train = math.ceil(n_total * training_data_split)
        obs_train = obs[:n_train]
        acs_train = acs[:n_train]
        obs_test  = obs[n_train:]
        acs_test  = acs[n_train:]

        trdata = []
        for i in range(ensemble_size):
            if self.ensemble_shuffle_type == 'sample_w_replace':
                print('***** sample_w_replace *****')
                obs_train_i = torch.stack([obs_train[random.randint(0, n_train - 1)] for _ in range(n_train)])
                acs_train_i = torch.stack([acs_train[random.randint(0, n_train - 1)] for _ in range(n_train)])
                shuffle = False
            else:
                perm_i = torch.randperm(n_train)
                obs_train_i = obs_train[perm_i]
                acs_train_i = acs_train[perm_i]
                shuffle = self.ensemble_shuffle_type == 'norm_shuffle'

            tr_batch_size = min(batch_size, len(obs_train_i))
            tr_drop_last = (tr_batch_size != len(obs_train_i))
            if not tr_drop_last:
                tr_batch_size = int(np.floor(tr_batch_size / ensemble_size) * ensemble_size)
                obs_train_i = obs_train_i[:tr_batch_size]
                acs_train_i = acs_train_i[:tr_batch_size]

            loader = DataLoader(TensorDataset(obs_train_i, acs_train_i),
                                batch_size=tr_batch_size,
                                shuffle=shuffle,
                                drop_last=tr_drop_last)
            trdata.append(loader)

        tedata = []
        if len(obs_test) > 0:
            for i in range(ensemble_size):
                te_batch_size = len(obs_test)  # full test set in one batch
                obs_test_i = obs_test
                acs_test_i = acs_test

                loader = DataLoader(TensorDataset(obs_test_i, acs_test_i),
                                    batch_size=te_batch_size,
                                    shuffle=False,  # disable shuffling 
                                    drop_last=False)
                tedata.append(loader)
        else:
            tedata = None

        return {'trdata': trdata, 'tedata': tedata}
        