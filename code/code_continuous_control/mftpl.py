#!/usr/bin/env python

import copy
import glob
import os
import time
from collections import deque
import sys
import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pybullet_envs

import random

from code_continuous_control.a2c_ppo_acktr import algo, utils
from code_continuous_control.a2c_ppo_acktr.algo import gail
from code_continuous_control.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from code_continuous_control.a2c_ppo_acktr.storage import RolloutStorage
import code_continuous_control.a2c_ppo_acktr.ensemble_models as ensemble_models

import pandas as pd
from instant_dataset import InstantDataset



def mftpl(args,envs,obs,acs,additional_obs=None,additional_acs=None, additional_sample_budget = None):
    
    if args.system == 'philly':
        args.demo_data_dir = os.getenv('PT_OUTPUT_DIR') + '/demo_data/'
        args.save_model_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_models/'
        args.save_results_dir = os.getenv('PT_OUTPUT_DIR') + '/trained_results/'


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device(args.device if args.cuda else "cpu")

    ensemble_size = args.ensemble_size
    try:
        num_actions = envs.action_space.n
    except:
        num_actions = envs.action_space.shape[0]
    ensemble_args = (envs.observation_space.shape[0], num_actions, args.hidden_size, ensemble_size)
    
    if len(envs.observation_space.shape) == 3:
        if args.env_name in ['duckietown']:
            policy_def = ensemble_models.PolicyEnsembleDuckieTownCNN
        else:
            policy_def = ensemble_models.PolicyEnsembleCNN
    else:
        if args.non_realizable:
            if args.linear:
                policy_def = ensemble_models.PolicyEnsembleMLP_linear
            else:
                policy_def = ensemble_models.PolicyEnsembleMLP_nonrealizable
        else:
            if args.use_log_loss:
                policy_def = ensemble_models.PolicyEnsembleMLP_Gaussian
            else:
                policy_def = ensemble_models.PolicyEnsembleMLP_simple
    ensemble_policy = policy_def(*ensemble_args).to(device)

    model_reward = None

    if not args.linear: 
        annotated_dataset = InstantDataset(obs, acs, args.env_name,\
                    args.num_trajs, args.seed, args.ensemble_shuffle_type, additional_obs, additional_acs, additional_sample_budget)

        ensemble = BehaviorCloning(ensemble_policy,device, batch_size=args.batch_size,\
                   lr=args.lr, envs=envs, training_data_split=args.training_data_split,\
                   annotated_dataset=annotated_dataset,ensemble_size=ensemble_size,use_log_loss=args.use_log_loss)
        
        best_val_loss = float('inf')
        params = None

        for bc_epoch in range(args.train_epoch):
            train_loss = ensemble.update(update=True, data_loader_type='train')


            if (bc_epoch + 1) % 250 == 0 and ensemble.tedata is not None:
                val_loss = ensemble.update(update=False, data_loader_type='test')
                # Save model if validation improves
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    params = copy.deepcopy(ensemble_policy.state_dict())
        if not params:
            params = copy.deepcopy(ensemble_policy.state_dict())
             

    return params
