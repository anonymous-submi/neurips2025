import gym, os
import numpy as np
import argparse
import random
import pandas as pd
import copy
import pybullet_envs

import sys
import torch
from gym import wrappers
import random
import torch.nn.functional as F
import torch.nn as nn
import torch as th

from code_continuous_control.a2c_ppo_acktr.envs import make_vec_envs
from code_continuous_control.a2c_ppo_acktr.model import Policy
from code_continuous_control.a2c_ppo_acktr.arguments import get_args
import code_continuous_control.a2c_ppo_acktr.ensemble_models as ensemble_models
from eval_ensemble import eval_ensemble_class
import os

from mftpl import mftpl
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env_to_hiddensize_dict = {
    'HopperBulletEnv-v0': 4, 
    'AntBulletEnv-v0': 8,
    'HalfCheetahBulletEnv-v0': 12,  
    'Walker2DBulletEnv-v0': 24   
    }

def custom_format(num):
    if num >= 1:
        return f"{num:.6f}"
    else:
        return f"{num:.3e}"

def soil_function(algorithm,ensemble_size,env_name,device,seed =1,offline_hybrid_rate= None,random_selection=True,non_realizable=False,noisy_expert=False
                  ,rl_baseline_zoo_dir='/workspace/rl-baselines-zoo'):
    args = get_args()

    args.behavior_cloning = True if algorithm=='bc' else False
    args.ensemble_shuffle_type = 'norm_shuffle'

    args.ensemble_size = ensemble_size
    args.device = device
    if non_realizable:
        args.rounds = 40 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 50
    else:
        args.rounds = 200 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 1000
    args.random_selection =random_selection
    args.non_realizable = non_realizable
    args.noisy_expert = noisy_expert
    args.seed = seed


    # new small state sample setting
    args.rounds = 1000 if env_name in ['AntBulletEnv-v0','HopperBulletEnv-v0'] else 5000
    args.num_processes = 1
    args.num_eval_processes = 25
    args.subsample_frequency = 999 #set to sample 1 state per traj
    args.data_per_round = 1 
    args.eval_interval = 50 # number of samples per eval
    args.use_log_loss = True

    # new warmstart data rounds selection
    if offline_hybrid_rate and offline_hybrid_rate>0:
        args.offline_rounds = int(args.rounds/offline_hybrid_rate)
    

    print('Env Name: ',env_name, 'rounds: ', args.rounds) 
    print('Use Noisy Expert: ', noisy_expert)
    print('Rounds of warmstart expert data',args.offline_rounds)


    if args.non_realizable:
        args.hidden_size = env_to_hiddensize_dict[env_name]

    if args.behavior_cloning:
        print("bc",args.ensemble_size)
        save_name_id = "bc_"+str(args.ensemble_size)
    elif args.ensemble_shuffle_type=='norm_shuffle':
        print("dagger",args.ensemble_size)
        save_name_id = "dagger_"+str(args.ensemble_size)

    args.env_name = env_name
    args.rl_baseline_zoo_dir = rl_baseline_zoo_dir

    args.recurrent_policy = False
    args.load_expert = True

    os.system(f'mkdir -p {args.demo_data_dir}')
    os.system(f'mkdir -p {args.demo_data_dir}/tmp/gym')
    sys.path.insert(1,os.path.join(args.rl_baseline_zoo_dir, 'utils'))
    from a2c_ppo_acktr.utils import get_saved_hyperparams

    device = torch.device(args.device if args.cuda else "cpu")
    print(f'device: {device}')
    # seed = args.seed
    print(f'seed: {args.seed}')

    if args.env_name in ['highway-v0']:
        raise NotImplementedError

    elif args.env_name in ['duckietown']:
        raise NotImplementedError
    
    else:
        print('[Setting environemnt hyperparams variables]')

        if args.env_name in ['AntBulletEnv-v0']:
            args.expert_algo = 'trpo'
        else:
            args.expert_algo = 'ppo2'
            
        stats_path = os.path.join(args.rl_baseline_zoo_dir, 'trained_agents', f'{args.expert_algo}',\
                            f'{args.env_name}')
        hyperparams, stats_path = get_saved_hyperparams(stats_path, test_mode=True,\
                                            norm_reward=args.norm_reward_stable_baseline)

        time_wrapper_envs = ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0', 'AntBulletEnv-v0']
        if args.env_name in time_wrapper_envs:
            time=True
            print('use time as feature')
        else:
            time = False

        env = make_vec_envs(args.env_name, args.seed, 1, 0.99, f'{args.demo_data_dir}/tmp/gym', device,\
                        True, stats_path=stats_path, hyperparams=hyperparams, time=time)#time
        
        th_model = Policy(
            env.observation_space.shape,
            env.action_space,
            load_expert=True,
            env_name=args.env_name,
            rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
            expert_algo=args.expert_algo,
            normalize=True if hasattr(gym.envs, 'atari') else False,
            base_kwargs={'recurrent': args.recurrent_policy}).to(device)
        th_model.dist = th_model.dist.to(device)
        expert_param = copy.deepcopy(th_model.state_dict())
        
    rtn_obs, rtn_acs, rtn_lens, ep_rewards = [], [], [], []
    obs = env.reset()
    if args.env_name in ['duckietown']:
        obs = torch.FloatTensor([obs])

    save = True
    idx = random.randint(1,args.subsample_frequency)

    saved_param = None
        
    # #define ensemble policy    
    ensemble_size = args.ensemble_size
    try:
        num_actions = env.action_space.n
    except:
        num_actions = env.action_space.shape[0]

    print('hidden size',args.hidden_size)
    ensemble_args = (env.observation_space.shape[0], num_actions, args.hidden_size, ensemble_size)
    
    if len(env.observation_space.shape) == 3:
        if args.env_name in ['duckietown']:
            policy_def = ensemble_models.PolicyEnsembleDuckieTownCNN
        else:
            policy_def = ensemble_models.PolicyEnsembleCNN
    else:
        if args.non_realizable:
            policy_def = ensemble_models.PolicyEnsembleMLP_nonrealizable
        else:
            if args.use_log_loss:
                policy_def = ensemble_models.PolicyEnsembleMLP_Gaussian
            else:
                policy_def = ensemble_models.PolicyEnsembleMLP_simple
        
    ensemble_policy = policy_def(*ensemble_args).to(device)

    saved_param = copy.deepcopy(ensemble_policy.state_dict())

    random_selection_list = np.random.randint(low=0, high=ensemble_size, size=1000)

    # set evaluation method
    eval = eval_ensemble_class(ensemble_size, None, args.env_name, args.seed,
                            args.num_eval_processes, None, device, num_episodes=args.num_eval_processes,
                            stats_path=stats_path, hyperparams=hyperparams, time=time, use_log_loss=args.use_log_loss)
    policy_list = []
    result_list = []
    std_list = []
    loss_list = []
    loss_std_list = []

    random_result_list = []
    random_std_list = []
    random_loss_list = []
    random_loss_std_list = []

    # env steps
    step = 0
    learning_round = 0

    while True:
        
        with torch.no_grad():
            if args.env_name in ['highway-v0']:
                action = torch.tensor([[th_model.act(obs)]])
            elif args.env_name in ['duckietown']:
                action = torch.FloatTensor([expert.predict(None)])
            elif hasattr(gym.envs, 'atari'):
                _, actor_features, _ = th_model.base(obs, None, None)
                dist = th_model.dist(actor_features)
                action = dist.sample()
            else:
                if not args.noisy_expert:
                    _, action, _, _ = th_model.act(obs, None, None, deterministic=True)
                else:
                    _, action, _, _ = th_model.act(obs, None, None, deterministic=False)

                ensemble_obs = torch.unsqueeze(obs, dim=0)
                ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
   
            if args.use_log_loss:
                mean, log_std = ensemble_policy(ensemble_obs)
                std = torch.exp(log_std)

                mean = mean.view(ensemble_size, obs.shape[0], -1)
                std = std.view(ensemble_size, obs.shape[0], -1)

                if args.random_selection:
                    ensemble_idx = random_selection_list[step]
                    dist = torch.distributions.Normal(mean[ensemble_idx], std[ensemble_idx])
                    selected_action = dist.rsample()
                else:
                    dist = torch.distributions.Normal(mean[0], std[0])
                    selected_action = dist.rsample()

            else:
                ensemble_actions = ensemble_policy(ensemble_obs).view(ensemble_size, obs.shape[0], -1)
                clip_ensemble_actions = np.clip(ensemble_actions.cpu(), env.action_space.low, env.action_space.high)
                if args.random_selection:
                    selected_action = clip_ensemble_actions[random_selection_list[step], 0]
                else:
                    selected_action = torch.mean(clip_ensemble_actions, dim=0)[0]

        if isinstance(env.action_space, gym.spaces.Box):
            clip_action = np.clip(action.cpu(), env.action_space.low, env.action_space.high)
        else:
            clip_action = action.cpu()

        activate = False
        
        if (step == idx and args.subsample) or not args.subsample:
            rtn_obs.append(obs.cpu().numpy().copy())
            rtn_acs.append(action.cpu().numpy().copy())
            idx += args.subsample_frequency
            activate = True
            
        if args.behavior_cloning or learning_round < args.offline_rounds:
            if args.env_name in ['duckietown']:
                obs, reward, done, infos = env.step(clip_action.squeeze())
                obs = torch.FloatTensor([obs])
            else:
                obs, reward, done, infos = env.step(clip_action)
        else:
            if args.env_name in ['duckietown']:
                obs, reward, done, infos = env.step(selected_action.squeeze())
                obs = torch.FloatTensor([obs])
            else:
                obs, reward, done, infos = env.step(selected_action)

        step += 1

        if args.env_name in ['duckietown']:
            if done:
                ep_rewards.append(reward)
                save = True
                obs = env.reset()
                obs = torch.FloatTensor([obs])
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                
        else:
            for info in infos or done:
                if 'episode' in info.keys():
                    ep_rewards.append(info['episode']['r'])
                    save = True
                    obs = env.reset()
                    step = 0
                    idx=random.randint(1,args.subsample_frequency)
                    random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)    
        
        if activate and len(rtn_obs) % args.data_per_round == 0:
            if int(len(rtn_obs)/args.data_per_round) in range(args.rounds+1):
                obs = env.reset()
                step = 0
                idx=random.randint(1,args.subsample_frequency)
                random_selection_list = np.random.randint(low=0, high=args.ensemble_size, size=1000)
                rtn_obs_ = np.concatenate(rtn_obs)
                rtn_acs_ = np.concatenate(rtn_acs)

                if (len(rtn_obs) % args.eval_interval == 0) or (not args.behavior_cloning and learning_round >= args.offline_rounds):
                    ensemble_param = mftpl(args,env,deepcopy(rtn_obs_),deepcopy(rtn_acs_))
                    saved_param = deepcopy(ensemble_param)
                    ensemble_policy.load_state_dict(saved_param)

                if len(rtn_obs)%args.eval_interval == 0:
                    result,std,loss, loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=False)
                    
                    if args.ensemble_size > 1:
                        random_result,random_std,random_loss,random_loss_std = eval.test(ensemble_policy,expert = th_model,random_selection=True)
                        print(
                            save_name_id, 
                            f'{len(rtn_obs)} samples avg: {custom_format(result)}, std: {custom_format(std)},'
                            f'loss: {custom_format(loss)},loss_std: {custom_format(loss_std)}, '
                            f'random: {custom_format(random_result)}, std: {custom_format(random_std)}, '
                            f'loss:{custom_format(random_loss)}, std: {custom_format(random_loss_std)}'
                        )
                        
                        random_result_list.append(random_result)
                        random_std_list.append(random_std)
                        random_loss_list.append(random_loss)
                        random_loss_std_list.append(random_loss_std)

                    else:
                        print(save_name_id, f'{len(rtn_obs)} samples: {result}, std: {std}, loss: {loss}, std: {loss_std}')
                    
                    # save trained policies and statistics
                    cpu_state_dict = {k: v.cpu() for k, v in saved_param.items()}
                    policy_list.append(cpu_state_dict)
                    
                    result_list.append(result)
                    std_list.append(std)
                    loss_list.append(loss)
                    loss_std_list.append(loss_std)


                
                learning_round =  int(len(rtn_obs)/args.data_per_round)
                if learning_round % args.rounds == 0:
                    eval.close()
                    id_list = list(range(args.rounds))

                    result_array_datasize_list = np.array(id_list) * args.data_per_round + args.data_per_round 

                    result_array = np.array(result_list)
                    rollout_std_array = np.array(std_list)
                    loss_array = np.array(loss_list)
                    loss_std_array = np.array(loss_std_list)
                    final_mean_result = [result_array,rollout_std_array,loss_array,loss_std_array]

                    random_result_array = np.array(random_result_list)
                    random_rollout_std_array = np.array(random_std_list)
                    random_loss_array = np.array(random_loss_list)
                    random_loss_std_array = np.array(random_loss_std_list)
                    final_random_result = [random_result_array,random_rollout_std_array,random_loss_array,random_loss_std_array]


                    return result_array_datasize_list, policy_list,final_mean_result,final_random_result,rtn_obs_,rtn_acs_


if __name__ == "__main__":
    datasize_list, policies, result, random_result, obs, acs = soil_function(
        algorithm='bc',ensemble_size=1,offline_hybrid_rate=None,env_name=  'Walker2DBulletEnv-v0' ,device = 'cuda:0', random_selection=False, noisy_expert=False, non_realizable = False) 