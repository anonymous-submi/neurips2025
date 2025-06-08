import gym, os
import numpy as np
import random
import pickle

import sys
import torch
# from gym import wrappers

from code_continuous_control.a2c_ppo_acktr.envs import make_vec_envs
from code_continuous_control.a2c_ppo_acktr.model import Policy
from code_continuous_control.a2c_ppo_acktr.arguments import get_args
import os
from evaluation import evaluate


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = get_args()
#set env threads
args.num_processes = 25

args.recurrent_policy = False
args.load_expert = True

os.system(f'mkdir -p {args.demo_data_dir}')
os.system(f'mkdir -p {args.demo_data_dir}/tmp/gym')
sys.path.insert(1,os.path.join(args.rl_baseline_zoo_dir, 'utils'))
from a2c_ppo_acktr.utils import get_saved_hyperparams

device = torch.device(args.device if args.cuda else "cpu")
print(f'device: {device}')
seed = args.seed
print(f'seed: {seed}')

args.env_name = 'Walker2DBulletEnv-v0'  ###  'HopperBulletEnv-v0', 'AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0'
args.rl_baseline_zoo_dir = '/workspace/rl-baselines-zoo/'
args.expert_algo = 'ppo2' #'ppo2'#'trpo' # trpo for ant otherwise ppo2

results_dict = {'concatenated_result': None, 'concatenated_rollout_std': None}
save_name_id = 'expert'
# Get path
current_path = os.getcwd()
save_path = os.path.join(current_path, '', args.env_name)   #set save dir
os.makedirs(save_path, exist_ok=True)
save_name = save_name_id + '.pkl'  

full_save_path = os.path.join(save_path, save_name)

if args.env_name in ['highway-v0']:
    raise NotImplementedError
elif args.env_name in ['duckietown']:
    raise NotImplementedError
else:
    print('[Setting environemnt hyperparams variables]')
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

    

    env = make_vec_envs(args.env_name, seed,1, 0.99, f'{args.demo_data_dir}/tmp/gym', device,\
                       True, stats_path=stats_path, hyperparams=hyperparams, time=time)
    print('loading policy')
    th_model = Policy(
           env.observation_space.shape,
           env.action_space,
           load_expert=True,
           env_name=args.env_name,
           rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
           expert_algo=args.expert_algo,
           # [Bug]: normalize=False,
           normalize=True if hasattr(gym.envs, 'atari') else False,
           base_kwargs={'recurrent': args.recurrent_policy}).to(device)



for i in range(10):
    print('ongoing:',i+1)
    expert_reward,expert_std = evaluate(th_model, None, args.env_name, random.randint(1, 10000),
                                args.num_processes, None, device, num_episodes=args.num_processes,
                                stats_path=stats_path, hyperparams=hyperparams, time=time)
    if i == 0:
        results_dict['concatenated_result'] = np.expand_dims(expert_reward, axis=0)
        results_dict['concatenated_rollout_std'] = np.expand_dims(expert_std, axis=0)
    else:
        results_dict['concatenated_result'] = np.concatenate([results_dict['concatenated_result'], np.expand_dims(expert_reward, axis=0)], axis=0)
        results_dict['concatenated_rollout_std'] = np.concatenate([results_dict['concatenated_rollout_std'], np.expand_dims(expert_std, axis=0)], axis=0)

with open(full_save_path, 'wb') as f:
                pickle.dump(results_dict, f)

for key, value in results_dict.items():
            print(f"Shape of {key}: {value.shape}")
            print(f"Values of {key}: {value}")
