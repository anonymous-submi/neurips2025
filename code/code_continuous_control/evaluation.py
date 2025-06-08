import numpy as np
import torch
import gym

from code_continuous_control.a2c_ppo_acktr import utils
from code_continuous_control.a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, stats_path=None, hyperparams=None, time=False):

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, 0.99,
                              eval_log_dir, device, True, stats_path=stats_path,
                              hyperparams=hyperparams, time=time)
    
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    current_rewards = np.zeros(num_processes)

    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, actions, _, _ = actor_critic.act(obs, None, None, deterministic=True)

        if isinstance(eval_envs.action_space, gym.spaces.Box):
            actions = torch.clamp(actions, float(eval_envs.action_space.low[0]),
                                  float(eval_envs.action_space.high[0]))

        obs, rewards, dones, _ = eval_envs.step(actions)

        current_rewards += rewards.cpu().numpy().flatten()

        for i, done_ in enumerate(dones):
            if done_:
                eval_episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0.0

    eval_envs.close()

    mean_reward = np.mean(eval_episode_rewards)
    std_reward = np.std(eval_episode_rewards)
    print(f"Evaluation over {len(eval_episode_rewards)} episodes: mean = {mean_reward:.2f}, std = {std_reward:.2f}")
    
    return mean_reward, std_reward
