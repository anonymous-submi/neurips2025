import numpy as np
import torch
import gym

from code_continuous_control.a2c_ppo_acktr import utils
from code_continuous_control.a2c_ppo_acktr.envs import make_vec_envs
import torch.nn.functional as F

class eval_ensemble_class:
    def __init__(self, ensemble_size, ob_rms, env_name, seed, num_processes, eval_log_dir,
        device, num_episodes=None, stats_path=None, hyperparams=None, time=False, use_log_loss=False):
        super(eval_ensemble_class, self).__init__()

        self.num_processes = num_processes
        self.num_episodes = num_episodes
        self.device = device
        self.ensemble_size = ensemble_size
        self.use_log_loss = use_log_loss
        self.eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, 0.99, eval_log_dir, device,
                    True, stats_path=stats_path, hyperparams=hyperparams, time=time)
        self.eval_envs.reset()

    def test(self, ensemble_policy, expert, random_selection):
        step = 0
        eval_episode_rewards = []
        eval_episode_loss = []
        obs = self.eval_envs.reset()
        current_episode_rewards = np.zeros(self.num_processes)
        current_episode_expert_loss = np.zeros(self.num_processes)

        while len(eval_episode_rewards) < self.num_episodes:
            with torch.no_grad():
                ensemble_obs = obs.unsqueeze(0).repeat(self.ensemble_size, 1, 1).view(-1, obs.shape[1])

                if self.use_log_loss:
                    mean, log_std = ensemble_policy(ensemble_obs)
                    std = torch.exp(log_std)

                    mean = mean.view(self.ensemble_size, obs.shape[0], -1)
                    std = std.view(self.ensemble_size, obs.shape[0], -1)
                    dist_all = torch.distributions.Normal(mean, std)
                    
                    if random_selection:
                        indices = torch.randint(low=0, high=self.ensemble_size, size=(obs.shape[0],))
                        selected_mean = mean[indices, torch.arange(obs.shape[0])]
                        selected_std = std[indices, torch.arange(obs.shape[0])]
                        dist = torch.distributions.Normal(selected_mean, selected_std)
                        selected_actions = dist.rsample()
                    else:
                        selected_mean = mean[0]
                        selected_std = std[0]
                        dist = torch.distributions.Normal(selected_mean, selected_std)
                        selected_actions = dist.rsample()
                else:
                    ensemble_actions = ensemble_policy(ensemble_obs)
                    if random_selection:
                        for i in range(obs.shape[0]):
                            selected_actions[i] = ensemble_actions[torch.randint(low=0, high=self.ensemble_size, size=())][i]   
                    else:
                        selected_actions = ensemble_actions

                _, expert_action, _, _ = expert.act(obs, None, None, deterministic=True)

            if isinstance(self.eval_envs.action_space, gym.spaces.Box):
                clip_ensemble_actions = torch.clamp(selected_actions,
                                                    float(self.eval_envs.action_space.low[0]),
                                                    float(self.eval_envs.action_space.high[0]))
                clip_expert_action = torch.clamp(expert_action.float(),
                                                 float(self.eval_envs.action_space.low[0]),
                                                 float(self.eval_envs.action_space.high[0]))
                if not self.use_log_loss and not random_selection:
                    clip_ensemble_actions = torch.mean(clip_ensemble_actions, dim=0)
            else:
                clip_ensemble_actions = selected_actions
                clip_expert_action = expert_action

                if not self.use_log_loss and not random_selection:
                    clip_ensemble_actions = torch.mean(clip_ensemble_actions, dim=0)

            obs, reward, done, _ = self.eval_envs.step(clip_ensemble_actions)
            step += 1
            current_episode_rewards += reward.cpu().numpy().flatten()

            if self.use_log_loss:
                loss_vals = -dist.log_prob(clip_expert_action).sum(dim=1)
                
            else:
                loss_vals = ((clip_ensemble_actions - clip_expert_action) ** 2).mean(dim=1)
                
            current_episode_expert_loss += loss_vals.cpu().numpy()

            for i, done_ in enumerate(done):
                if done_:
                    eval_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0
                    eval_episode_loss.append(current_episode_expert_loss[i] / step)
                    current_episode_expert_loss[i] = 0

            if len(eval_episode_rewards) >= self.num_episodes:
                break

        return np.mean(eval_episode_rewards), np.std(eval_episode_rewards), \
               np.mean(eval_episode_loss), np.std(eval_episode_loss)

    def close(self):
        self.eval_envs.close()


def eval_ensemble(ensemble_policy, ensemble_size, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, stats_path=None, hyperparams=None, time=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, 0.99, eval_log_dir, device,\
                    True, stats_path=stats_path, hyperparams=hyperparams, time=time)
    eval_episode_rewards = []
    obs = eval_envs.reset()
    current_episode_rewards = np.zeros(num_processes) 
    while len(eval_episode_rewards) < num_episodes:
        selected_actions = torch.zeros((obs.shape[0],eval_envs.action_space.low.shape[0])).to(device)
        with torch.no_grad():
            ensemble_obs = torch.unsqueeze(obs, dim=0)
            ensemble_obs = torch.cat([ensemble_obs.repeat(ensemble_size, *[1]*len(ensemble_obs.shape[1:]))], dim=0)
            ensemble_actions = ensemble_policy(ensemble_obs)
            for i in range(obs.shape[0]):
                selected_actions[i] = ensemble_actions[torch.randint(low=0, high=ensemble_size, size=())][i]   
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_ensemble_actions = torch.clamp(selected_actions, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0])) 
        else:
            clip_ensemble_actions = selected_actions

        obs, reward, done, infos = eval_envs.step(clip_ensemble_actions)
        
        current_episode_rewards += reward.cpu().numpy().flatten()

        # Check if episodes are done
        for i, done_ in enumerate(done):
            if done_:
                eval_episode_rewards.append(current_episode_rewards[i])
                # print('done',current_episode_rewards[i],num_episodes)
                # Reset the reward accumulator 
                current_episode_rewards[i] = 0  

        if len(eval_episode_rewards) >= num_episodes:
            # print('gather enouth trails',num_episodes)
            break

    eval_envs.close()

    return np.mean(eval_episode_rewards),np.std(eval_episode_rewards)
