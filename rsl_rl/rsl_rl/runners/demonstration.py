import os
import os.path as osp
import json
import pickle

import numpy as np
import torch

from rsl_rl.utils.utils import get_obs_slice
import rsl_rl.utils.data_compresser as compresser
from rsl_rl.storage.rollout_storage import RolloutStorage

class DemonstrationSaver:
    def __init__(self,
            env,
            policy, # any object with "act(obs, critic_obs)" method to get actions and "get_hidden_states()" method to get hidden states
            save_dir,
            rollout_storage_length= 64,
            min_timesteps= 1e6,
            min_episodes= 10000,
            success_traj_only = False, # if true, the trajectory terminated no by timeout will be dumped.
            use_critic_obs= False,
            obs_disassemble_mapping= None,
        ):
        """
        Args:
            obs_disassemble_mapping (dict):
                If set, the obs segment will be compressed using given type.
                example: {"forward_depth": "normalized_image", "forward_rgb": "normalized_image"}
        """
        self.env = env
        self.policy = policy

        self.save_dir = save_dir
        self.rollout_storage_length = rollout_storage_length
        self.min_timesteps = min_timesteps
        self.min_episodes = min_episodes
        self.use_critic_obs = use_critic_obs
        self.success_traj_only = success_traj_only
        self.obs_disassemble_mapping = obs_disassemble_mapping
        self.RolloutStorageCls = RolloutStorage

    def init_traj_handlers(self):
        # check if data exists, continue
        if len(os.listdir(self.save_dir)) > 1:
            print("Continuing from previous data. You have to make sure the environment configuration is the same.")
            prev_traj = [x for x in os.listdir(self.save_dir) if x.startswith("trajectory_")]
            prev_traj.sort(key= lambda x: int(x.split("_")[1]))
            # fill up the traj_idxs
            self.traj_idxs = []
            for f in prev_traj:
                if len(os.listdir(osp.join(self.save_dir, f))) == 0:
                    self.traj_idxs.append(int(f.split("_")[1]))
            if len(self.traj_idxs) < self.env.num_envs:
                max_traj_idx = max(self.traj_idxs) if len(self.traj_idxs) > 0 else int(prev_traj[-1].split("_")[1])
                for _ in range(self.env.num_envs - len(self.traj_idxs)):
                    self.traj_idxs.append(max_traj_idx + 1)
                    max_traj_idx += 1
            self.traj_idxs = np.array(self.traj_idxs[:self.env.num_envs])
            # load the dataset statistics
            with open(osp.join(self.save_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            self.total_traj_completed = metadata["total_trajectories"]
            self.total_timesteps = metadata["total_timesteps"]
        else:
            self.traj_idxs = np.arange(self.env.num_envs)
            self.total_traj_completed = 0
            self.total_timesteps = 0
        self.metadata["total_timesteps"] = self.total_timesteps
        self.metadata["total_trajectories"] = self.total_traj_completed
        for traj_idx in self.traj_idxs:
            os.makedirs(osp.join(self.save_dir, f"trajectory_{traj_idx}"), exist_ok= True)
        self.dumped_traj_lengths = np.zeros(self.env.num_envs, dtype= np.int32)

        # initialize compressing parameters if needed
        if not self.obs_disassemble_mapping is None:
            self.metadata["obs_segments"] = self.env.obs_segments
            self.metadata["obs_disassemble_mapping"] = self.obs_disassemble_mapping

    def init_storage_buffer(self):
        self.rollout_storage = self.RolloutStorageCls(
            self.env.num_envs,
            self.rollout_storage_length,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            self.env.device,
        )
        self.transition = self.RolloutStorageCls.Transition()
        self.transition_has_timeouts = False
        self.transition_timeouts = torch.zeros(self.rollout_storage_length, self.env.num_envs, dtype= torch.bool, device= self.env.device)

    def check_stop(self):
        return (self.total_traj_completed >= self.min_episodes) \
            and (self.total_timesteps >= self.min_timesteps)

    @torch.no_grad()
    def collect_step(self, step_i):
        """ Collect one step of demonstration data """
        actions, rewards, dones, infos, n_obs, n_critic_obs = self.get_transition()

        self.build_transition(step_i, actions, rewards, dones, infos)
        self.add_transition(step_i, infos)
        self.transition.clear()

        self.policy_reset(dones)
        self.obs, self.critic_obs = n_obs, n_critic_obs

    def get_transition(self):
        if self.use_critic_obs:
            actions = self.policy.act_inference(self.critic_obs)
        else:
            actions = self.policy.act_inference(self.obs)
        n_obs, n_critic_obs, rewards, dones, infos = self.env.step(actions)
        return actions, rewards, dones, infos, n_obs, n_critic_obs

    def build_transition(self, step_i, actions, rewards, dones, infos):
        """ Fill the transition to meet the interface of rollout storage """
        self.transition.observations = self.obs
        if not self.critic_obs is None: self.transition.critic_observations = self.critic_obs
        # if self.policy.is_recurrent:
        #     self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = actions
        self.transition.rewards = rewards
        self.transition.dones = dones

        # fill up some of the attributes to meet the interface of rollout storage, but not collected to files
        self.transition.values = torch.zeros_like(rewards).unsqueeze(-1)
        self.transition.actions_log_prob = torch.zeros_like(rewards)
        self.transition.action_mean = torch.zeros_like(actions)
        self.transition.action_sigma = torch.zeros_like(actions)

    def add_transition(self, step_i, infos):
        self.rollout_storage.add_transitions(self.transition)
        if "time_outs" in infos:
            self.transition_has_timeouts = True
            self.transition_timeouts[step_i] = infos["time_outs"]

    def policy_reset(self, dones):
        if dones.any():
            self.policy.reset(dones)

    def dump_to_file(self, env_i, step_slice):
        """ dump the part of trajectory to the trajectory directory """
        traj_idx = self.traj_idxs[env_i]
        traj_dir = osp.join(self.save_dir, f"trajectory_{traj_idx}")
        traj_file = osp.join(
            traj_dir,
            f"traj_{self.dumped_traj_lengths[env_i]:06d}_{self.dumped_traj_lengths[env_i]+step_slice.stop-step_slice.start:06d}.pickle",
        )
        trajectory = self.wrap_up_trajectory(env_i, step_slice)
        with open(traj_file, 'wb') as f:
            pickle.dump(trajectory, f)
        self.dumped_traj_lengths[env_i] += step_slice.stop - step_slice.start
        self.total_timesteps += step_slice.stop - step_slice.start
        with open(osp.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent= 4)

    def wrap_up_trajectory(self, env_i, step_slice):
        trajectory = dict(
            privileged_observations= self.rollout_storage.privileged_observations[step_slice, env_i].cpu().numpy(),
            actions= self.rollout_storage.actions[step_slice, env_i].cpu().numpy(),
            rewards= self.rollout_storage.rewards[step_slice, env_i].cpu().numpy(),
            dones= self.rollout_storage.dones[step_slice, env_i].cpu().numpy(),
            values= self.rollout_storage.values[step_slice, env_i].cpu().numpy(),
        )
        # compress observations components if set
        if not self.obs_disassemble_mapping is None:
            observations = self.rollout_storage.observations[step_slice, env_i].cpu().numpy() # (n_steps, d_obs)
            for component_name in self.metadata["obs_segments"].keys():
                obs_slice = get_obs_slice(self.metadata["obs_segments"], component_name)
                obs_component = observations[..., obs_slice[0]]
                if component_name in self.obs_disassemble_mapping:
                    # compress the component
                    obs_component = getattr(
                        compresser,
                        "compress_" + self.obs_disassemble_mapping[component_name],
                    )(obs_component)
                trajectory["obs_" + component_name] = obs_component
        else:
            trajectory["observations"] = self.rollout_storage.observations[step_slice, env_i].cpu().numpy(),
        if self.transition_has_timeouts:
            trajectory["timeouts"] = self.transition_timeouts[step_slice, env_i].cpu().numpy()
        return trajectory

    def update_traj_handler(self, env_i, step_slice):
        """ update the trajectory file handler for the env_i """
        # save the metadatas for current trajectory
        traj_idx = self.traj_idxs[env_i]

        if self.success_traj_only:
            if self.rollout_storage.dones[step_slice.stop-1, env_i] and (not self.transition_timeouts[step_slice.stop-1, env_i]):
                # done by termination not timeout (failed)
                # remove all files in current trajectory directory
                traj_dir = osp.join(self.save_dir, f"trajectory_{traj_idx}")
                for f in os.listdir(traj_dir):
                    try:
                        if f.startswith("traj_"):
                            start_timestep, stop_timestep = f.split("_")[1:]
                            start_timestep = int(start_timestep)
                            stop_timestep = int(stop_timestep)
                            self.total_timesteps -= stop_timestep - start_timestep
                    except:
                        pass
                    os.remove(osp.join(traj_dir, f))
                self.dumped_traj_lengths[env_i] = 0
                return

        # update the handlers to a new trajectory
        # Also, skip the trajectory directory that has data collected before this run.
        while len(os.listdir(osp.join(self.save_dir, f"trajectory_{traj_idx}"))) > 0:
            traj_idx = max(self.traj_idxs) + 1
            os.makedirs(osp.join(self.save_dir, f"trajectory_{traj_idx}"), exist_ok= True)
        self.traj_idxs[env_i] = traj_idx
        self.total_traj_completed += 1
        self.dumped_traj_lengths[env_i] = 0

    def save_steps(self):
        """ dump a series or transitions to the file """
        for rollout_env_i in range(self.rollout_storage.num_envs):
            done_idxs = torch.where(self.rollout_storage.dones[:, rollout_env_i, 0])[0]
            if len(done_idxs) == 0:
                # dump the whole rollout for this env
                self.dump_to_file(rollout_env_i, slice(0, self.rollout_storage.num_transitions_per_env))
            else:
                start_idx = 0
                di = 0
                while di < done_idxs.shape[0]:
                    end_idx = done_idxs[di].item()

                    # dump and update the traj_idx for this env
                    self.dump_to_file(rollout_env_i, slice(start_idx, end_idx+1))
                    self.update_traj_handler(rollout_env_i, slice(start_idx, end_idx+1))

                    start_idx = end_idx + 1
                    di += 1

    def collect_and_save(self, config= None):
        """ Run the rolllout to collect the demonstration data and save it to the file """
        # create directory and save metadata file
        self.metadata = {
            'config': config,
            'env': self.env.__class__.__name__,
            'policy': self.policy.__class__.__name__,
            'rollout_storage_length': self.rollout_storage_length,
            'success_traj_only': self.success_traj_only,
            'min_timesteps': self.min_timesteps,
            'min_episodes': self.min_episodes,
            'use_critic_obs': self.use_critic_obs,
            
        }
        # create env-wise trajectory file handler
        os.makedirs(self.save_dir, exist_ok= True)
        self.init_traj_handlers()
        self.init_storage_buffer()

        with open(osp.join(self.save_dir, 'metadata.json'), 'w') as f:
            # It will be refreshed once the collection is done.
            json.dump(self.metadata, f, indent= 4)
        
        # collect the demonstration data
        self.env.reset()
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        self.obs, self.critic_obs = obs, critic_obs
        while not self.check_stop():
            for step_i in range(self.rollout_storage_length):
                self.collect_step(step_i)
            self.save_steps()
            self.rollout_storage.clear()
            self.print_log()

        # close the trajectory file handlers
        self.close()

    def print_log(self):
        """ print the log """
        print("total_timesteps:", self.total_timesteps)
        print("total_trajectories", self.total_traj_completed)

    def close(self):
        """ check empty directories and remove them """
        pass

    def __del__(self):
        """ Incase the process stops accedentally, close the file handlers """
        for traj_idx in self.traj_idxs:
            traj_dir = osp.join(self.save_dir, f"trajectory_{traj_idx}")
            # remove the empty directories
            if len(os.listdir(traj_dir)) == 0:
                os.rmdir(traj_dir)
        for timestep_count in self.dumped_traj_lengths:
            self.total_timesteps += timestep_count
        self.metadata["total_timesteps"] = self.total_timesteps.item() if isinstance(self.total_timesteps, np.int64) else self.total_timesteps
        self.metadata["total_trajectories"] = self.total_traj_completed
        with open(osp.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent= 4)
        print(f"Saved dataset in {self.save_dir}")
