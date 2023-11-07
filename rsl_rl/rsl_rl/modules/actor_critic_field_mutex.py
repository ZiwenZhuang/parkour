import os
import os.path as osp
import json
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from rsl_rl.utils.utils import get_obs_slice
from rsl_rl.modules.actor_critic_mutex import ActorCriticMutex

class ActorCriticFieldMutex(ActorCriticMutex):
    def __init__(self,
            *args,
            cmd_vel_mapping = dict(),
            reset_non_selected = "all",
            action_smoothing_buffer_len = 1,
            **kwargs,
        ):
        """ NOTE: This implementation only supports subpolicy output to (-1., 1.) range.
        Args:
            override_cmd_vel (dict): override the velocity command for each sub policy for their
                best performance. The key is the sub policy idx, and the value is the +x velocity 
        """
        super().__init__(*args, **kwargs)
        self.cmd_vel_mapping = cmd_vel_mapping
        self.reset_non_selected = reset_non_selected
        self.action_smoothing_buffer_len = action_smoothing_buffer_len
        self.action_smoothing_buffer = None

        # load cmd_scale to assign the cmd_vel during overriding
        self.cmd_scales = []
        for sub_path in self.sub_policy_paths:
            with open(osp.join(sub_path, "config.json"), "r") as f:
                policy_kwargs = json.load(f, object_pairs_hook= OrderedDict)
                cmd_scale = policy_kwargs["normalization"]["obs_scales"]
            self.cmd_scales.append(cmd_scale)
        self.cmd_vel_current = dict()
        self.resample_cmd_vel_current()

    def resample_cmd_vel_current(self, dones= None):
        """ In case cmd_vel_mapping has tuple for randomness """
        for idx, vel in self.cmd_vel_mapping.items():
            idx = int(idx)
            if isinstance(vel, tuple):
                new_cmd_vel = np.random.uniform(*vel)
            else:
                new_cmd_vel = vel
            if dones is None:
                self.cmd_vel_current[idx] = new_cmd_vel
            else:
                # make the cmd_vel_current batchwise
                self.cmd_vel_current[idx] = torch.ones_like(dones).to(torch.float32) * self.cmd_vel_current[idx]
                self.cmd_vel_current[idx][dones] = new_cmd_vel

    def recover_last_action(self, observations, policy_selection):
        """ Consider the action is scaled when some sub policy have different action scale, it need
        be recovered to its intitial range.
        """
        try:
            obs_slice = get_obs_slice(self.obs_segments, "proprioception")
        except AssertionError:
            return observations
        proprioception_obs = observations[..., obs_slice[0]].reshape(*observations.shape[:-1], *obs_slice[1])
        for idx in range(len(self.submodules)):
            proprioception_obs[policy_selection[..., idx], -12:] *= self.env_action_scale / getattr(self, "subpolicy_action_scale_{:d}".format(idx))
        observations = torch.cat([
            observations[..., :obs_slice[0].start],
            proprioception_obs.reshape(*observations.shape[:-1], -1),
            observations[..., obs_slice[0].stop:],
        ], dim= -1)
        return observations
    
    def get_policy_selection(self, observations):
        """ This is an example when using legged_robot_field environment. Please override this for
        other purpose.
        NOTE: For the generality, returns the onehot id for each env.
        """
        obs_slice = get_obs_slice(self.obs_segments, "engaging_block")
        engaging_block_obs = observations[..., obs_slice[0]].reshape(*observations.shape[:-1], *obs_slice[1])
        obstacle_id_onehot = engaging_block_obs[..., 1:6]
        obstacle_id_onehot[torch.logical_not(obstacle_id_onehot.any(dim= -1)), 0] = 1. # if all zero, choose the first one
        return obstacle_id_onehot.to(torch.bool) # (N, ..., selection)
    
    def override_cmd_vel(self, observations, policy_selection):
        """ Override the velocity command based on proprioception (part of observation)
        """
        obs_slice = get_obs_slice(self.obs_segments, "proprioception")
        proprioception_obs = observations[..., obs_slice[0]].reshape(*observations.shape[:-1], *obs_slice[1])
        for idx, vel in self.cmd_vel_current.items():
            idx = int(idx)
            selected_proprioception = proprioception_obs[policy_selection[..., idx]]
            selected_proprioception[..., 9] = vel[policy_selection[..., idx]] if torch.is_tensor(vel) else vel
            selected_proprioception[..., 9] *= self.cmd_scales[idx]["lin_vel"]
            proprioception_obs[policy_selection[..., idx]] = selected_proprioception
        observations = torch.cat([
            observations[..., :obs_slice[0].start],
            proprioception_obs.reshape(*observations.shape[:-1], -1),
            observations[..., obs_slice[0].stop:],
        ], dim= -1)
        return observations

    @torch.no_grad()
    def act_inference(self, observations):
        # run entire batch for each sub policy in case the batch size and length problem.
        policy_selection = self.get_policy_selection(observations)
        if self.action_smoothing_buffer is None:
            self.action_smoothing_buffer = torch.zeros(
                self.action_smoothing_buffer_len,
                *policy_selection.shape,
                device= policy_selection.device,
                dtype= torch.float,
            ) # (len, N, ..., selection)
        self.action_smoothing_buffer = torch.cat([
            self.action_smoothing_buffer[1:],
            policy_selection.unsqueeze(0),
        ], dim= 0) # put the new one at the end
        observations = self.recover_last_action(observations, policy_selection)
        if self.cmd_vel_mapping:
            observations = self.override_cmd_vel(observations, policy_selection)
        outputs = [p.act_inference(observations) for p in self.submodules]
        output = torch.zeros_like(outputs[0])
        for idx, out in enumerate(outputs):
            output += out * getattr(self, "subpolicy_action_scale_{:d}".format(idx)) / self.env_action_scale \
                 * self.action_smoothing_buffer[..., idx].mean(dim= 0).unsqueeze(-1)
            # choose one or none reset method
            if self.reset_non_selected == "all" or self.reset_non_selected == True:
                self.submodules[idx].reset(self.action_smoothing_buffer[..., idx].sum(0) == 0)
            elif self.reset_non_selected == "when_skill" and idx > 0:
                self.submodules[idx].reset(torch.logical_and(
                    ~policy_selection[..., idx],
                    ~policy_selection[..., 0],
                ))
            # self.submodules[idx].reset(torch.ones(observations.shape[0], dtype= bool, device= observations.device))
        return output
    
    @torch.no_grad()
    def reset(self, dones=None):
        self.resample_cmd_vel_current(dones)
        return super().reset(dones)
    
class ActorCriticClimbMutex(ActorCriticFieldMutex):
    """ A variant to handle jump-up and jump-down with seperate policies
    Jump-down policy will be the last submodule in the list
    """
    JUMP_OBSTACLE_ID = 3 # starting from 0, referring to barrker_track.py:BarrierTrack.track_options_id_dict
    def __init__(self,
            *args,
            sub_policy_paths: list = None,
            jump_down_policy_path: str = None,
            jump_down_vel: float = None, # can be tuple/list, use it to stop using jump up velocity command
            **kwargs,):
        sub_policy_paths = sub_policy_paths + [jump_down_policy_path]
        self.jump_down_vel = jump_down_vel
        super().__init__(
            *args,
            sub_policy_paths= sub_policy_paths,
            **kwargs,
        )

    def resample_cmd_vel_current(self, dones=None):
        return_ = super().resample_cmd_vel_current(dones)
        if self.jump_down_vel is None:
            self.cmd_vel_current[len(self.submodules) - 1] = self.cmd_vel_current[self.JUMP_OBSTACLE_ID]
        elif isinstance(self.jump_down_vel, (tuple, list)):
            self.cmd_vel_current[len(self.submodules) - 1] = np.random.uniform(*self.jump_down_vel)
        else:
            self.cmd_vel_current[len(self.submodules) - 1] = self.jump_down_vel
        return return_

    def get_policy_selection(self, observations):
        obstacle_id_onehot = super().get_policy_selection(observations)
        obs_slice = get_obs_slice(self.obs_segments, "engaging_block")
        engaging_block_obs = observations[..., obs_slice[0]].reshape(*observations.shape[:-1], *obs_slice[1])
        jump_up_mask = engaging_block_obs[..., -1] > 0 # jump-up or jump-down
        obstacle_id_onehot = torch.cat([
            obstacle_id_onehot,
            torch.logical_and(
                obstacle_id_onehot[..., self.JUMP_OBSTACLE_ID],
                torch.logical_not(jump_up_mask),
            ).unsqueeze(-1)
        ], dim= -1)
        obstacle_id_onehot[..., self.JUMP_OBSTACLE_ID] = torch.logical_and(
            obstacle_id_onehot[..., self.JUMP_OBSTACLE_ID],
            jump_up_mask,
        )
        return obstacle_id_onehot.to(torch.bool) # (N, ..., selection)
