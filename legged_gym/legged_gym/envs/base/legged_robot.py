# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
import itertools
from collections import OrderedDict, defaultdict
from copy import copy

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    # Due the safety of using getattr(), we need to specify the available sensors here
    available_sensors = [
        "proprioception",
        "forward_camera",
    ]
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", False)
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.pre_decimation_step(dec_i)
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions):
        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_low", None), (tuple, list)):
            self.cfg.normalization.clip_actions_low = torch.tensor(
                self.cfg.normalization.clip_actions_low,
                device= self.device
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_high", None), (tuple, list)):
            self.cfg.normalization.clip_actions_high = torch.tensor(
                self.cfg.normalization.clip_actions_high,
                device= self.device
            )
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        # some customized action clip methods to bound the action output
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
        elif getattr(self.cfg.normalization, "clip_actions_method", None) == "hard":
            actions_low = getattr(
                self.cfg.normalization, "clip_actions_low",
                self.dof_pos_limits[:, 0] - self.default_dof_pos,
            )
            actions_high = getattr(
                self.cfg.normalization, "clip_actions_high",
                self.dof_pos_limits[:, 1] - self.default_dof_pos,
            )
            self.actions = torch.clip(actions, actions_low, actions_high)
        else:
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # update some buffers before the source are refreshed
        self.last_contact_forces[:] = self.contact_forces
        self.volume_sample_points_refreshed = False

    def pre_decimation_step(self, dec_i):
        self.last_dof_vel[:] = self.dof_vel

    def post_decimation_step(self, dec_i):
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])
        self.substep_exceed_dof_pos_limit_abs[:, dec_i, :] = torch.clip(torch.maximum(
            self.dof_pos_limits[:, 0] - self.dof_pos,
            self.dof_pos - self.dof_pos_limits[:, 1],
        ), min= 0) # make sure the value is non-negative

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _check_abnormal_dynamics(self):
        """ Check if there are abnormal dynamics
        """
        assert hasattr(self.cfg, "termination") and hasattr(self.cfg.termination, "abnormal_dynamics_kwargs")
        if self.cfg.termination.abnormal_dynamics_kwargs.get("max_contact_force", None) is not None:
            max_contact_force = self.cfg.termination.abnormal_dynamics_kwargs["max_contact_force"]
            return torch.norm(self.contact_forces, dim= -1).max(-1)[0] > max_contact_force
        if self.cfg.termination.abnormal_dynamics_kwargs.get("max_root_lin_vel", None) is not None:
            max_root_lin_vel = self.cfg.termination.abnormal_dynamics_kwargs["max_root_lin_vel"]
            return torch.norm(self.root_states[:, 7:10], dim= -1) > max_root_lin_vel
        if self.cfg.termination.abnormal_dynamics_kwargs.get("max_root_ang_vel", None) is not None:
            max_root_ang_vel = self.cfg.termination.abnormal_dynamics_kwargs["max_root_ang_vel"]
            return torch.norm(self.root_states[:, 10:13], dim= -1) > max_root_ang_vel
        if self.cfg.termination.abnormal_dynamics_kwargs.get("max_dof_vel", None) is not None:
            max_dof_vel = self.cfg.termination.abnormal_dynamics_kwargs["max_dof_vel"]
            return torch.abs(self.dof_vel).max(-1)[0] > max_dof_vel
        if self.cfg.termination.abnormal_dynamics_kwargs.get("max_dof_acc", None) is not None:
            max_dof_acc = self.cfg.termination.abnormal_dynamics_kwargs["max_dof_acc"]
            return torch.abs(self.dof_vel - self.last_dof_vel).max(-1)[0] > max_dof_acc

        return torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        if hasattr(self.cfg, "termination"):
            # more sophisticated termination conditions
            r, p, y = get_euler_xyz(self.base_quat)
            r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
            p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
            z = self.root_states[:, 2] - self.env_origins[:, 2]

            if "roll" in self.cfg.termination.termination_terms:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]
                self.reset_buf |= r_term_buff
            if "pitch" in self.cfg.termination.termination_terms:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]
                self.reset_buf |= p_term_buff
            if "z_low" in self.cfg.termination.termination_terms:
                z_term_buff = z < self.cfg.termination.z_low_kwargs["threshold"]
                self.reset_buf |= z_term_buff
            if "z_high" in self.cfg.termination.termination_terms:
                z_term_buff = z > self.cfg.termination.z_high_kwargs["threshold"]
                self.reset_buf |= z_term_buff
            if "abnormal_dynamics" in self.cfg.termination.termination_terms:
                # not updating time_out_buf so that policy might learn to avoid abnormal dynamics situation
                self.abnormal_dynamics_buf = self._check_abnormal_dynamics()
                if self.cfg.termination.abnormal_dynamics_kwargs.get("as_timeout", False):
                    self.time_out_buf |= self.abnormal_dynamics_buf
                else:
                    self.reset_buf |= self.abnormal_dynamics_buf
                self.abnormal_traj_length_mean = self.episode_length_buf[self.abnormal_dynamics_buf].to(torch.float32).mean().item()

        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "timeout_at_border", False):
            border_buff = self.terrain.in_terrain_range(self.root_states[:, :3]).logical_not()
            self.time_out_buf |= border_buff

        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._reset_buffers(env_ids)

    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _get_lin_vel_obs(self, privileged= False):
        # backward compatibile for proprioception obs components and use_lin_vel related args
        obs_buf = self.base_lin_vel.clone()
        if (not privileged) and (not getattr(self.cfg.env, "use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        if privileged and (not getattr(self.cfg.env, "privileged_use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        return obs_buf
    
    def _get_ang_vel_obs(self, privileged= False):
        return self.base_ang_vel
    
    def _get_projected_gravity_obs(self, privileged= False):
        return self.projected_gravity

    def _get_commands_obs(self, privileged= False):
        return self.commands[:, :3]
    
    def _get_dof_pos_obs(self, privileged= False):
        return (self.dof_pos - self.default_dof_pos)
    
    def _get_dof_vel_obs(self, privileged= False):
        return self.dof_vel
    
    def _get_last_actions_obs(self, privileged= False):
        return self.actions

    def _get_height_measurements_obs(self, privileged= False):
        # not tested
        height_offset = getattr(self.cfg.normalization, "height_measurements_offset", -0.5)
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) + height_offset - self.measured_heights, -1, 1.)
        return heights

    def _get_base_pose_obs(self, privileged= False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        return torch.cat([
            self.root_states[:, :3] - self.env_origins,
            torch.stack([roll, pitch, yaw], dim= -1),
        ], dim= -1)
    
    def _get_robot_config_obs(self, privileged= False):
        return self.robot_config_buffer

    def _get_forward_depth_obs(self, privileged= False):
        return torch.stack(self.sensor_tensor_dict["forward_depth"]).flatten(start_dim= 1)

    ##### The wrapper function to build and help processing observations #####
    def _get_obs_from_components(self, components: list, privileged= False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            # get the observation from specific component name
            # such as "_get_lin_vel_obs", "_get_ang_vel_obs", "_get_dof_pos_obs", "_get_forward_depth_obs"
            obs.append(
                getattr(self, "_get_" + k + "_obs")(privileged) * \
                getattr(self.obs_scales, k, 1.)
            )
        obs = torch.cat(obs, dim= 1)
        return obs

    # defines observation segments, which tells the order of the entire flattened obs
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "lin_vel" in components:
            segments["lin_vel"] = (3,)
        if "ang_vel" in components:
            segments["ang_vel"] = (3,)
        if "projected_gravity" in components:
            segments["projected_gravity"] = (3,)
        if "commands" in components:
            segments["commands"] = (3,)
        if "dof_pos" in components:
            segments["dof_pos"] = (self.num_dof,)
        if "dof_vel" in components:
            segments["dof_vel"] = (self.num_dof,)
        if "last_actions" in components:
            segments["last_actions"] = (self.num_actions,)
        if "height_measurements" in components:
            assert self.cfg.terrain.measure_heights, "You must set measure_heights to True in terrain config to use height_measurements observation component."
            segments["height_measurements"] = (1, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y))
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + self.num_actions,)
        return segments

    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
    
    def compute_observations(self):
        """ Computes observations
        """
        # force refresh graphics if needed
        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        if hasattr(self.cfg.env, "privileged_obs_components"):
            self.privileged_obs_buf = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", True),
            )
        else:
            self.privileged_obs_buf = None

        # wrap up to read the graphics data
        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break
        
        # add simple noise if needed
        if self.add_noise == "uniform" or self.add_noise == True:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        elif self.add_noise == "gaussian":
            self.obs_buf += torch.randn_like(self.obs_buf) * self.noise_scale_vec
    
    # utility functions to meet old APIs and fit new obs logic
    """ Some critical concepts:
    - obs_components: a list of strings (no order required), each string is a name of a component
    - obs_segments: an OrderedDict, each key is a string, each value is a tuple of ints representing the shape of the component.
    - num_obs: an int, the total number of observations
    - all_obs_components: a set of strings, the union of obs_components and privileged_obs_components
    """
    @property
    def all_obs_components(self):
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def obs_segments(self):
        return self.get_obs_segment_from_components(self.cfg.env.obs_components)
    @property
    def privileged_obs_segments(self):
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            if "proprioception" in components:
                # backward compatibile for proprioception obs components
                print("\033[1;36m Warning: proprioception is deprecated, use lin_vel, ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions instead.\033[0;0m")
                components.remove("proprioception")
                components += ["lin_vel", "ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel", "last_actions"]
            return self.get_obs_segment_from_components(components)
    @property
    def num_obs(self):
        """ get this value from self.cfg.env """
        if "proprioception" in self.cfg.env.obs_components:
            # backward compatibile for proprioception obs components
            print("\033[1;36m Warning: proprioception is deprecated, use lin_vel, ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions instead.\033[0;0m")
            self.cfg.env.obs_components.remove("proprioception")
            self.cfg.env.obs_components = ["lin_vel", "ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel", "last_actions"] + self.cfg.env.obs_components
        return self.get_num_obs_from_components(self.cfg.env.obs_components)
    @num_obs.setter
    def num_obs(self, value):
        """ avoid setting self.num_obs """
        pass
    @property
    def num_privileged_obs(self):
        """ get this value from self.cfg.env """
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_num_obs_from_components(components)
    @num_privileged_obs.setter
    def num_privileged_obs(self, value):
        """ avoid setting self.num_privileged_obs """
        pass
    @property
    def num_actions(self):
        return self.num_dof
    @num_actions.setter
    def num_actions(self, value):
        """ avoid setting self.num_actions """
        pass
    #Done The wrapper function to build and help processing observations #####

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_terrain()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction for viewer
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # allow config to override torque limits
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits, dtype= torch.float, device= self.device, requires_grad= False)
        # asset options to override dof props
        if hasattr(self.cfg.asset, "dof_damping_override"):
            if isinstance(self.cfg.asset.dof_damping_override, (tuple, list)):
                props["damping"][:] = torch.tensor(self.cfg.asset.dof_damping_override, dtype= torch.float, device= self.device, requires_grad= False)
            else:
                props["damping"][:] = self.cfg.asset.dof_damping_override
        if hasattr(self.cfg.asset, "dof_friction_override"):
            if isinstance(self.cfg.asset.dof_friction_override, (tuple, list)):
                props["friction"][:] = torch.tensor(self.cfg.asset.dof_friction_override, dtype= torch.float, device= self.device, requires_grad= False)
            else:
                props["friction"][:] = self.cfg.asset.dof_friction_override
        if hasattr(self.cfg.asset, "dof_velocity_override"):
            if isinstance(self.cfg.asset.dof_velocity_override, (tuple, list)):
                props["velocity"][:] = torch.tensor(self.cfg.asset.dof_velocity_override, dtype= torch.float, device= self.device, requires_grad= False)
            else:
                props["velocity"][:] = self.cfg.asset.dof_velocity_override
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if getattr(self.cfg.domain_rand, "randomize_com", False):
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # log max power across current env step
        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
        )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        lin_cmd_cutoff = getattr(self.cfg.commands, "lin_cmd_cutoff", 0.2)
        ang_cmd_cutoff = getattr(self.cfg.commands, "ang_cmd_cutoff", 0.1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > lin_cmd_cutoff).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > ang_cmd_cutoff)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if getattr(self.cfg.domain_rand, "init_dof_pos_ratio_range", None) is not None:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.init_dof_pos_ratio_range[0],
                self.cfg.domain_rand.init_dof_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_vel[env_ids] = 0. # history init method
        dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-3., 3.])
        self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)

        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        dof_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        dof_idx_int32 = dof_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.all_dof_states),
                                              gymtorch.unwrap_tensor(dof_idx_int32), len(dof_idx_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "init_base_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(env_ids), 1), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation (roll and pitch)
        if hasattr(self.cfg.domain_rand, "init_base_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_yaw = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range.get("yaw", [-np.pi, np.pi]),
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
            self.root_states[env_ids, 3:7] = base_quat
        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        if isinstance(base_vel_range, (tuple, list)):
            self.root_states[env_ids, 7:13] = torch_rand_float(
                *base_vel_range,
                (len(env_ids), 6),
                device=self.device,
            ) # [7:10]: lin vel, [10:13]: ang vel
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")
        
        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        actor_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        move_up, move_down = self._get_terrain_curriculum_move(env_ids)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_terrain_curriculum_move(self, env_ids):
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        return move_up, move_down
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        obs_segments = self.get_obs_segment_from_components(cfg.env.obs_components)
        # write noise for each corresponding component.
        for k, v in obs_segments.items():
            segment_length = np.prod(v)
            # write sensor scale to provided noise_vec
            # for example "_write_forward_depth_noise"
            getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
            segment_start_idx += segment_length

        return noise_vec

    def _write_lin_vel_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.lin_vel * self.cfg.noise.noise_level * self.obs_scales.lin_vel

    def _write_ang_vel_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.ang_vel * self.cfg.noise.noise_level * self.obs_scales.ang_vel
    
    def _write_projected_gravity_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.gravity * self.cfg.noise.noise_level
    
    def _write_commands_noise(self, noise_vec):
        noise_vec[:] = 0.

    def _write_dof_pos_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.dof_pos * self.cfg.noise.noise_level * self.obs_scales.dof_pos

    def _write_dof_vel_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.dof_vel * self.cfg.noise.noise_level * self.obs_scales.dof_vel

    def _write_last_actions_noise(self, noise_vec):
        noise_vec[:] = 0.

    def _write_height_measurements_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

    def _write_forward_depth_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.forward_depth * self.cfg.noise.noise_level * self.obs_scales.forward_depth

    def _write_base_pose_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "base_pose"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.base_pose * self.cfg.noise.noise_level * self.obs_scales.base_pose
    
    def _write_robot_config_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_config"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_config * self.cfg.noise.noise_level * getattr(self.obs_scales, "robot_config", 1.)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :] # (num_envs, 13)
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[:, :self.num_dof, :] # (num_envs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limit_abs = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)

        if hasattr(self.cfg.sim, "body_measure_points"):
            self._init_body_volume_points()
            self._init_volume_sample_points()
            print("Total number of volume estimation points for each robot is:", self.volume_sample_points.shape[1])

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # motor_strength
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if getattr(self.cfg.domain_rand, "randomize_motor", False):
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        # robot_config
        all_obs_components = self.all_obs_components
        if "robot_config" in all_obs_components:
            self.robot_config_buffer = torch.empty(
                self.num_envs, 1 + 3 + 1 + self.num_actions,
                dtype= torch.float32,
                device= self.device,
            )
            assert len(self.envs) == len(self.actor_handles), "Number of envs and actor_handles must be the same. Other actor handles in the env must be put in npc_handles."
            for env_id, (env_h, actor_h) in enumerate(zip(self.envs, self.actor_handles)):
                actor_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_h, actor_h)
                actor_dof_props = self.gym.get_actor_dof_properties(env_h, actor_h)
                actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_h, actor_h)
                self.robot_config_buffer[env_id, 0] = actor_rigid_shape_props[0].friction
                self.robot_config_buffer[env_id, 1] = actor_rigid_body_props[0].com.x
                self.robot_config_buffer[env_id, 2] = actor_rigid_body_props[0].com.y
                self.robot_config_buffer[env_id, 3] = actor_rigid_body_props[0].com.z
                self.robot_config_buffer[env_id, 4] = actor_rigid_body_props[0].mass
            self.robot_config_buffer[:, 5:5+self.num_actions] = self.motor_strength

        # sensor tensors
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state) # (num_envs*num_bodies, 13)
        # adding sensor_tensor_dict for acquiring sensor tensors
        self.sensor_tensor_dict = defaultdict(list)
        for env_i, env_handle in enumerate(self.envs):
            self._init_sensor_buffers(env_i, env_handle)

    def _init_body_volume_points(self):
        """ Generate a series of points grid so that they can be bind to those rigid bodies
        By 'those rigid bodies', we mean the rigid bodies that are specified in the `body_measure_points`
        """
        # read and specify the order of which body to sample from and its relative sample points.
        self.body_measure_name_order = [] # order specified
        self.body_sample_indices = [] # index in environment domain
        # NOTE: assuming all envs have the same number of actors and rigid bodies
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        for name, measure_name in itertools.product(rigid_body_names, self.cfg.sim.body_measure_points.keys()):
            if measure_name in name:
                self.body_sample_indices.append(
                    self.gym.find_actor_rigid_body_index(
                        self.envs[0],
                        self.actor_handles[0],
                        name,
                        gymapi.IndexDomain.DOMAIN_ENV,
                ))
                self.body_measure_name_order.append(measure_name) # order specified
        self.body_sample_indices = torch.tensor(self.body_sample_indices, device= self.sim_device).flatten() # n_bodies (each env)

        # compute and store each sample points in body frame.
        self.body_volume_points = dict()
        for measure_name, points_cfg in self.cfg.sim.body_measure_points.items():
            x = torch.tensor(points_cfg["x"], device= self.device, dtype= torch.float32, requires_grad= False)
            y = torch.tensor(points_cfg["y"], device= self.device, dtype= torch.float32, requires_grad= False)
            z = torch.tensor(points_cfg["z"], device= self.device, dtype= torch.float32, requires_grad= False)
            t = torch.tensor(points_cfg["transform"][0:3], device= self.device, dtype= torch.float32, requires_grad= False)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid_points = torch.stack([
                grid_x.flatten(),
                grid_y.flatten(),
                grid_z.flatten(),
            ], dim= -1) # n_points, 3
            if "points" in points_cfg.keys():
                # additional unstructured points
                unstructured_points = torch.tensor(points_cfg["points"], device= self.device, dtype= torch.float32)
                assert len(unstructured_points.shape) == 2 and unstructured_points.shape[1] == 3, "Unstructured points must be a list of 3D points"
                grid_points = torch.cat([
                    grid_points,
                    unstructured_points,
                ], dim= 0)
            q = quat_from_euler_xyz(
                torch.tensor(points_cfg["transform"][3], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][4], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][5], device= self.sim_device, dtype= torch.float32),
            )
            self.body_volume_points[measure_name] = tf_apply(
                q.expand(grid_points.shape[0], 4),
                t.expand(grid_points.shape[0], 3),
                grid_points,
            )

    def _init_volume_sample_points(self):
        """ Build sample points for penetration volume estimation
        NOTE: self.cfg.sim.body_measure_points must be a dict with
            key: body name (or part of the body name) to estimate
            value: dict(
                x, y, z: sample points to form a meshgrid
                transform: [x, y, z, roll, pitch, yaw] for transforming the meshgrid w.r.t body frame
            )
        """
        num_sample_points_per_env = 0
        for body_name in self.body_measure_name_order:
            for measure_name in self.body_volume_points.keys():
                if measure_name in body_name:
                    num_sample_points_per_env += self.body_volume_points[measure_name].shape[0]
        self.volume_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )
        self.volume_sample_points_vel = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )

    def refresh_volume_sample_points(self):
        """ NOTE: call this method whenever you need to access `volume_sample_points` and `volume_sample_points_vel`,
        Don't worry about repeated calls, it will only compute once in each env.step() call.
        """
        if self.volume_sample_points_refreshed:
            # use `volume_sample_points_refreshed` to avoid repeated computation
            return
        sample_points_start_idx = 0
        for body_idx, body_measure_name in enumerate(self.body_measure_name_order):
            volume_points = self.body_volume_points[body_measure_name] # (n_points, 3)
            num_volume_points = volume_points.shape[0]
            rigid_body_index = self.body_sample_indices[body_idx:body_idx+1] # size: torch.Size([1])
            point_positions_w, point_velocities_w = self._get_target_pos_vel(
                rigid_body_index.expand(num_volume_points,),
                volume_points,
                domain= gymapi.DOMAIN_ENV,
            )
            self.volume_sample_points_vel[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_velocities_w
            self.volume_sample_points[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_positions_w
            sample_points_start_idx += num_volume_points
        self.volume_sample_points_refreshed = True

    def _init_sensor_buffers(self, env_i, env_handle):
        if "forward_depth" in self.all_obs_components:
            self.sensor_tensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    env_handle,
                    self.sensor_handles[env_i]["forward_camera"],
                    gymapi.IMAGE_DEPTH,
            )))
        if "forward_color" in self.all_obs_components:
            self.sensor_tensor_dict["forward_color"].append(gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    env_handle,
                    self.sensor_handles[env_i]["forward_camera"],
                    gymapi.IMAGE_COLOR,
            )))

    def _reset_buffers(self, env_ids):
        if getattr(self.cfg.init_state, "zero_actions", False):
            self.actions[env_ids] = 0.
        if hasattr(self, "volume_sample_points_vel"):
            self.volume_sample_points_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.max_power_per_timestep[env_ids] = 0.

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_sensors(self, env_handle= None, actor_handle= None):
        """ attach necessary sensors for each actor in each env
        Considering only one robot in each environment, this method takes only one actor_handle.
        Args:
            env_handle: env_handle from gym.create_env
            actor_handle: actor_handle from gym.create_actor
        Return:
            sensor_handle_dict: a dict of sensor_handles with key as sensor name (defined in cfg["sensor"])
        """
        sensor_handle_dict = dict()
        all_obs_components = self.all_obs_components

        if "forward_depth" in all_obs_components or "forward_color" in all_obs_components:
            camera_handle = self._create_onboard_camera(env_handle, actor_handle, "forward_camera")
            sensor_handle_dict["forward_camera"] = camera_handle
            
        return sensor_handle_dict
    
    def _create_onboard_camera(self, env_handle, actor_handle, sensor_name):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = getattr(self.cfg.sensor, sensor_name).resolution[0]
        camera_props.width = getattr(self.cfg.sensor, sensor_name).resolution[1]
        if hasattr(getattr(self.cfg.sensor, sensor_name), "near_plane"):
            camera_props.near_plane = getattr(self.cfg.sensor, sensor_name).near_plane
        if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
            camera_props.horizontal_fov = np.random.uniform(
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[0],
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[1],
            ) if isinstance(getattr(self.cfg.sensor, sensor_name).horizontal_fov, (tuple, list)) else getattr(self.cfg.sensor, sensor_name).horizontal_fov
            # vertical_fov = horizontal_fov * camera_props.height / camera_props.width
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        if isinstance(getattr(self.cfg.sensor, sensor_name).position, dict):
            # allow domain randomization across robots.
            # sample from "mean" and "std" attributes.
            # each must be a list of 3 elements.
            cam_x = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][0],
                getattr(self.cfg.sensor, sensor_name).position["std"][0],
            )
            cam_y = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][1],
                getattr(self.cfg.sensor, sensor_name).position["std"][1],
            )
            cam_z = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][2],
                getattr(self.cfg.sensor, sensor_name).position["std"][2],
            )
            local_transform.p = gymapi.Vec3(cam_x, cam_y, cam_z)
        else:
            local_transform.p = gymapi.Vec3(*getattr(self.cfg.sensor, sensor_name).position)
        if isinstance(getattr(self.cfg.sensor, sensor_name).rotation, dict):
            # allow domain randomization across robots
            # sample from "lower" and "upper" attributes.
            # each must be a list of 3 elements (in radian).
            cam_roll = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][0] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            cam_pitch = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][1] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            cam_yaw = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][2] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            local_transform.r = gymapi.Quat.from_euler_zyx(cam_yaw, cam_pitch, cam_roll)
        else:
            local_transform.r = gymapi.Quat.from_euler_zyx(*getattr(self.cfg.sensor, sensor_name).rotation)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        
        return camera_handle

    def _create_npc(self, env_handle, env_idx):
        """ create additional opponent for each environment such as static objects, random agents
        or turbulance.
        """
        return dict()

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        # for computing dof_error_named reward
        if hasattr(self.cfg.rewards, "dof_error_names"):
            self.dof_error_named_indices = torch.tensor(
                [self.dof_names.index(name) for name in self.cfg.rewards.dof_error_names],
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.npc_handles = [] # surrounding actors or objects or oppoents in each environment.
        self.sensor_handles = []
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
            npc_handle_dict = self._create_npc(env_handle, i)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.sensor_handles.append(sensor_handle_dict)
            self.npc_handles.append(npc_handle_dict)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _create_terrain(self):
        if getattr(self.cfg.terrain, "selected", None) is None:
            self._create_ground_plane()
        else:
            terrain_cls = self.cfg.terrain.selected
            self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs)
            self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if getattr(self.cfg.terrain, "selected", None) is not None:
            assert getattr(self.cfg.terrain, "mesh_type", None) is None, "Cannot have both terrain.selected and terrain.mesh_type"
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = min(self.cfg.terrain.max_init_terrain_level, self.terrain.cfg.num_rows - 1)
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = getattr(self.cfg.env, "env_spacing", 3.)
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = copy(self.cfg.normalization.obs_scales)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    ##### draw debug vis and the sub functions #####
    def _draw_measure_heights_vis(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
    def _draw_height_measurements_vis(self):
        """ Draws height measurements as animated depth image
        """
        if "height_measurements" in self.all_obs_components:
            # plot height measurements as animated depth image
            import matplotlib.pyplot as plt
            height_measurements_shape = self.get_obs_segment_from_components("height_measurements")["height_measurements"]
            height_measurements = self._get_height_measurements_obs()
            height_measurements_0 = height_measurements[0].cpu().numpy() # already range [-1, 1]
            height_measurements_0 = height_measurements_0.reshape(height_measurements_shape[1:])
            plt.imshow(height_measurements_0, vmin=-1., vmax=1., cmap='gray')
            plt.pause(0.001)

    def _draw_sensor_vis(self, env_h, sensor_hd):
        for sensor_name, sensor_h in sensor_hd.items():
            if "camera" in sensor_name:
                camera_transform = self.gym.get_camera_transform(self.sim, env_h, sensor_h)
                cam_axes = gymutil.AxesGeometry(scale= 0.1)
                gymutil.draw_lines(cam_axes, self.gym, self.viewer, env_h, camera_transform)

    def _draw_sensor_reading_vis(self, env_h, sensor_hd):
        """ Draw sensor readings by ploting it """
        pass

    def _draw_commands_vis(self):
        """ """
        xy_commands = (self.commands[:, :3] * getattr(self.obs_scales, "commands", 1.)).clone()
        yaw_commands = xy_commands.clone()
        xy_commands[:, 2] = 0.
        yaw_commands[:, :2] = 0.
        color = gymapi.Vec3(*self.cfg.viewer.commands.color)
        xy_commands_global = tf_apply(self.root_states[:, 3:7], self.root_states[:, :3], xy_commands * self.cfg.viewer.commands.size)
        yaw_commands_global = tf_apply(self.root_states[:, 3:7], self.root_states[:, :3], yaw_commands * self.cfg.viewer.commands.size)
        for i in range(self.num_envs):
            gymutil.draw_line(
                gymapi.Vec3(*self.root_states[i, :3].cpu().tolist()),
                gymapi.Vec3(*xy_commands_global[i].cpu().tolist()),
                color,
                self.gym,
                self.viewer,
                self.envs[i],
            )
            gymutil.draw_line(
                gymapi.Vec3(*self.root_states[i, :3].cpu().tolist()),
                gymapi.Vec3(*yaw_commands_global[i].cpu().tolist()),
                color,
                self.gym,
                self.viewer,
                self.envs[i],
            )

    def _draw_volume_sample_points_vis(self):
        self.refresh_volume_sample_points()
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0., 1., 0.))
        for env_idx in range(self.num_envs):
            for point_idx in range(self.volume_sample_points.shape[1]):
                sphere_pose = gymapi.Transform(gymapi.Vec3(*self.volume_sample_points[env_idx, point_idx]), r= None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # clear all drawings first
        self.gym.clear_lines(self.viewer)

        # draw debug visuals
        if getattr(self.terrain.cfg, "measure_heights", False) and getattr(self.cfg.viewer, "draw_measure_heights", False):
            self._draw_measure_heights_vis()
        if getattr(self.cfg.viewer, "draw_height_measurements", False):
            self._draw_height_measurements_vis()
        if getattr(self.cfg.viewer, "draw_sensors", False):
            for env_h, sensor_hd in zip(self.envs, self.sensor_handles):
                self._draw_sensor_vis(env_h, sensor_hd)
        if getattr(self.cfg.viewer, "draw_sensor_readings", False):
            for env_h, sensor_hd in zip(self.envs, self.sensor_handles):
                self._draw_sensor_reading_vis(env_h, sensor_hd)
        if self.cfg.viewer.draw_commands:
            self._draw_commands_vis()
        if hasattr(self, "volume_sample_points") and getattr(self.cfg.viewer, "draw_volume_sample_points", False):
            self._draw_volume_sample_points_vis()

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        return self.terrain.get_terrain_heights(points)

    def _get_robot_dof_indices(self, target_dof_name):
        """ A helper function for later acquiring joint-specific data from the simulation. """
        joint_indices = [] # 1-d tensor (length depends on the robot)
        dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        for i in range(len(dof_names)):
            if target_dof_name in dof_names[i]:
                joint_indices.append(self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], dof_names[i]))
        return torch.tensor(joint_indices, dtype=torch.long, device=self.device, requires_grad=False)
    
    def _get_target_pos_vel(self, target_link_indices, target_local_pos, domain= gymapi.DOMAIN_SIM):
        """ Get the target pos/vel in world frame, given the body_indices (SIM_DOMAIN) and
        the local position of which this target is rigidly attached to.
        
        NOTE: Before this method, `refresh_rigid_body_state_tensor` must be called before this 
        function and after each `gym.simulate`.

        Args:
            target_link_indices (torch.Tensor): shape (n_envs*n_targets_per_robot) for DOMAIN_SIM,
                or (n_targets_per_robot,) for DOMAIN_ENV
            target_local_pos (torch.Tensor): shape (n_targets_per_robot, 3)
            domain: gymapi.DOMAIN_SIM or gymapi.DOMAIN_ENV, not recommending using gymapi.DOMAIN_ACTOR
                since a env may contain multiple actors with different bodies.

        Returns:
            target_pos_world (torch.Tensor): shape (n_envs, n_targets_per_robot, 3)
            target_vel_world (torch.Tensor): shape (n_envs, n_targets_per_robot, 3)
        """
        # shape: (n_envs, n_targets_per_robot, 13)
        if domain == gymapi.DOMAIN_SIM:
            target_body_states = self.all_rigid_body_states[target_link_indices].view(self.num_envs, -1, 13)
        elif domain == gymapi.DOMAIN_ENV:
            # NOTE: maybe also acceping DOMAIN_ACTOR, but do this at your own risk
            target_body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, target_link_indices]
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        # shape: (n_envs, n_targets_per_robot, 3)
        target_pos = target_body_states[:, :, 0:3]
        # shape: (n_envs, n_targets_per_robot, 4)
        target_quat = target_body_states[:, :, 3:7]
        # shape: (n_envs * n_targets_per_robot, 3)
        target_pos_world_ = tf_apply(
            target_quat.view(-1, 4),
            target_pos.view(-1, 3),
            target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape).reshape(-1, 3), # using reshape because of contiguous issue
        )
        # shape: (n_envs, n_targets_per_robot, 3)
        target_pos_world = target_pos_world_.view(self.num_envs, -1, 3)
        # shape: (n_envs, n_targets_per_robot, 3)
        # NOTE: assuming the angular velocity here is the same as the time derivative of the axis-angle
        target_vel_world = torch.cross(
            target_body_states[:, :, 10:13],
            target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape),
            dim= -1,
        )
        return target_pos_world, target_vel_world
    
    def _get_feet_heights(self):
        """ A helper function that returns the heights of the feet of the robot, w.r.t the terrain.
        Only available when `self.terrain.get_terrain_heights` is available.
        """
        if not hasattr(self, "feet_indices"):
            raise AttributeError("feet_indices is not available")
        # shape: (n_envs, n_feet)
        foot_heights = self.terrain.get_terrain_heights(self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :3])
        return self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, 2] - foot_heights
    
    def _fill_extras(self, env_ids):
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_frame_' + key] = torch.nanmean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if len(env_ids) > 0:
                self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels[env_ids].float())
                self.extras["episode"]["terrain_level_min"] = torch.min(self.terrain_levels[env_ids].float())
        # log power related info
        self.extras["episode"]["max_power_throughout_episode"] = self.max_power_per_timestep[env_ids].max().cpu().item()
        # log running range info
        pos_x = self.root_states[env_ids][:, 0] - self.env_origins[env_ids][:, 0]
        pos_y = self.root_states[env_ids][:, 1] - self.env_origins[env_ids][:, 1]
        self.extras["episode"]["max_pos_x"] = torch.max(pos_x).cpu()
        self.extras["episode"]["min_pos_x"] = torch.min(pos_x).cpu()
        self.extras["episode"]["max_pos_y"] = torch.max(pos_y).cpu()
        self.extras["episode"]["min_pos_y"] = torch.min(pos_y).cpu()

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = self.time_out_buf.float().sum() / self.reset_buf.float().sum()
        self.extras["episode"]["num_terminated"] = self.reset_buf.float().sum()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # log abnormal dynamics ratio among all reset episodes
        if hasattr(self, "abnormal_dynamics_buf"):
            self.extras["episode"]["abnormal_dynamics_ratio"] = self.abnormal_dynamics_buf.float().sum() / self.reset_buf.float().sum()
            self.extras["episode"]["abnormal_traj_length_mean"] = self.abnormal_traj_length_mean

    #------------ reward functions----------------
    def _reward_alive(self):
        return 1.
    
    def _reward_lin_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim= 1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_dof_error_named(self):
        """ Reward for a given named joints """
        assert hasattr(self, "dof_error_named_indices"), "dof_error_named_indices is not available, please set 'dof_error_names' in cfg.rewards to use this reward."
        dof_error = torch.sum(torch.square(self.dof_pos[:, self.dof_error_named_indices] - self.default_dof_pos[:, self.dof_error_named_indices]), dim=1)
        return dof_error
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        last_contact = self.last_contact_forces[:, self.feet_indices, 2] > 1.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, last_contact) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1) \
            * (torch.abs(self.commands[:, 2] < 0.2))
    
    def _reward_stop_lin_vel(self):
        # Penalize x/y/z speed at zero commands
        return torch.sum(torch.square(self.base_lin_vel), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1) \
            * (torch.abs(self.commands[:, 2] < 0.2))
    
    def _reward_stop_ang_vel(self):
        # Penalize angular speed at zero commands
        return torch.sum(torch.square(self.base_ang_vel), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1) \
            * (torch.abs(self.commands[:, 2] < 0.2))
    
    def _reward_stop_dof_vel(self):
        # Penalize dof velocities at zero commands
        return torch.sum(torch.square(self.dof_vel), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1) \
            * (torch.abs(self.commands[:, 2] < 0.2))
    
    def _reward_stop_yaw_vel(self):
        # Penalize yaw speed at zero commands
        return torch.square(self.base_ang_vel[:, 2]) \
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1) \
            * (torch.abs(self.commands[:, 2] < 0.2))
    
    def _reward_lazy_stop(self):
        # Penalize too slow when command is not below cutoff threshold
        return (torch.norm(self.root_states[:, 7:9] - self.commands[:, :2], dim=1) > getattr(self.cfg.commands, "lin_cmd_cutoff", 0.2)) \
            * torch.logical_or(
                (torch.norm(self.commands[:, :2], dim=1) > getattr(self.cfg.commands, "lin_cmd_cutoff", 0.2)),
                (torch.abs(self.commands[:, 2]) > getattr(self.cfg.commands, "ang_cmd_cutoff", 0.1)),
            )

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)
    
    def _reward_exceed_torque_limits_square(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.square(exceeded_torques).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_exceed_torque_limit_ratio(self):
        """ ratio of exceeded torque to the limit """
        torques_ratio = torch.abs(self.substep_torques) / (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        torques_ratio[torques_ratio < 1.] = 0.
        # square sum along decimation axis and dof axis
        return torch.square(torques_ratio).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)
    
    def _reward_exceed_dof_pos_limit_abs(self):
        return self.substep_exceed_dof_pos_limit_abs.sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limit_square(self):
        return torch.square(self.substep_exceed_dof_pos_limit_abs).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limit_ratio(self):
        """ Return the ratio of exceeding dof_position limit, exceed_abs / range """
        substep_ratio = self.substep_exceed_dof_pos_limit_abs / (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        ).unsqueeze(0).unsqueeze(0) # shape: (n_envs, n_substeps, n_dofs)
        return torch.square(substep_ratio).sum(dim= 1).sum(dim= 1)
