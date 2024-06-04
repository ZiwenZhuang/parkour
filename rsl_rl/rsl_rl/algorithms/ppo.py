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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 velocity_planner,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 clip_min_std= 1e-15, # clip the policy.std if it supports, check update()
                 optimizer_class_name= "Adam",
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 **kwargs
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = getattr(optim, optimizer_class_name)(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Velocity Planner
        self.velocity_planner = velocity_planner
        self.velocity_optimizer = getattr(optim, optimizer_class_name)(self.velocity_planner.parameters(), lr=learning_rate)
        self.lin_vel_x = kwargs.get('lin_vel_x', None)
        self.command_scale = kwargs.get('command_scale', 2.0)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_min_std = torch.tensor(clip_min_std, device= self.device) if isinstance(clip_min_std, (tuple, list)) else clip_min_std
        
        # algorithm status
        self.current_learning_iteration = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
        self.velocity_planner.eval()
    
    def train_mode(self):
        self.actor_critic.train()
        self.velocity_planner.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        vel_obs = torch.cat([obs[..., :9], obs[..., 12:]], dim=-1)
        velocity = self.velocity_planner(vel_obs)
        # if self.lin_vel_x is not None:
        #     velocity = torch.clip(velocity, self.lin_vel_x[0], self.lin_vel_x[1])
        self.transition.actions = self.actor_critic.act(obs, velocity=velocity * self.command_scale)[0].detach()
        # critic_obs[..., 9] = velocity.squeeze(-1) * self.command_scale
        self.transition.values = self.actor_critic.evaluate(vel_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = vel_obs
        self.transition.critic_observations = vel_obs
        return self.transition.actions, velocity.squeeze().detach()
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_critic_obs = torch.cat([last_critic_obs[..., :9], last_critic_obs[..., 12:]], dim=-1)
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(lambda :0.)
        average_stats = defaultdict(lambda :0.)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:

                losses, _, stats = self.compute_losses(minibatch, current_learning_iteration=current_learning_iteration)

                loss = 0.
                for k, v in losses.items():
                    loss += getattr(self, k + "_coef", 1.) * v
                    mean_losses[k] = mean_losses[k] + v.detach()
                mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
                for k, v in stats.items():
                    average_stats[k] = average_stats[k] + v.detach()

                # Gradient step
                self.optimizer.zero_grad()
                self.velocity_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.velocity_optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            average_stats[k] = average_stats[k] / num_updates
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min= self.clip_min_std)

        return mean_losses, average_stats

    def compute_losses(self, minibatch, current_learning_iteration=None):
        obs = copy.deepcopy(minibatch.obs)

        # vel_obs = torch.cat([obs[..., :9], obs[..., 12:]], dim=-1)

        velocity = self.velocity_planner(obs)
        zeros = torch.zeros_like(velocity, requires_grad=False)
        obs_with_cmd = torch.cat([obs[..., :9], velocity, zeros, zeros, obs[..., 9:]], dim=-1)
        # if self.lin_vel_x is not None:
        #     velocity = torch.clip(velocity, self.lin_vel_x[0], self.lin_vel_x[1])
        self.actor_critic.act(obs_with_cmd, masks=minibatch.masks, hidden_states=minibatch.hid_states[0], velocity=velocity * self.command_scale)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch = self.actor_critic.evaluate(obs, masks=minibatch.masks, hidden_states=minibatch.hid_states[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                            torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate


        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        surrogate = -torch.squeeze(minibatch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param,
                                                                                                    self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - value_batch).pow(2).mean()
        
        # Velocity loss
        if current_learning_iteration is None:
            vel_loss = 0
        else:
            vel_loss = torch.square(velocity-2).mean() * np.exp(-0.001 * current_learning_iteration + 12)
            vel_loss += torch.square(torch.clamp_max(velocity, 1.) - 1).mean()

        return_ = dict(
            surrogate_loss= surrogate_loss,
            value_loss= value_loss,
            vel_loss = vel_loss
        )
        if entropy_batch is not None:
            return_["entropy"] = - entropy_batch.mean()
        
        inter_vars = dict(
            ratio= ratio,
            surrogate= surrogate,
            surrogate_clipped= surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == 'adaptive':
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped
        return return_, inter_vars, dict()
