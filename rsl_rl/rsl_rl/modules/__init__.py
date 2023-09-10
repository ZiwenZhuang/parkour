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

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .visual_actor_critic import VisualDeterministicRecurrent, VisualDeterministicAC
from .actor_critic_mutex import ActorCriticMutex
from .actor_critic_field_mutex import ActorCriticFieldMutex, ActorCriticClimbMutex

def build_actor_critic(env, policy_class_name, policy_cfg):
    """ NOTE: This method allows to hack the policy kwargs by adding the env attributes to the policy_cfg. """
    actor_critic_class = globals()[policy_class_name] # ActorCritic

    policy_cfg = policy_cfg.copy()
    if env.num_privileged_obs is not None:
        num_critic_obs = env.num_privileged_obs 
    else:
        num_critic_obs = env.num_obs
    if hasattr(env, "obs_segments") and "obs_segments" not in policy_cfg:
        policy_cfg["obs_segments"] = env.obs_segments
    if hasattr(env, "privileged_obs_segments") and "privileged_obs_segments" not in policy_cfg:
        policy_cfg["privileged_obs_segments"] = env.privileged_obs_segments
    if not "num_actor_obs" in policy_cfg:
        policy_cfg["num_actor_obs"] = env.num_obs
    if not "num_critic_obs" in policy_cfg:
        policy_cfg["num_critic_obs"] = num_critic_obs
    if not "num_actions" in policy_cfg:
        policy_cfg["num_actions"] = env.num_actions
    
    actor_critic: ActorCritic = actor_critic_class(**policy_cfg)

    return actor_critic