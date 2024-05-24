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

from collections import OrderedDict
import torch
import numpy as np

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    NOTE: The output shape will also be [time, number of envs, additional dimensions].
        But `time` might be smaller than the input `time`.
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    if padded_trajectories.shape[0] < tensor.shape[0]:
    #     trajectory_masks = trajectory_masks[:padded_trajectories.shape[0]]
        padded_trajectories = torch.cat([
            padded_trajectories,
            torch.empty((tensor.shape[0] - padded_trajectories.shape[0], *padded_trajectories.shape[1:]), device= tensor.device),
        ], dim= 0)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, masks.shape[0], trajectories.shape[-1]).transpose(1, 0)

def get_obs_slice(segments: OrderedDict, component_name: str):
    """ Get the slice from segments and name. Return the slice and component shape """
    obs_start = obs_end = 0
    component_shape = None
    for k, v in segments.items():
        obs_start = obs_end
        obs_end = obs_start + np.prod(v)
        if k == component_name:
            component_shape = v # tuple
            break
    assert component_shape is not None, "No component ({}) is found in the given components {}".format(component_name, [segments.keys()])
    return slice(obs_start, obs_end), component_shape

""" NOTE:
* Loop through obs_segments to get the same order of components defined in obs_segments
* These operations does not require the obs to be a 2-d tensor, but the last dimension must be packed
    with a connected set of components.
"""
def get_subobs_size(obs_segments, component_names):
    """ Compute the size of a subset of observations. """
    obs_size = 0
    for component in obs_segments.keys():
        if component in component_names:
            obs_slice, _ = get_obs_slice(obs_segments, component)
            obs_size += obs_slice.stop - obs_slice.start
    return obs_size

def get_subobs_by_components(observations, component_names, obs_segments):
    """ Get a subset of observations from the full observation tensor. """
    estimator_input = []
    for component in obs_segments.keys():
        if component in component_names:
            obs_slice, _ = get_obs_slice(obs_segments, component)
            estimator_input.append(observations[..., obs_slice])
    return torch.cat(estimator_input, dim= -1) # NOTE: this is a 2-d tensor with (batch_size, obs_size)

def substitute_estimated_state(observations, target_components, estimated_state, obs_segments):
    """ Substitute the estimated state into part of the observations.
    """
    estimated_state_start = 0
    for component in obs_segments:
        if component in target_components:
            obs_slice, obs_shape = get_obs_slice(obs_segments, component)
            estimated_state_end = estimated_state_start + np.prod(obs_shape)
            observations[..., obs_slice] = estimated_state[..., estimated_state_start:estimated_state_end]
            estimated_state_start = estimated_state_end
    return observations

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

@torch.jit.script
def quat_to_rotmat(q):
    """ q: shape (N, 4) quaternion """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotmat = torch.zeros(q.shape[0], 3, 3, device= q.device)
    rotmat[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    rotmat[:, 0, 1] = 2*x*y - 2*z*w
    rotmat[:, 0, 2] = 2*x*z + 2*y*w
    rotmat[:, 1, 0] = 2*x*y + 2*z*w
    rotmat[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    rotmat[:, 1, 2] = 2*y*z - 2*x*w
    rotmat[:, 2, 0] = 2*x*z - 2*y*w
    rotmat[:, 2, 1] = 2*y*z + 2*x*w
    rotmat[:, 2, 2] = 1 - 2*x**2 - 2*y**2
    return rotmat

def rotmat_to_euler_zxy(mat):
    """ mat: shape (N, 3, 3) 3d rotation matrix """
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    y = torch.atan2(mat[:, 0, 2], mat[:, 2, 2]) # y
    x = torch.asin(-mat[:, 1, 2]) # x
    z = torch.atan2(mat[:, 1, 0], mat[:, 1, 1]) # z
    y = wrap_to_pi(y)
    x = wrap_to_pi(x)
    z = wrap_to_pi(z)
    return z, x, y

def rotmat_to_euler_yzx(mat):
    """ mat: shape (N, 3, 3) 3d rotation matrix """
    # get the rotation parameters in x(q0)z(q1)y(q2) sequence
    x = torch.atan2(mat[:, 2, 1], mat[:, 1, 1]) # x
    z = torch.asin(-mat[:, 0, 1]) # z
    y = torch.atan2(mat[:, 0, 2], mat[:, 0, 0]) # y
    x = wrap_to_pi(x)
    z = wrap_to_pi(z)
    y = wrap_to_pi(y)
    return y, z, x

def rotmat_to_euler_xzy(mat):
    """ mat: shape (N, 3, 3) 3d rotation matrix """
    # get the rotation parameters in y(q0)z(q1)x(q2) sequence
    y = torch.atan2(-mat[:, 2, 0], mat[:, 0, 0]) # y
    z = torch.asin(mat[:, 1, 0]) # z
    x = torch.atan2(-mat[:, 1, 2], mat[:, 1, 1]) # x
    y = wrap_to_pi(y)
    z = wrap_to_pi(z)
    x = wrap_to_pi(x)
    return x, z, y