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

from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import time
import numpy as np
np.float = np.float32
import isaacgym
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, Logger
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_class_from_dict
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger

import numpy as np
import torch

def create_recording_camera(gym, env_handle,
        resolution= (1920, 1080),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

@torch.no_grad()
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.load_cfg:
        with open(os.path.join("logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
            d = json.load(f, object_pairs_hook= OrderedDict)
            update_class_from_dict(env_cfg, d, strict= True)
            update_class_from_dict(train_cfg, d, strict= True)

    # override some parameters for testing
    if env_cfg.terrain.selected == "BarrierTrack":
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.env.episode_length_s = 20
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 4
        env_cfg.terrain.num_cols = 8
        env_cfg.terrain.BarrierTrack_kwargs["options"] = [
            "jump",
            "leap",
            "down",
            "hurdle",
            "tilted_ramp",
            "stairsup",
            "discrete_rect",
            "wave",
        ]
        env_cfg.terrain.BarrierTrack_kwargs["leap"]["fake_offset"] = 0.1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.env.episode_length_s = 60
        env_cfg.terrain.terrain_length = 8
        env_cfg.terrain.terrain_width = 8
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
    # env_cfg.terrain.curriculum = False
    # env_cfg.asset.fix_base_link = True
    env_cfg.env.episode_length_s = 1000
    env_cfg.commands.resampling_time = int(1e16)
    env_cfg.commands.ranges.lin_vel_x = [1.2, 1.2]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.init_base_pos_range = dict(
        x= [0.6, 0.6],
        y= [-0.05, 0.05],
    )
    # env_cfg.termination.termination_terms = []
    env_cfg.termination.timeout_at_border = False
    env_cfg.termination.timeout_at_finished = False
    env_cfg.viewer.debug_viz = True
    env_cfg.viewer.draw_measure_heights = False
    env_cfg.viewer.draw_height_measurements = False
    env_cfg.viewer.draw_volume_sample_points = False
    env_cfg.viewer.draw_sensors = False
    if hasattr(env_cfg.terrain, "BarrierTrack_kwargs"):
        env_cfg.terrain.BarrierTrack_kwargs["draw_virtual_terrain"] = True
    # train_cfg.runner.resume = (args.load_run is not None)
    train_cfg.runner_class_name = "OnPolicyRunner"
    
    if args.no_throw:
        env_cfg.init_state.pos[2] = 0.4
        env_cfg.domain_rand.init_base_pos_range["x"] = [0.4, 0.4]
        env_cfg.domain_rand.init_base_vel_range = [0., 0.]
        env_cfg.domain_rand.init_dof_vel_range = [0., 0.]
        env_cfg.domain_rand.init_base_rot_range["roll"] = [0., 0.]
        env_cfg.domain_rand.init_base_rot_range["pitch"] = [0., 0.]
        env_cfg.domain_rand.init_base_rot_range["yaw"] = [0., 0.]
        env_cfg.domain_rand.init_base_vel_range = [0., 0.]
        env_cfg.domain_rand.init_dof_pos_ratio_range = [1., 1.]

    # default camera position
    # env_cfg.viewer.lookat = [0.6, 1.2, 0.5]
    # env_cfg.viewer.pos = [0.6, 0., 0.5]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    print("terrain_levels:", env.terrain_levels.float().mean(), env.terrain_levels.float().max(), env.terrain_levels.float().min())
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    # register debugging options to manually trigger disruption
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_P, "push_robot")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_J, "action_jitter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_ESCAPE, "exit")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_R, "agent_full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_U, "full_reset")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_C, "resample_commands")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_W, "forward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_S, "backward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_A, "leftward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_D, "rightward")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_F, "leftturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_G, "rightturn")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_B, "leftdrag")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_M, "rightdrag")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_X, "stop")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_K, "mark")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_I, "more_plots")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_T, "switch_teacher")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_O, "lean_fronter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_L, "lean_backer")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_COMMA, "lean_lefter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_PERIOD, "lean_righter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_DOWN, "slower")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_UP, "faster")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_LEFT, "lefter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_RIGHT, "righter")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_LEFT_BRACKET, "terrain_left")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_RIGHT_BRACKET, "terrain_right")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_MINUS, "terrain_back")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, isaacgym.gymapi.KEY_EQUAL, "terrain_forward")
    # load policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg= False,
    )
    agent_model = ppo_runner.alg.actor_critic
    policy = ppo_runner.get_inference_policy(device=env.device)
    ### get obs_slice to read the obs
    # obs_slice = get_obs_slice(env.obs_segments, "engaging_block")
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    if RECORD_FRAMES:
        os.mkdir(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images"), exist_ok= True)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*env_cfg.viewer.pos)
        transform.r = gymapi.Quat.from_euler_zyx(0., 0., -np.pi/2)
        recording_camera = create_recording_camera(
            env.gym,
            env.envs[0],
            transform= transform,
        )

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = args.plot_time # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.6, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    camera_follow_id = 0 # only effective when CAMERA_FOLLOW
    img_idx = 0

    if hasattr(env, "motor_strength"):
        print("motor_strength:", env.motor_strength[robot_index].cpu().numpy().tolist())
    print("torque_limits:", env.torque_limits)
    start_time = time.time_ns()
    for i in range(10*int(env.max_episode_length)):
        if "obs_slice" in locals().keys():
            obs_component = obs[:, obs_slice[0]].reshape(-1, *obs_slice[1])
            print(obs_component[robot_index])
        actions = policy(obs.detach())
        teacher_actions = actions
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            filename = os.path.join(
                os.path.abspath("logs/images/"),
                f"{img_idx:04d}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1
        if MOVE_CAMERA:
            if CAMERA_FOLLOW:
                camera_position[:] = env.root_states[camera_follow_id, :3].cpu().numpy() - camera_direction
            else:
                camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        for ui_event in env.gym.query_viewer_action_events(env.viewer):
            if ui_event.action == "push_robot" and ui_event.value > 0:
                # manully trigger to push the robot
                env._push_robots()
            if ui_event.action == "action_jitter" and ui_event.value > 0:
                # assuming wrong action is taken
                obs, critic_obs, rews, dones, infos = env.step(actions + torch.randn_like(actions) * 0.2)
            if ui_event.action == "exit" and ui_event.value > 0:
                print("exit")
                exit(0)
            if ui_event.action == "agent_full_reset" and ui_event.value > 0:
                print("agent_full_reset")
                agent_model.reset()
            if ui_event.action == "full_reset" and ui_event.value > 0:
                print("full_reset")
                agent_model.reset()
                if hasattr(ppo_runner.alg, "teacher_actor_critic"):
                    ppo_runner.alg.teacher_actor_critic.reset()
                # print(env._get_terrain_curriculum_move([robot_index]))
                obs, _ = env.reset()
            if ui_event.action == "resample_commands" and ui_event.value > 0:
                print("resample_commands")
                env._resample_commands(torch.arange(env.num_envs, device= env.device))
            if ui_event.action == "stop" and ui_event.value > 0:
                if hasattr(env, "sampled_x_cmd_buffer"):
                    env.sampled_x_cmd_buffer[:] = 0
                env.commands[:, :] = 0
                if hasattr(env, "orientation_cmds"):
                    env.orientation_cmds[:] = env.gravity_vec
                # env.stop_position.copy_(env.root_states[:, :3])
                # env.command_ranges["lin_vel_x"] = [0, 0]
                # env.command_ranges["lin_vel_y"] = [0, 0]
                # env.command_ranges["ang_vel_yaw"] = [0, 0]
            if ui_event.action == "forward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[1]
                # env.command_ranges["lin_vel_x"] = [env_cfg.commands.ranges.lin_vel_x[1], env_cfg.commands.ranges.lin_vel_x[1]]
            if ui_event.action == "backward" and ui_event.value > 0:
                env.commands[:, 0] = env_cfg.commands.ranges.lin_vel_x[0]
                # env.command_ranges["lin_vel_x"] = [env_cfg.commands.ranges.lin_vel_x[0], env_cfg.commands.ranges.lin_vel_x[0]]
            if ui_event.action == "leftward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[1]
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[1], env_cfg.commands.ranges.lin_vel_y[1]]
            if ui_event.action == "rightward" and ui_event.value > 0:
                env.commands[:, 1] = env_cfg.commands.ranges.lin_vel_y[0]
                # env.command_ranges["lin_vel_y"] = [env_cfg.commands.ranges.lin_vel_y[0], env_cfg.commands.ranges.lin_vel_y[0]]
            if ui_event.action == "leftturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[1]
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[1], env_cfg.commands.ranges.ang_vel_yaw[1]]
            if ui_event.action == "rightturn" and ui_event.value > 0:
                env.commands[:, 2] = env_cfg.commands.ranges.ang_vel_yaw[0]
                # env.command_ranges["ang_vel_yaw"] = [env_cfg.commands.ranges.ang_vel_yaw[0], env_cfg.commands.ranges.ang_vel_yaw[0]]
            if ui_event.action == "leftdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., 0.5, 0.]], device= env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "rightdrag" and ui_event.value > 0:
                env.root_states[:, 7:10] += quat_rotate(env.base_quat, torch.tensor([[0., -0.5, 0.]], device= env.device))
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.all_root_states))
            if ui_event.action == "mark" and ui_event.value > 0:
                logger.plot_states()
            if ui_event.action == "more_plots" and ui_event.value > 0:
                logger.plot_additional_states()
            if ui_event.action == "switch_teacher" and ui_event.value > 0:
                args.show_teacher = not args.show_teacher
                print("show_teacher:", args.show_teacher)
            if ui_event.action == "lean_fronter" and ui_event.value > 0 and hasattr(env, "orientation_cmds"):
                env.orientation_cmds[:, 0] += 0.1
                print("orientation_cmds:", env.orientation_cmds[:, 0])
            if ui_event.action == "lean_backer" and ui_event.value > 0 and hasattr(env, "orientation_cmds"):
                env.orientation_cmds[:, 0] -= 0.1
                print("orientation_cmds:", env.orientation_cmds[:, 0])
            if ui_event.action == "lean_lefter" and ui_event.value > 0 and hasattr(env, "orientation_cmds"):
                env.orientation_cmds[:, 1] += 0.1
                print("orientation_cmds:", env.orientation_cmds[:, 1])
            if ui_event.action == "lean_righter" and ui_event.value > 0 and hasattr(env, "orientation_cmds"):
                env.orientation_cmds[:, 1] -= 0.1
                print("orientation_cmds:", env.orientation_cmds[:, 1])
            if ui_event.action == "slower" and ui_event.value > 0:
                if hasattr(env, "sampled_x_cmd_buffer"):
                    env.sampled_x_cmd_buffer[:] -= 0.2
                env.commands[:, 0] -= 0.2
                print("command_x:", env.commands[:, 0])
            if ui_event.action == "faster" and ui_event.value > 0:
                if hasattr(env, "sampled_x_cmd_buffer"):
                    env.sampled_x_cmd_buffer[:] += 0.2
                env.commands[:, 0] += 0.2
                print("command_x:", env.commands[:, 0])
            if ui_event.action == "lefter" and ui_event.value > 0:
                if env.commands[:, 2] < 0:
                    env.commands[:, 2] = 0.
                else:
                    env.commands[:, 2] += 0.4
                print("command_yaw:", env.commands[:, 2])
            if ui_event.action == "righter" and ui_event.value > 0:
                if env.commands[:, 2] > 0:
                    env.commands[:, 2] = 0.
                else:
                    env.commands[:, 2] -= 0.4
                print("command_yaw:", env.commands[:, 2])
            if ui_event.action == "terrain_forward" and ui_event.value > 0:
                # env.cfg.terrain.curriculum = False
                env.terrain_levels[:] += 1
                env.terrain_levels = torch.clip(
                    env.terrain_levels,
                    min= 0,
                    max= env.cfg.terrain.num_rows - 1,
                )
                print("before", env.terrain_levels)
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
                env.reset()
                agent_model.reset()
                print("after", env.terrain_levels)
            if ui_event.action == "terrain_back" and ui_event.value > 0:
                # env.cfg.terrain.curriculum = False
                env.terrain_levels[:] -= 1
                env.terrain_levels = torch.clip(
                    env.terrain_levels,
                    min= 0,
                    max= env.cfg.terrain.num_rows - 1,
                )
                print("before", env.terrain_levels)
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
                env.reset()
                agent_model.reset()
                print("after", env.terrain_levels)
            if ui_event.action == "terrain_right" and ui_event.value > 0:
                # env.cfg.terrain.curriculum = False
                env.terrain_types[:] -= 1
                env.terrain_types = torch.clip(
                    env.terrain_types,
                    min= 0,
                    max= env.cfg.terrain.num_cols - 1,
                )
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
                env.reset()
                agent_model.reset()
            if ui_event.action == "terrain_left" and ui_event.value > 0:
                # env.cfg.terrain.curriculum = False
                env.terrain_types[:] += 1
                env.terrain_types = torch.clip(
                    env.terrain_types,
                    min= 0,
                    max= env.cfg.terrain.num_cols - 1,
                )
                env.env_origins[:] = env.terrain_origins[env.terrain_levels[:], env.terrain_types[:]]
                env.reset()
                agent_model.reset()
        # if (env.contact_forces[robot_index, env.feet_indices, 2] > 200).any():
        #     print("contact_forces:", env.contact_forces[robot_index, env.feet_indices, 2])
        # if (abs(env.substep_torques[robot_index]) > 35.).any():
            # exceed_idxs = torch.where(abs(env.substep_torques[robot_index]) > 35.)
            # print("substep_torques:", exceed_idxs[1], env.substep_torques[robot_index][exceed_idxs[0], exceed_idxs[1]])
        if env.torque_exceed_count_envstep[robot_index].any():
            print("substep torque exceed limit ratio", 
                (torch.abs(env.substep_torques[robot_index]) / (env.torque_limits.unsqueeze(0))).max(),
                "joint index",
                torch.where((torch.abs(env.substep_torques[robot_index]) > env.torque_limits.unsqueeze(0) * env.cfg.rewards.soft_torque_limit).any(dim= 0))[0],
                "timestep", i,
            )
            env.torque_exceed_count_envstep[robot_index] = 0
        # if (torch.abs(env.torques[robot_index]) > env.torque_limits.unsqueeze(0) * env.cfg.rewards.soft_torque_limit).any():
        #     print("torque exceed limit ratio",
        #         (torch.abs(env.torques[robot_index]) / (env.torque_limits.unsqueeze(0))).max(),
        #         "joint index",
        #         torch.where((torch.abs(env.torques[robot_index]) > env.torque_limits.unsqueeze(0) * env.cfg.rewards.soft_torque_limit).any(dim= 0))[0],
        #         "timestep", i,
        #     )
        # dof_exceed_mask = ((env.dof_pos[robot_index] > env.dof_pos_limits[:, 1]) | (env.dof_pos[robot_index] < env.dof_pos_limits[:, 0]))
        # if dof_exceed_mask.any():
        #     print("dof pos exceed limit: joint index",
        #         torch.where(dof_exceed_mask)[0],
        #         "amount",
        #         torch.maximum(
        #             env.dof_pos[robot_index][dof_exceed_mask] - env.dof_pos_limits[dof_exceed_mask][:, 1],
        #             env.dof_pos_limits[dof_exceed_mask][:, 0] - env.dof_pos[robot_index][dof_exceed_mask],
        #         ),
        #         "dof value:",
        #         env.dof_pos[robot_index][dof_exceed_mask],
        #         "timestep", i,
        #     )

        if i < stop_state_log:
            if torch.is_tensor(env.cfg.control.action_scale):
                action_scale = env.cfg.control.action_scale.detach().cpu().numpy()[joint_index]
            else:
                action_scale = env.cfg.control.action_scale
            base_roll = get_euler_xyz(env.base_quat)[0][robot_index].item()
            base_pitch = get_euler_xyz(env.base_quat)[1][robot_index].item()
            if base_pitch > torch.pi: base_pitch -= torch.pi * 2
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * action_scale,
                    'dof_pos': (env.dof_pos - env.default_dof_pos)[robot_index, joint_index].item(),
                    'dof_vel': env.substep_dof_vel[robot_index, 0, joint_index].max().item(),
                    'dof_torque': env.substep_torques[robot_index, 0, joint_index].max().item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_pitch': base_pitch,
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'max_torques': torch.abs(env.substep_torques).max().item(),
                    "student_action": actions[robot_index, 2].item(),
                    "teacher_action": teacher_actions[robot_index, 2].item(),
                    "reward": rews[robot_index].item(),
                    'all_dof_vel': env.substep_dof_vel[robot_index].mean(-2).cpu().numpy(),
                    'all_dof_torque': env.substep_torques[robot_index].mean(-2).cpu().numpy(),
                    "power": torch.max(torch.sum(env.substep_torques * env.substep_dof_vel, dim= -1), dim= -1)[0][robot_index].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
            env._get_terrain_curriculum_move(torch.tensor([0], device= env.device))
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        if dones.any():
            agent_model.reset(dones)
            if env.time_out_buf[dones].any():
                print("env dones because of timeout")
            else:
                print("env dones because of failure")
            # print(infos)
        if i % 100 == 0:
            print("frame_rate:" , 100/(time.time_ns() - start_time) * 1e9, 
                  "command_x:", env.commands[robot_index, 0],
            )
            start_time = time.time_ns()

if __name__ == '__main__':
    EXPORT_POLICY = False
    args = get_args([
        dict(name= "--slow", type= float, default= 0., help= "slow down the simulation by sleep secs (float) every frame"),
        dict(name= "--show_teacher", action= "store_true", default= False, help= "show teacher actions"),
        dict(name= "--no_teacher", action= "store_true", default= False, help= "whether to disable teacher policy when running the script"),
        dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
        dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
        dict(name= "--plot_time", type= int, default= -1, help= "plot states after this time"),
        dict(name= "--no_throw", action= "store_true", default= False),
        dict(name= "--load_cfg", action= "store_true", default= False, help= "use the config from the logdir"),
        dict(name= "--record", action= "store_true", default= False, help= "record frames"),
        dict(name= "--frames_dir", type= str, default= "images", help= "which folder to store intermediate recorded frames."),
    ])
    MOVE_CAMERA = (args.num_envs is None)
    CAMERA_FOLLOW = MOVE_CAMERA
    RECORD_FRAMES = args.record
    try:
        play(args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        if RECORD_FRAMES and args.load_run is not None:
            import subprocess
            print("converting frames to video")
            log_dir = args.load_run if os.path.isabs(args.load_run) \
                else os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    task_registry.get_cfgs(name=args.task)[1].runner.experiment_name,
                    args.load_run,
                )
            subprocess.run(["ffmpeg",
                "-framerate", "50",
                "-r", "50",
                "-i", "logs/images/%04d.png",
                "-c:v", "libx264",
                "-hide_banner", "-loglevel", "error",
                os.path.join(log_dir, f"video_{args.checkpoint}.mp4")
            ])
            print("done converting frames to video, deleting frame images")
            for f in os.listdir(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "images")):
                os.remove(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.frames_dir, f))
            print("done deleting frame images")