import random
from collections import OrderedDict, defaultdict
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
import cv2

class LeggedRobotParkour(LeggedRobot):

    # ------------init--------------
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._prepare_termination_function()

    def step(self, actions, commands=None):
        if commands is not None:
            self.commands = commands
        super().step(actions)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def pre_physics_step(self, actions):
        self.volume_sample_points_refreshed = False
        actions_preprocessed = False
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
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
            actions_preprocessed = True
        elif getattr(self.cfg.normalization, "clip_actions_method", None) == "hard":
            if self.cfg.control.control_type == "P":
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
                raise NotImplementedError()
            actions_preprocessed = True
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        if not actions_preprocessed:
            return super().pre_physics_step(actions)

    def post_decimation_step(self, dec_i):
        return_ = super().post_decimation_step(dec_i)
        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        ### The set torque limit is usally smaller than the robot dataset
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits).any(dim= -1)] += 1
        ### Hack to check the torque limit exceeding by your own value.
        # self.torque_exceed_count_envstep[(torch.abs(self.torques) > 38.).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        
        return return_
    
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.roll, self.pitch, self.yaw = get_euler_xyz(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._update_goals()
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
        self.last_root_pos[:] = self.root_states[:, :3]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def check_termination(self):
        return super().check_termination()
    

    # ---------CallBacks-------------
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
            pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
            self.extras["episode"]["max_pos_x"] = max(self.extras["episode"]["max_pos_x"], torch.max(pos_x).cpu())
            self.extras["episode"]["min_pos_x"] = min(self.extras["episode"]["min_pos_x"], torch.min(pos_x).cpu())
            self.extras["episode"]["max_pos_y"] = max(self.extras["episode"]["max_pos_y"], torch.max(pos_y).cpu())
            self.extras["episode"]["min_pos_y"] = min(self.extras["episode"]["min_pos_y"], torch.min(pos_y).cpu())
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()

        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
        )

        if hasattr(self, "actions_history_buffer"):
            resampling_time = getattr(self.cfg.control, "action_delay_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_action_delay(resample_env_ids)

        if hasattr(self, "proprioception_buffer"):
            resampling_time = getattr(self.cfg.sensor.proprioception, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_proprioception_latency(resample_env_ids)
        
        if hasattr(self, "forward_depth_buffer"):
            resampling_time = getattr(self.cfg.sensor.forward_camera, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_forward_camera_latency(resample_env_ids)

        self.torque_exceed_count_envstep[(torch.abs(self.substep_torques) > self.torque_limits).any(dim= 1).any(dim= 1)] += 1

    # -------------------------------
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        
        super()._init_buffers()

        # additional gym GPU state tensors
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # additional wrapper tensors
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()

        # additional data initializations
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.sensor_tensor_dict = defaultdict(list)

        # gym sensing tensors
        for env_i, env_handle in enumerate(self.envs):
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

        # projected gravity bias (if needed)
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            print("Initializing gravity bias for domain randomization")
            # add cross trajectory domain randomization on projected gravity bias
            # uniform sample from range
            self.gravity_bias = torch.rand(self.num_envs, 3, dtype= torch.float, device= self.device, requires_grad= False)
            self.gravity_bias[:, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[:, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)
        all_obs_components = self.all_obs_components

        # init action delay buffer
        if getattr(self.cfg.control, "action_delay", False):
            assert hasattr(self.cfg.control, "action_delay_range") and hasattr(self.cfg.control, "action_delay_resample_time"), "Please specify action_delay_range and action_delay_resample_time in the config file."
            """ Used in pre-physics step """
            self.cfg.control.action_history_buffer_length = int((self.cfg.control.action_delay_range[1] + self.dt) / self.dt)
            self.actions_history_buffer = torch.zeros(
                (
                    self.cfg.control.action_history_buffer_length,
                    self.num_envs,
                    self.num_actions,
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_action_delay = torch_rand_float(
                self.cfg.control.action_delay_range[0],
                self.cfg.control.action_delay_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.action_delayed_frames = ((self.current_action_delay / self.dt) + 1).to(int)

        # init proprioception delay buffer
        if "proprioception" in all_obs_components and hasattr(self.cfg.sensor, "proprioception"):
            """ Adding proprioception delay buffer """
            self.cfg.sensor.proprioception.buffer_length = int((self.cfg.sensor.proprioception.latency_range[1] + self.dt) / self.dt)
            self.proprioception_buffer = torch.zeros(
                (
                    self.cfg.sensor.proprioception.buffer_length,
                    self.num_envs,
                    self.get_num_obs_from_components(["proprioception"]),
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_proprioception_latency = torch_rand_float(
                self.cfg.sensor.proprioception.latency_range[0],
                self.cfg.sensor.proprioception.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)

        # init depth sensor and delay
        if "forward_depth" in all_obs_components and hasattr(self.cfg.sensor, "forward_camera"):
            output_resolution = getattr(self.cfg.sensor.forward_camera, "output_resolution", self.cfg.sensor.forward_camera.resolution)
            self.cfg.sensor.forward_camera.buffer_length = int((self.cfg.sensor.forward_camera.latency_range[1] + self.cfg.sensor.forward_camera.refresh_duration) / self.dt)
            self.forward_depth_buffer = torch.zeros(
                (
                    self.cfg.sensor.forward_camera.buffer_length,
                    self.num_envs, 
                    1,
                    output_resolution[0],
                    output_resolution[1],
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.forward_depth_delayed_frames = torch.ones((self.num_envs,), device= self.device, dtype= int) * self.cfg.sensor.forward_camera.buffer_length
            self.current_forward_camera_latency = torch_rand_float(
                self.cfg.sensor.forward_camera.latency_range[0],
                self.cfg.sensor.forward_camera.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            if hasattr(self.cfg.sensor.forward_camera, "resized_resolution"):
                self.forward_depth_resize_transform = T.Resize(
                    self.cfg.sensor.forward_camera.resized_resolution,
                    interpolation= T.InterpolationMode.BICUBIC,
                )
        
        self.contour_detection_kernel = torch.zeros(
            (8, 1, 3, 3),
            dtype= torch.float32,
            device= self.device,
        )
        # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def _create_envs(self):
        if self.cfg.domain_rand.randomize_motor:
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        return_ = super()._create_envs()
        
        front_hip_names = getattr(self.cfg.asset, "front_hip_names", ["FR_hip_joint", "FL_hip_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_hip_names = getattr(self.cfg.asset, "rear_hip_names", ["RR_hip_joint", "RL_hip_joint"])
        self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_hip_names):
            self.rear_hip_indices[i] = self.dof_names.index(name)

        hip_names = front_hip_names + rear_hip_names
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        
        return return_
    
    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
    
    def _prepare_termination_function(self):
        self.termination_functions = []
        for key in self.cfg.termination.termination_terms:
            name = '_termination_' + key
            self.termination_functions.append(getattr(self, name))
    
    def _termination_roll(self):
        roll_cutoff = torch.abs(self.roll) > self.cfg.termination.termination_threshold['roll']
        self.reset_buf |= roll_cutoff
    
    def _termination_pitch(self):
        pitch_cutoff = torch.abs(self.pitch) > self.cfg.termination.termination_threshold['pitch']
        self.reset_buf |= pitch_cutoff
    
    def _termination_height(self):
        if isinstance(self.cfg.termination.termination_threshold['height'], (tuple, list)):
            z = self.root_states[:, 2] - self.env_origins[:, 2]
            height_low_cutoff = z < self.cfg.termination.termination_threshold['height'][0]
            height_high_cutoff = z > self.cfg.termination.termination_threshold['height'][1]
            self.reset_buf |= height_low_cutoff
            self.reset_buf |= height_high_cutoff
    
    
