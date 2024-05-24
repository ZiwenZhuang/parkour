import random

import numpy as np
from isaacgym.torch_utils import torch_rand_float, tf_apply
from isaacgym import gymutil, gymapi
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class LeggedRobotNoisyMixin:
    """ This class should be independent from the terrain, but depend on the sensors of the parent
    class.
    """

    def clip_position_action_by_torque_limit(self, actions_scaled):
        """ For position control, scaled actions should be in the coordinate of robot default dof pos
        """
        if hasattr(self, "dof_vel_obs_output_buffer"):
            dof_vel = self.dof_vel_obs_output_buffer / self.obs_scales.dof_vel
        else:
            dof_vel = self.dof_vel
        if hasattr(self, "dof_pos_obs_output_buffer"):
            dof_pos_ = self.dof_pos_obs_output_buffer / self.obs_scales.dof_pos
        else:
            dof_pos_ = self.dof_pos - self.default_dof_pos
        p_limits_low = (-self.torque_limits) + self.d_gains*dof_vel
        p_limits_high = (self.torque_limits) + self.d_gains*dof_vel
        actions_low = (p_limits_low/self.p_gains) + dof_pos_
        actions_high = (p_limits_high/self.p_gains) + dof_pos_
        actions_scaled_clipped = torch.clip(actions_scaled, actions_low, actions_high)
        return actions_scaled_clipped

    def pre_physics_step(self, actions):
        self.set_buffers_refreshed_to_false()
        return_ = super().pre_physics_step(actions)

        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        if getattr(self.cfg.control, "computer_clip_torque", False):
            self.actions_scaled = self.actions * self.cfg.control.action_scale
            control_type = self.cfg.control.control_type
            if control_type == "P":
                actions_scaled_clipped = self.clip_position_action_by_torque_limit(self.actions_scaled)
            else:
                raise NotImplementedError
        else:
            actions_scaled_clipped = self.actions * self.cfg.control.action_scale

        if getattr(self.cfg.control, "action_delay", False):
            # always put the latest action at the end of the buffer
            self.actions_history_buffer = torch.roll(self.actions_history_buffer, shifts= -1, dims= 0)
            self.actions_history_buffer[-1] = actions_scaled_clipped
            # get the delayed action
            action_delayed_frames = ((self.action_delay_buffer / self.dt) + 1).to(int)
            self.actions_scaled_clipped = self.actions_history_buffer[
                -action_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ]
        else:
            self.actions_scaled_clipped = actions_scaled_clipped
        
        return return_
    
    def _compute_torques(self, actions):
        """ The input actions will not be used, instead the scaled clipped actions will be used.
        Please check the computation logic whenever you change anything.
        """
        if not hasattr(self.cfg.control, "motor_clip_torque"):
            return super()._compute_torques(actions)
        else:
            if hasattr(self, "motor_strength"):
                actions_scaled_clipped = self.motor_strength * self.actions_scaled_clipped
            else:
                actions_scaled_clipped = self.actions_scaled_clipped
            control_type = self.cfg.control.control_type
            if control_type == "P":
                torques = self.p_gains * (actions_scaled_clipped + self.default_dof_pos - self.dof_pos) \
                    - self.d_gains * self.dof_vel
            else:
                raise NotImplementedError
            if self.cfg.control.motor_clip_torque:
                torques = torch.clip(
                    torques,
                    -self.torque_limits * self.cfg.control.motor_clip_torque,
                    self.torque_limits * self.cfg.control.motor_clip_torque,
                )
            return torques
        
    def post_decimation_step(self, dec_i):
        return_ = super().post_decimation_step(dec_i)
        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        ### The set torque limit is usally smaller than the robot dataset
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits * self.cfg.rewards.soft_torque_limit).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        ### or using a1_const.h value to check whether the robot is out of dof pos limit
        # joint_pos_limit_high = torch.tensor([0.802, 4.19, -0.916] * 4, device= self.device) - 0.001
        # joint_pos_limit_low = torch.tensor([-0.802, -1.05, -2.7] * 4, device= self.device) + 0.001
        # self.out_of_dof_pos_limit_count_substep += (self.dof_pos > joint_pos_limit_high.unsqueeze(0)).sum(-1).int()
        # self.out_of_dof_pos_limit_count_substep += (self.dof_pos < joint_pos_limit_low.unsqueeze(0)).sum(-1).int()
        
        return return_
    
    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)
        
        self.extras["episode"]["max_torques"] = self.max_torques[env_ids]
        self.max_torques[env_ids] = 0.
        self.extras["episode"]["torque_exceed_count_substeps_per_envstep"] = self.torque_exceed_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.torque_exceed_count_substep[env_ids] = 0
        self.extras["episode"]["torque_exceed_count_envstep"] = self.torque_exceed_count_envstep[env_ids]
        self.torque_exceed_count_envstep[env_ids] = 0
        self.extras["episode"]["out_of_dof_pos_limit_count_substep"] = self.out_of_dof_pos_limit_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.out_of_dof_pos_limit_count_substep[env_ids] = 0

        return return_
    
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        if hasattr(self, "actions_history_buffer"):
            resampling_time = getattr(self.cfg.control, "action_delay_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_action_delay(resample_env_ids)

        for sensor_name in self.available_sensors:
            if hasattr(self, sensor_name + "_latency_buffer"):
                self._resample_sensor_latency_if_needed(sensor_name)

        self.torque_exceed_count_envstep[(torch.abs(self.substep_torques) > self.torque_limits * self.cfg.rewards.soft_torque_limit).any(dim= 1).any(dim= 1)] += 1

    def _init_buffers(self):
        return_ = super()._init_buffers()
        all_obs_components = self.all_obs_components

        if getattr(self.cfg.control, "action_delay", False):
            assert hasattr(self.cfg.control, "action_delay_range") and hasattr(self.cfg.control, "action_delay_resample_time"), "Please specify action_delay_range and action_delay_resample_time in the config file."
            self.build_action_delay_buffer()

        self.component_governed_by_sensor = dict()
        for sensor_name in self.available_sensors:
            if hasattr(self.cfg.sensor, sensor_name):
                self.set_latency_buffer_for_sensor(sensor_name)
                for component in getattr(self.cfg.sensor, sensor_name).obs_components:
                    assert not hasattr(self, component + "_obs_buffer"), "The obs component {} already has a buffer and corresponding sensor. Should not be governed also by {}".format(component, sensor_name)
                    self.set_obs_buffers_for_component(component, sensor_name)
                    if component == "forward_depth":
                        self.build_depth_image_processor_buffers(sensor_name)

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
        return return_

    def build_action_delay_buffer(self):
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
        self.action_delay_buffer = torch_rand_float(
                self.cfg.control.action_delay_range[0],
                self.cfg.control.action_delay_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
        self.action_delayed_frames = ((self.action_delay_buffer / self.dt) + 1).to(int)

    def build_depth_image_processor_buffers(self, sensor_name):
        assert sensor_name == "forward_camera", "Only forward_camera is supported for now."
        if hasattr(getattr(self.cfg.sensor, sensor_name), "resized_resolution"):
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
    
    """ Considering sensors are not necessarily matching observation component,
    we need to set the buffers for each obs component and latency buffer for each sensor.
    """
    def set_obs_buffers_for_component(self, component, sensor_name):
        buffer_length = int(getattr(self.cfg.sensor, sensor_name).latency_range[1] / self.dt) + 1
        # use super().get_obs_segment_from_components() to get the obs shape to prevent post processing
        # overrides the buffer shape
        obs_buffer = torch.zeros(
            (
                buffer_length,
                self.num_envs,
                *(self.get_obs_segment_from_components([component])[component]), # tuple(obs_shape)
            ),
            dtype= torch.float32,
            device= self.device,
        )
        setattr(self, component + "_obs_buffer", obs_buffer)
        setattr(self, component + "_obs_refreshed", False)
        self.component_governed_by_sensor[component] = sensor_name

    def set_latency_buffer_for_sensor(self, sensor_name):
        latency_buffer = torch_rand_float(
            getattr(self.cfg.sensor, sensor_name).latency_range[0],
            getattr(self.cfg.sensor, sensor_name).latency_range[1],
            (self.num_envs, 1),
            device= self.device,
        ).flatten()
        # using setattr to set the buffer
        setattr(self, sensor_name + "_latency_buffer", latency_buffer)
        if "camera" in sensor_name:
            setattr(
                self,
                sensor_name + "_delayed_frames",
                torch.zeros_like(latency_buffer, dtype= torch.long, device= self.device),
            )

    def _resample_sensor_latency_if_needed(self, sensor_name):
        resampling_time = getattr(getattr(self.cfg.sensor, sensor_name), "latency_resampling_time", self.dt)
        resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
        if len(resample_env_ids) > 0:
            getattr(self, sensor_name + "_latency_buffer")[resample_env_ids] = torch_rand_float(
                getattr(getattr(self.cfg.sensor, sensor_name), "latency_range")[0],
                getattr(getattr(self.cfg.sensor, sensor_name), "latency_range")[1],
                (len(resample_env_ids), 1),
                device= self.device,
            ).flatten()
        
    def _resample_action_delay(self, env_ids):
        self.action_delay_buffer[env_ids] = torch_rand_float(
            self.cfg.control.action_delay_range[0],
            self.cfg.control.action_delay_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "actions_history_buffer"):
            self.actions_history_buffer[:, env_ids] = 0.
            self.action_delayed_frames[env_ids] = self.cfg.control.action_history_buffer_length
        for sensor_name in self.available_sensors:
            if not hasattr(self.cfg.sensor, sensor_name):
                continue
            for component in getattr(self.cfg.sensor, sensor_name).obs_components:
                if hasattr(self, component + "_obs_buffer"):
                    getattr(self, component + "_obs_buffer")[:, env_ids] = 0.
                    setattr(self, component + "_obs_refreshed", False)
            if "camera" in sensor_name:
                getattr(self, sensor_name + "_delayed_frames")[env_ids] = 0
        return return_

    def _build_forward_depth_intrinsic(self):
        sim_raw_resolution = self.cfg.sensor.forward_camera.resolution
        sim_cropping_h = self.cfg.sensor.forward_camera.crop_top_bottom
        sim_cropping_w = self.cfg.sensor.forward_camera.crop_left_right
        cropped_resolution = [ # (H, W)
                            sim_raw_resolution[0] - sum(sim_cropping_h),
                            sim_raw_resolution[1] - sum(sim_cropping_w),
                        ]
        network_input_resolution = self.cfg.sensor.forward_camera.output_resolution
        x_fov = torch.mean(torch.tensor(self.cfg.sensor.forward_camera.horizontal_fov).float()) \
            / 180 * np.pi
        fx = (sim_raw_resolution[1]) / (2 * torch.tan(x_fov / 2))
        fy = fx # * (sim_raw_resolution[0] / sim_raw_resolution[1])
        fx = fx * network_input_resolution[1] / cropped_resolution[1]
        fy = fy * network_input_resolution[0] / cropped_resolution[0]
        cx = (sim_raw_resolution[1] / 2) - sim_cropping_w[0]
        cy = (sim_raw_resolution[0] / 2) - sim_cropping_h[0]
        cx = cx * network_input_resolution[1] / cropped_resolution[1]
        cy = cy * network_input_resolution[0] / cropped_resolution[0]
        self.forward_depth_intrinsic = torch.tensor([
                            [fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.],
                        ], device= self.device)
        x_arr = torch.linspace(
            0,
            network_input_resolution[1] - 1,
            network_input_resolution[1],
        )
        y_arr = torch.linspace(
            0,
            network_input_resolution[0] - 1,
            network_input_resolution[0],
        )
        x_mesh, y_mesh = torch.meshgrid(x_arr, y_arr, indexing= "xy")
        # (H, W, 2) -> (H * W, 3) row wise
        self.forward_depth_pixel_mesh = torch.stack([
            x_mesh,
            y_mesh,
            torch.ones_like(x_mesh),
        ], dim= -1).view(-1, 3).float().to(self.device)
    
    def _draw_pointcloud_from_depth_image(self,
            env_h, sensor_h,
            camera_intrinsic,
            depth_image, # torch tensor (H, W)
            offset = [-2, 0, 1],
        ):
        """
        Args:
            offset: drawing points directly based on the sensor will interfere with the
                depth image rendering. so we need to provide the offset to the pointcloud
        """
        assert self.num_envs == 1, "LeggedRobotNoisy: Only implemented when num_envs == 1"
        camera_transform = self.gym.get_camera_transform(self.sim, env_h, sensor_h)
        camera_intrinsic_inv = torch.inverse(camera_intrinsic)

        # (H * W, 3) -> (3, H * W) -> (H * W, 3)
        depth_image = depth_image * (self.cfg.sensor.forward_camera.depth_range[1] - self.cfg.sensor.forward_camera.depth_range[0]) + self.cfg.sensor.forward_camera.depth_range[0]
        points = camera_intrinsic_inv @ self.forward_depth_pixel_mesh.T * depth_image.view(-1)
        points = points.T

        sphere_geom = gymutil.WireframeSphereGeometry(0.008, 8, 8, None, color= (0.9, 0.1, 0.9))
        for p in points:
            sphere_pose = gymapi.Transform(
                p= camera_transform.transform_point(gymapi.Vec3(
                    p[2] + offset[0],
                    -p[0] + offset[1],
                    -p[1] + offset[2],
                )), # +z forward to +x forward
                r= None,
            )
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env_h, sphere_pose)

    def _draw_sensor_reading_vis(self, env_h, sensor_hd):
        return_ = super()._draw_sensor_reading_vis(env_h, sensor_hd)
        if hasattr(self, "forward_depth_output"):
            if self.num_envs == 1:
                if getattr(self.cfg.viewer, "forward_depth_as_pointcloud", False):
                    if not hasattr(self, "forward_depth_intrinsic"):
                        self._build_forward_depth_intrinsic()
                    for sensor_name, sensor_h in sensor_hd.items():
                        if "forward_camera" in sensor_name:
                            self._draw_pointcloud_from_depth_image(
                                env_h, sensor_h,
                                self.forward_depth_intrinsic,
                                self.forward_depth_output[0, 0].detach(),
                            )
                else:
                    import matplotlib.pyplot as plt
                    forward_depth_np = self.forward_depth_output[0, 0].detach().cpu().numpy() # (H, W)
                    plt.imshow(forward_depth_np, cmap= "gray", vmin= 0, vmax= 1)
                    plt.pause(0.001)
            else:
                print("LeggedRobotNoisy: More than one robot, stop showing camera image")
        return return_

    """ ########## Steps to simulate stereo camera depth image ########## """
    def _add_depth_contour(self, depth_images):
        mask =  F.max_pool2d(
            torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding= 1)).max(dim= -3, keepdim= True)[0],
            kernel_size= self.cfg.noise.forward_depth.contour_detection_kernel_size,
            stride= 1,
            padding= int(self.cfg.noise.forward_depth.contour_detection_kernel_size / 2),
        ) > self.cfg.noise.forward_depth.contour_threshold
        depth_images[mask] = 0.
        return depth_images

    @torch.no_grad()
    def form_artifacts(self,
            H, W, # image resolution
            tops, bottoms, # artifacts positions (in pixel) shape (n_,)
            lefts, rights,
        ):
        """ Paste an artifact to the depth image.
        NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
        entire depth image.
        """
        batch_size = tops.shape[0]
        tops, bottoms = tops[:, None, None], bottoms[:, None, None]
        lefts, rights = lefts[:, None, None], rights[:, None, None]

        # build the source patch
        source_patch = torch.zeros((batch_size, 1, 25, 25), device= self.device)
        source_patch[:, :, 1:24, 1:24] = 1.

        # build the grid
        grid = torch.zeros((batch_size, H, W, 2), device= self.device)
        grid[..., 0] = torch.linspace(-1, 1, W, device= self.device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device= self.device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

        # sample using the grid and form the artifacts for the entire depth image
        artifacts = torch.clip(
            F.grid_sample(
                source_patch,
                grid,
                mode= "bilinear",
                padding_mode= "zeros",
                align_corners= False,
            ).sum(dim= 0).view(H, W),
            0, 1,
        )

        return artifacts

    def _add_depth_artifacts(self, depth_images,
            artifacts_prob,
            artifacts_height_mean_std,
            artifacts_width_mean_std,
        ):
        """ Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
        should be an artifacts, the mask is 1.
        """
        N, _, H, W = depth_images.shape
        def _clip(x, dim):
            return torch.clip(x, 0., (H, W)[dim])

        # random patched artifacts
        artifacts_mask = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, H, W) < artifacts_prob
        artifacts_mask = artifacts_mask & (depth_images[:, 0] > 0.)
        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32) # (n_, 3) n_ <= N * H * W
        artifcats_size = (
            torch.clip(
                artifacts_height_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_height_mean_std[1],
                0., H,
            ),
            torch.clip(
                artifacts_width_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_width_mean_std[1],
                0., W,
            ),
        ) # (n_,), (n_,)
        artifacts_top_left = (
            _clip(artifacts_coord[:, 1] - artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] - artifcats_size[1] / 2, 1),
        )
        artifacts_bottom_right = (
            _clip(artifacts_coord[:, 1] + artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] + artifcats_size[1] / 2, 1),
        )
        for i in range(N):
            # NOTE: make sure the artifacts points are as few as possible
            artifacts_mask = self.form_artifacts(
                H, W,
                artifacts_top_left[0][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[0][artifacts_coord[:, 0] == i],
                artifacts_top_left[1][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[1][artifacts_coord[:, 0] == i],
            )
            depth_images[i] *= (1 - artifacts_mask)

        return depth_images
    
    def _recognize_top_down_too_close(self, too_close_mask):
        """ Based on real D435i image pattern, there are two situations when pixels are too close
        Whether there is too-close pixels all the way across the image vertically.
        """
        # vertical_all_too_close = too_close_mask.all(dim= 2, keepdim= True)
        vertical_too_close = too_close_mask.sum(dim= -2, keepdim= True) > (too_close_mask.shape[-2] * 0.6)
        return vertical_too_close
    
    def _add_depth_stereo(self, depth_images):
        """ Simulate the noise from the depth limit of the stereo camera. """
        N, _, H, W = depth_images.shape
        far_mask = depth_images > self.cfg.noise.forward_depth.stereo_far_distance
        too_close_mask = depth_images < self.cfg.noise.forward_depth.stereo_min_distance
        near_mask = (~far_mask) & (~too_close_mask)

        # add noise to the far points
        far_noise = torch_rand_float(
            0., self.cfg.noise.forward_depth.stereo_far_noise_std,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W)
        far_noise = far_noise * far_mask
        depth_images += far_noise

        # add noise to the near points
        near_noise = torch_rand_float(
            0., self.cfg.noise.forward_depth.stereo_near_noise_std,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W)
        near_noise = near_noise * near_mask
        depth_images += near_noise

        # add artifacts to the too close points
        vertical_block_mask = self._recognize_top_down_too_close(too_close_mask)
        full_block_mask = vertical_block_mask & too_close_mask
        half_block_mask = (~vertical_block_mask) & too_close_mask
        # add artifacts where vertical pixels are all too close
        for pixel_value in random.sample(
                self.cfg.noise.forward_depth.stereo_full_block_values,
                len(self.cfg.noise.forward_depth.stereo_full_block_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.noise.forward_depth.stereo_full_block_artifacts_prob,
                self.cfg.noise.forward_depth.stereo_full_block_height_mean_std,
                self.cfg.noise.forward_depth.stereo_full_block_width_mean_std,
            )
            depth_images[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]
        # add artifacts where not all the same vertical pixels are too close
        half_block_spark = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W) < self.cfg.noise.forward_depth.stereo_half_block_spark_prob
        depth_images[half_block_mask] = (half_block_spark.to(torch.float32) * self.cfg.noise.forward_depth.stereo_half_block_value)[half_block_mask]

        return depth_images
    
    def _recognize_top_down_seeing_sky(self, too_far_mask):
        N, _, H, W = too_far_mask.shape
        # whether there is too-far pixels with all pixels above it too-far
        num_too_far_above = too_far_mask.cumsum(dim= -2)
        all_too_far_above_threshold = torch.arange(H, device= self.device).view(1, 1, H, 1)
        all_too_far_above = num_too_far_above > all_too_far_above_threshold # (N, 1, H, W) mask
        return all_too_far_above
    
    def _add_sky_artifacts(self, depth_images):
        """ Incase something like ceiling pattern or stereo failure happens. """
        N, _, H, W = depth_images.shape
        
        possible_to_sky_mask = depth_images > self.cfg.noise.forward_depth.sky_artifacts_far_distance
        to_sky_mask = self._recognize_top_down_seeing_sky(possible_to_sky_mask)
        isinf_mask = depth_images.isinf()
        
        # add artifacts to the regions where they are seemingly pointing to sky
        for pixel_value in random.sample(
                self.cfg.noise.forward_depth.sky_artifacts_values,
                len(self.cfg.noise.forward_depth.sky_artifacts_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.noise.forward_depth.sky_artifacts_prob,
                self.cfg.noise.forward_depth.sky_artifacts_height_mean_std,
                self.cfg.noise.forward_depth.sky_artifacts_width_mean_std,
            )
            depth_images[to_sky_mask & (~isinf_mask)] *= artifacts_buffer[to_sky_mask & (~isinf_mask)]
            depth_images[to_sky_mask & isinf_mask & (artifacts_buffer < 1)] = 0.
            depth_images[to_sky_mask] += ((1 - artifacts_buffer) * pixel_value)[to_sky_mask]
            pass
        
        return depth_images

    def _crop_depth_images(self, depth_images):
        H, W = depth_images.shape[-2:]
        return depth_images[...,
            self.cfg.sensor.forward_camera.crop_top_bottom[0]: H - self.cfg.sensor.forward_camera.crop_top_bottom[1],
            self.cfg.sensor.forward_camera.crop_left_right[0]: W - self.cfg.sensor.forward_camera.crop_left_right[1],
        ]

    def _normalize_depth_images(self, depth_images):
        depth_images = torch.clip(
            depth_images,
            self.cfg.sensor.forward_camera.depth_range[0],
            self.cfg.sensor.forward_camera.depth_range[1],
        )
        # normalize depth image to (0, 1)
        depth_images = (depth_images - self.cfg.sensor.forward_camera.depth_range[0]) / (
            self.cfg.sensor.forward_camera.depth_range[1] - self.cfg.sensor.forward_camera.depth_range[0]
        )
        return depth_images
    
    @torch.no_grad()
    def _process_depth_image(self, depth_images):
        # depth_images length N list with shape (H, W)
        # reverse the negative depth (according to the document)
        depth_images_ = torch.stack(depth_images).unsqueeze(1).contiguous().detach().clone() * -1
        if hasattr(self.cfg.noise, "forward_depth"):
            if getattr(self.cfg.noise.forward_depth, "countour_threshold", 0.) > 0.:
                depth_images_ = self._add_depth_contour(depth_images_)
            if getattr(self.cfg.noise.forward_depth, "artifacts_prob", 0.) > 0.:
                depth_images_ = self._add_depth_artifacts(depth_images_,
                    self.cfg.noise.forward_depth.artifacts_prob,
                    self.cfg.noise.forward_depth.artifacts_height_mean_std,
                    self.cfg.noise.forward_depth.artifacts_width_mean_std,
                )
            if getattr(self.cfg.noise.forward_depth, "stereo_min_distance", 0.) > 0.:
                depth_images_ = self._add_depth_stereo(depth_images_)
            if getattr(self.cfg.noise.forward_depth, "sky_artifacts_prob", 0.) > 0.:
                depth_images_ = self._add_sky_artifacts(depth_images_)
        # if self.num_envs == 1:
        #     import matplotlib.pyplot as plt
        #     plt.cla()
        #     __depth_images = depth_images_[0, 0].detach().cpu().numpy() # (H, W)
        #     plt.imshow(__depth_images, cmap= "gray",
        #         vmin= self.cfg.sensor.forward_camera.depth_range[0],
        #         vmax= self.cfg.sensor.forward_camera.depth_range[1],
        #     )
        #     plt.draw()
        #     plt.pause(0.001)
        depth_images_ = self._normalize_depth_images(depth_images_)
        depth_images_ = self._crop_depth_images(depth_images_)
        if hasattr(self, "forward_depth_resize_transform"):
            depth_images_ = self.forward_depth_resize_transform(depth_images_)
        depth_images_ = depth_images_.clip(0, 1)
        return depth_images_.unsqueeze(0) # (1, N, 1, H, W)
    
    """ ########## Override the observation functions to add latencies and artifacts ########## """
    def set_buffers_refreshed_to_false(self):
        for sensor_name in self.available_sensors:
            if hasattr(self.cfg.sensor, sensor_name):
                for component in getattr(self.cfg.sensor, sensor_name).obs_components:
                    setattr(self, component + "_obs_refreshed", False)
    
    @torch.no_grad()
    def _get_ang_vel_obs(self, privileged= False):
        if hasattr(self, "ang_vel_obs_buffer") and (not self.ang_vel_obs_refreshed) and (not privileged):
            self.ang_vel_obs_buffer = torch.cat([
                self.ang_vel_obs_buffer[1:],
                super()._get_ang_vel_obs().unsqueeze(0),
            ], dim= 0)
            component_governed_by = self.component_governed_by_sensor["ang_vel"]
            buffer_delayed_frames = ((getattr(self, component_governed_by + "_latency_buffer") / self.dt) + 1).to(int)
            self.ang_vel_obs_output_buffer = self.ang_vel_obs_buffer[
                -buffer_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.ang_vel_obs_refreshed = True
        if privileged or not hasattr(self, "ang_vel_obs_buffer"):
            return super()._get_ang_vel_obs(privileged)
        return self.ang_vel_obs_output_buffer
    
    @torch.no_grad()
    def _get_projected_gravity_obs(self, privileged= False):
        if hasattr(self, "projected_gravity_obs_buffer") and (not self.projected_gravity_obs_refreshed) and (not privileged):
            self.projected_gravity_obs_buffer = torch.cat([
                self.projected_gravity_obs_buffer[1:],
                super()._get_projected_gravity_obs().unsqueeze(0),
            ], dim= 0)
            component_governed_by = self.component_governed_by_sensor["projected_gravity"]
            buffer_delayed_frames = ((getattr(self, component_governed_by + "_latency_buffer") / self.dt) + 1).to(int)
            self.projected_gravity_obs_output_buffer = self.projected_gravity_obs_buffer[
                -buffer_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.projected_gravity_obs_refreshed = True
        if privileged or not hasattr(self, "projected_gravity_obs_buffer"):
            return super()._get_projected_gravity_obs(privileged)
        return self.projected_gravity_obs_output_buffer
    
    @torch.no_grad()
    def _get_commands_obs(self, privileged= False):
        if hasattr(self, "commands_obs_buffer") and (not self.commands_obs_refreshed) and (not privileged):
            self.commands_obs_buffer = torch.cat([
                self.commands_obs_buffer[1:],
                super()._get_commands_obs().unsqueeze(0),
            ], dim= 0)
            component_governed_by = self.component_governed_by_sensor["commands"]
            buffer_delayed_frames = ((getattr(self, component_governed_by + "_latency_buffer") / self.dt) + 1).to(int)
            self.commands_obs_output_buffer = self.commands_obs_buffer[
                -buffer_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.commands_obs_refreshed = True
        if privileged or not hasattr(self, "commands_obs_buffer"):
            return super()._get_commands_obs(privileged)
        return self.commands_obs_output_buffer
    
    @torch.no_grad()
    def _get_dof_pos_obs(self, privileged= False):
        if hasattr(self, "dof_pos_obs_buffer") and (not self.dof_pos_obs_refreshed) and (not privileged):
            self.dof_pos_obs_buffer = torch.cat([
                self.dof_pos_obs_buffer[1:],
                super()._get_dof_pos_obs().unsqueeze(0),
            ], dim= 0)
            component_governed_by = self.component_governed_by_sensor["dof_pos"]
            buffer_delayed_frames = ((getattr(self, component_governed_by + "_latency_buffer") / self.dt) + 1).to(int)
            self.dof_pos_obs_output_buffer = self.dof_pos_obs_buffer[
                -buffer_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.dof_pos_obs_refreshed = True
        if privileged or not hasattr(self, "dof_pos_obs_buffer"):
            return super()._get_dof_pos_obs(privileged)
        return self.dof_pos_obs_output_buffer
    
    @torch.no_grad()
    def _get_dof_vel_obs(self, privileged= False):
        if hasattr(self, "dof_vel_obs_buffer") and (not self.dof_vel_obs_refreshed) and (not privileged):
            self.dof_vel_obs_buffer = torch.cat([
                self.dof_vel_obs_buffer[1:],
                super()._get_dof_vel_obs().unsqueeze(0),
            ], dim= 0)
            component_governed_by = self.component_governed_by_sensor["dof_vel"]
            buffer_delayed_frames = ((getattr(self, component_governed_by + "_latency_buffer") / self.dt) + 1).to(int)
            self.dof_vel_obs_output_buffer = self.dof_vel_obs_buffer[
                -buffer_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.dof_vel_obs_refreshed = True
        if privileged or not hasattr(self, "dof_vel_obs_buffer"):
            return super()._get_dof_vel_obs(privileged)
        return self.dof_vel_obs_output_buffer

    @torch.no_grad()
    def _get_forward_depth_obs(self, privileged= False):
        if hasattr(self, "forward_depth_obs_buffer") and (not self.forward_depth_obs_refreshed) and hasattr(self.cfg.sensor, "forward_camera") and (not privileged):
            # TODO: any variables named with "forward_camera" here should be rearanged 
            # in a individual class method.
            self.forward_depth_obs_buffer = torch.cat([
                self.forward_depth_obs_buffer[1:],
                self._process_depth_image(self.sensor_tensor_dict["forward_depth"]),
            ], dim= 0)
            delay_refresh_mask = (self.episode_length_buf % int(self.cfg.sensor.forward_camera.refresh_duration / self.dt)) == 0
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used.
            frame_select = (self.forward_camera_latency_buffer / self.dt).to(int)
            self.forward_camera_delayed_frames = torch.where(
                delay_refresh_mask,
                torch.minimum(
                    frame_select,
                    self.forward_camera_delayed_frames + 1,
                ),
                self.forward_camera_delayed_frames + 1,
            )
            self.forward_camera_delayed_frames = torch.clip(
                self.forward_camera_delayed_frames,
                0,
                self.forward_depth_obs_buffer.shape[0],
            )
            self.forward_depth_output = self.forward_depth_obs_buffer[
                -self.forward_camera_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.forward_depth_refreshed = True
        if not hasattr(self.cfg.sensor, "forward_camera") or privileged:
            return super()._get_forward_depth_obs(privileged).reshape(self.num_envs, -1)

        return self.forward_depth_output.flatten(start_dim= 1)

    def get_obs_segment_from_components(self, obs_components):
        obs_segments = super().get_obs_segment_from_components(obs_components)
        if "forward_depth" in obs_components:
            obs_segments["forward_depth"] = (1, *getattr(
                self.cfg.sensor.forward_camera,
                "output_resolution",
                self.cfg.sensor.forward_camera.resolution,
            ))
        return obs_segments
