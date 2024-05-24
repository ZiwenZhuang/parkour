import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    # MotorState,
    # IMUState,
    LowCmd,
    # MotorCmd,
)
from std_msgs.msg import Float32MultiArray
if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "x86",
    ))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "aarch64",
    ))
from crc_module import get_crc

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

@torch.jit.script
def quat_rotate_inverse(q, v):
    """ q must be in x, y, z, w order """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

class RobotCfgs:
    class H1:
        pass

    class Go2:
        NUM_DOF = 12
        NUM_ACTIONS = 12
        dof_map = [ # from isaacgym simulation joint order to real robot joint order
            3, 4, 5,
            0, 1, 2,
            9, 10, 11,
            6, 7, 8,
        ]
        dof_names = [ # NOTE: order matters. This list is the order in simulation.
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        dof_signs = [1.] * 12
        joint_limits_high = torch.tensor([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ], device= "cpu", dtype= torch.float32)
        joint_limits_low = torch.tensor([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ], device= "cpu", dtype= torch.float32)
        torque_limits = torch.tensor([ # from urdf and in simulation order
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
        ], device= "cpu", dtype= torch.float32)
        turn_on_motor_mode = [0x01] * 12
        

class UnitreeRos2Real(Node):
    """ A proxy implementation of the real H1 robot. """
    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self,
            robot_namespace= None,
            low_state_topic= "/lowstate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            forward_depth_topic= None, # if None and still need access, set to str "pyrealsense"
            forward_depth_embedding_topic= "/forward_depth_embedding",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            replace_obs_with_embeddings= [], # a list of strings, e.g. ["forward_depth"] then the corrseponding obs will be processed by _get_forward_depth_embedding_obs()
            move_by_wireless_remote= True, # if True, the robot will be controlled by a wireless remote
            model_device= "cpu",
            dof_pos_protect_ratio= 1.1, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "H1",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        # Generate a unique cmd topic so that the low_cmd will not send to the robot's motor.
        self.low_cmd_topic = low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic
        self.forward_depth_topic = forward_depth_topic
        self.forward_depth_embedding_topic = forward_depth_embedding_topic
        self.cfg = cfg
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range
        self.replace_obs_with_embeddings = replace_obs_with_embeddings
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        self.dof_map = getattr(RobotCfgs, robot_class_name).dof_map
        self.dof_names = getattr(RobotCfgs, robot_class_name).dof_names
        self.dof_signs = getattr(RobotCfgs, robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, robot_class_name).turn_on_motor_mode

        self.parse_config()

    def parse_config(self):
        """ parse, set attributes from config dict, initialize buffers to speed up the computation """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.model_device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1
        
        # observations
        self.clip_obs = self.cfg["normalization"]["clip_observations"]
        self.obs_scales = self.cfg["normalization"]["obs_scales"]
        for k, v in self.obs_scales.items():
            if isinstance(v, (list, tuple)):
                self.obs_scales[k] = torch.tensor(v, device= self.model_device, dtype= torch.float32)
        # check whether there are embeddings in obs_components and launch encoder process later
        if len(self.replace_obs_with_embeddings) > 0:
            for comp in self.replace_obs_with_embeddings:
                self.get_logger().warn(f"{comp} will be replaced with its embedding when get_obs, don't forget to launch the corresponding process before running the policy.")
        self.obs_segments = self.get_obs_segment_from_components(self.cfg["env"]["obs_components"])
        self.num_obs = self.get_num_obs_from_components(self.cfg["env"]["obs_components"])
        if "privileged_obs_components" in self.cfg["env"].keys():
            self.privileged_obs_segments = self.get_obs_segment_from_components(self.cfg["env"]["privileged_obs_components"])
            self.num_privileged_obs = self.get_num_obs_from_components(self.cfg["env"]["privileged_obs_components"])
        for obs_component in self.cfg["env"]["obs_components"]:
            if "orientation_cmds" in obs_component:
                self.roll_pitch_yaw_cmd = torch.zeros(1, 3, device= self.model_device, dtype= torch.float32)
        
        # controls
        self.control_type = self.cfg["control"]["control_type"]
        if not (self.control_type == "P"):
            raise NotImplementedError("Only position control is supported for now.")
        self.p_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] # set p_gains in simulation order
            for k, v in self.cfg["control"]["stiffness"].items():
                if k in name:
                    self.p_gains.append(v)
                    break # only one match
        self.p_gains = torch.tensor(self.p_gains, device= self.model_device, dtype= torch.float32)
        self.d_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] # set d_gains in simulation order
            for k, v in self.cfg["control"]["damping"].items():
                if k in name:
                    self.d_gains.append(v)
                    break
        self.d_gains = torch.tensor(self.d_gains, device= self.model_device, dtype= torch.float32)
        self.default_dof_pos = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            # in simulation order.
            self.default_dof_pos[i] = default_joint_angle
        self.computer_clip_torque = self.cfg["control"].get("computer_clip_torque", True)
        self.get_logger().info("Computer Clip Torque (onboard) is " + str(self.computer_clip_torque))
        self.torque_limits = getattr(RobotCfgs, self.robot_class_name).torque_limits.to(self.model_device)
        if self.computer_clip_torque:
            assert hasattr(self, "torque_limits") and (len(self.torque_limits) == self.NUM_DOF), f"torque_limits must be set with the length of {self.NUM_DOF} if computer_clip_torque is True"
            self.get_logger().info("[Env] torque limit: " + ",".join("{:.1f}".format(x) for x in self.torque_limits))
        
        # actions
        self.num_actions = self.NUM_ACTIONS
        self.action_scale = self.cfg["control"]["action_scale"]
        self.get_logger().info("[Env] action scale: {:.1f}".format(self.action_scale))
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"].get("clip_actions_method", None) == "hard":
            self.get_logger().info("clip_actions_method with hard mode")
            self.get_logger().info("clip_actions_high: " + str(self.cfg["normalization"]["clip_actions_high"]))
            self.get_logger().info("clip_actions_low: " + str(self.cfg["normalization"]["clip_actions_low"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = torch.tensor(self.cfg["normalization"]["clip_actions_low"], device= self.model_device, dtype= torch.float32)
            self.clip_actions_high = torch.tensor(self.cfg["normalization"]["clip_actions_high"], device= self.model_device, dtype= torch.float32)
        else:
            self.get_logger().info("clip_actions_method is " + str(self.cfg["normalization"].get("clip_actions_method", None)))
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)
            

        # hardware related, in simulation order
        self.joint_limits_high = getattr(RobotCfgs, self.robot_class_name).joint_limits_high.to(self.model_device)
        self.joint_limits_low = getattr(RobotCfgs, self.robot_class_name).joint_limits_low.to(self.model_device)
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio

    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """

        # ROS publishers
        self.low_cmd_pub = self.create_publisher(
            LowCmd,
            self.low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()

        # ROS subscribers
        self.low_state_sub = self.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )

        if self.forward_depth_topic is not None:
            self.forward_camera_sub = self.create_subscription(
                Image,
                self.forward_depth_topic,
                self._forward_depth_callback,
                1
            )

        if self.forward_depth_embedding_topic is not None and "forward_depth" in self.replace_obs_with_embeddings:
            self.forward_depth_embedding_sub = self.create_subscription(
                Float32MultiArray,
                self.forward_depth_embedding_topic,
                self._forward_depth_embedding_callback,
                1,
            )

        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state message received, the robot is ready to go.")

    """ ROS callbacks and handlers that update the buffer """
    def _low_state_callback(self, msg):
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state

        # refresh dof_pos and dof_vel
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]
        # automatic safety check
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if self.dof_pos_[0, sim_idx] > self.joint_pos_protect_high[sim_idx] or \
                self.dof_pos_[0, sim_idx] < self.joint_pos_protect_low[sim_idx]:
                self.get_logger().error(f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}")
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()

    def _joy_stick_callback(self, msg):
        self.joy_stick_buffer = msg
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for turning left/right
            lx = -msg.lx
            if lx > self.ang_vel_deadband:
                yaw = (lx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif lx < -self.ang_vel_deadband:
                yaw = (lx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            # right-x for side moving left/right
            rx = -msg.rx
            if rx > self.lin_vel_deadband:
                vy = (rx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif rx < -self.lin_vel_deadband:
                vy = (rx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            self.xyyaw_command = torch.tensor([vx, vy, yaw], device= self.model_device, dtype= torch.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
            self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
            self._turn_off_motors()
            raise SystemExit()

        # roll-pitch target
        if hasattr(self, "roll_pitch_yaw_cmd"):
            if (msg.keys & self.WirelessButtons.up):
                self.roll_pitch_yaw_cmd[0, 1] += 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.down):
                self.roll_pitch_yaw_cmd[0, 1] -= 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.left):
                self.roll_pitch_yaw_cmd[0, 0] -= 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.right):
                self.roll_pitch_yaw_cmd[0, 0] += 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))

    def _forward_depth_callback(self, msg):
        """ store and handle depth camera data """
        pass

    def _forward_depth_embedding_callback(self, msg):
        self.forward_depth_embedding_buffer = torch.tensor(msg.data, device= self.model_device, dtype= torch.float32).view(1, -1)

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """
    def _get_lin_vel_obs(self):
        return torch.zeros(1, 3, device= self.model_device, dtype= torch.float32)
    
    def _get_ang_vel_obs(self):
        buffer = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0)
        return buffer

    def _get_projected_gravity_obs(self):
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion[1],
            self.low_state_buffer.imu_state.quaternion[2],
            self.low_state_buffer.imu_state.quaternion[3],
            self.low_state_buffer.imu_state.quaternion[0],
        ], device= self.model_device, dtype= torch.float32).unsqueeze(0)
        return quat_rotate_inverse(
            quat_xyzw,
            self.gravity_vec,
        )

    def _get_commands_obs(self):
        return self.xyyaw_command.unsqueeze(0) # (1, 3)

    def _get_dof_pos_obs(self):
        return self.dof_pos_ - self.default_dof_pos.unsqueeze(0)

    def _get_dof_vel_obs(self):
        return self.dof_vel_

    def _get_last_actions_obs(self):
        return self.actions

    def _get_forward_depth_embedding_obs(self):
        return self.forward_depth_embedding_buffer
    
    def _get_forward_depth_obs(self):
        raise NotImplementedError()
    
    def _get_orientation_cmds_obs(self):
        return quat_rotate_inverse(
            quat_from_euler_xyz(self.roll_pitch_yaw_cmd[:, 0], self.roll_pitch_yaw_cmd[:, 1], self.roll_pitch_yaw_cmd[:, 2]),
            self.gravity_vec,
        )
    
    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
        
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "lin_vel" in components:
            print("Warning: lin_vel is not typically available or accurate enough on the real robot. Will return zeros.")
            segments["lin_vel"] = (3,)
        if "ang_vel" in components:
            segments["ang_vel"] = (3,)
        if "projected_gravity" in components:
            segments["projected_gravity"] = (3,)
        if "commands" in components:
            segments["commands"] = (3,)
        if "dof_pos" in components:
            segments["dof_pos"] = (self.NUM_DOF,)
        if "dof_vel" in components:
            segments["dof_vel"] = (self.NUM_DOF,)
        if "last_actions" in components:
            segments["last_actions"] = (self.NUM_ACTIONS,)
        if "height_measurements" in components:
            print("Warning: height_measurements is not typically available on the real robot.")
            segments["height_measurements"] = (1, len(self.cfg["terrain"]["measured_points_x"]), len(self.cfg["terrain"]["measured_points_y"]))
        if "forward_depth" in components:
            if "output_resolution" in self.cfg["sensor"]["forward_camera"]:
                segments["forward_depth"] = (1, *self.cfg["sensor"]["forward_camera"]["output_resolution"])
            else:
                segments["forward_depth"] = (1, *self.cfg["sensor"]["forward_camera"]["resolution"])
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            print("Warning: height_measurements is not typically available on the real robot.")
            segments["robot_config"] = (1 + 3 + 1 + self.NUM_ACTIONS,)

        """ NOTE: The following components are not directly set in legged_robot.py.
        Please check the order or extend the class implementation if needed.
        """
        if "joints_target" in components:
            # o[0] for target value, 0[1] for wether the target should be tracked (1) or not (0)
            segments["joints_target"] = (2, self.NUM_DOF)
        if "projected_gravity_target" in components:
            # projected_gravity for which the robot should track the target
            # last value as a mask of whether to follow the target or not
            segments["projected_gravity_target"] = (3+1,)
        if "orientation_cmds" in components:
            segments["orientation_cmds"] = (3,)
        return segments
    
    def get_obs(self, obs_segments= None):
        """ Extract from the buffers and build the 1d observation tensor
        Each get ... obs function does not do the obs_scale multiplication.
        NOTE: obs_buffer has the batch dimension, whose size is 1.
        """
        if obs_segments is None:
            obs_segments = self.obs_segments
        obs_buffer = []
        for k, v in obs_segments.items():
            if k in self.replace_obs_with_embeddings:
                obs_component_value = getattr(self, "_get_" + k + "_embedding_obs")()
            else:
                obs_component_value = getattr(self, "_get_" + k + "_obs")() * self.obs_scales.get(k, 1.0)
            obs_buffer.append(obs_component_value)
        obs_buffer = torch.cat(obs_buffer, dim=1)
        obs_buffer = torch.clamp(obs_buffer, -self.clip_obs, self.clip_obs)
        return obs_buffer
    """ Done: refresh observation buffer and corresponding sub-functions """

    """ Control related functions """
    def clip_action_before_scale(self, action):
        action = torch.clip(action, -self.clip_actions, self.clip_actions)
        if getattr(self, "clip_actions_method", None) == "hard":
            action = torch.clip(action, self.clip_actions_low, self.clip_actions_high)
        return action

    def clip_by_torque_limit(self, actions_scaled):
        """ Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        control_type = self.cfg["control"]["control_type"]
        if control_type == "P":
            p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel_
            p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel_
            actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos_
            actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos_
        else:
            raise NotImplementedError

        return torch.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        self.actions = self.clip_action_before_scale(actions)
        if self.computer_clip_torque:
            clipped_scaled_action = self.clip_by_torque_limit(actions * self.action_scale)
        else:
            self.get_logger().warn("Computer Clip Torque is False, the robot may be damaged.", throttle_duration_sec= 1)
            clipped_scaled_action = actions * self.action_scale
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos.unsqueeze(0)

        self._publish_legs_cmd(robot_coordinates_action[0])

    """ Done: Control related functions """

    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains[sim_idx].item()
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """
