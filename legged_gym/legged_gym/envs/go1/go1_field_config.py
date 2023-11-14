import numpy as np
import os
from os import path as osp
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO

go1_const_dof_range = dict(
    Hip_max= 1.047,
    Hip_min=-1.047,
    Thigh_max= 2.966,
    Thigh_min= -0.663,
    Calf_max= -0.837,
    Calf_min= -2.721,
)

go1_action_scale = 0.5

class Go1FieldCfg( A1FieldCfg ):
    class env( A1FieldCfg.env ):
        num_envs = 8192

    class init_state( A1FieldCfg.init_state ):
        pos = [0., 0., 0.7]
        zero_actions = False

    class sensor( A1FieldCfg.sensor ):
        class proprioception( A1FieldCfg.sensor.proprioception ):
            delay_action_obs = False
            latency_range = [0.04-0.0025, 0.04+0.0075] # comment this if it is too hard to train.

    class terrain( A1FieldCfg.terrain ):
        num_rows = 20
        num_cols = 80

        # curriculum = True # for tilt, crawl, jump, leap
        curriculum = False # for walk
        pad_unavailable_info = True

        BarrierTrack_kwargs = merge_dict(A1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                # "jump",
                # "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            # randomize_obstacle_order= True,
            track_width= 1.6,
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= 0.0,
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        ))

        TerrainPerlin_kwargs = dict(
            zScale= [0.08, 0.15],
            frequency= 10,
        )

    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]


    class control( A1FieldCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 0.5}
        action_scale = go1_action_scale
        torque_limits = [20., 20., 25.] * 4
        computer_clip_torque = False
        motor_clip_torque = False

    class asset( A1FieldCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        sdk_dof_range = go1_const_dof_range

    class termination( A1FieldCfg.termination ):
        roll_kwargs = dict(
            threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.5,
        )

    class domain_rand( A1FieldCfg.domain_rand ):
        class com_range( A1FieldCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]

        init_base_pos_range = merge_dict(A1FieldCfg.domain_rand.init_base_pos_range, dict(
            x= [0.05, 0.6],
        ))
        init_base_rot_range = dict(
            roll= [-0.75, 0.75],
            pitch= [-0.75, 0.75],
        )
        # init_base_vel_range = [-1.0, 1.0]
        init_base_vel_range = dict(
            x= [-0.2, 1.5],
            y= [-0.2, 0.2],
            z= [-0.2, 0.2],
            roll= [-1., 1.],
            pitch= [-1., 1.],
            yaw= [-1., 1.],
        )
        init_dof_vel_range = [-5, 5]

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            tracking_world_vel = 3.
            # world_vel_l2norm = -2.
            # alive = 3.
            legs_energy_substeps = -2e-5
            # penalty for hardware safety
            exceed_dof_pos_limits = -8e-1
            exceed_torque_limits_l1norm = -8e-1
            # penalty for walking gait, probably no need
            lin_vel_z = -1.
            ang_vel_xy = -0.05
            orientation = -4.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1e-7
            torques = -1.e-5
            yaw_abs = -0.8
            lin_pos_y = -0.8
            hip_pos = -0.4
            dof_error = -0.04
        soft_dof_pos_limit = 0.8 # only in training walking
        max_contact_force = 200.0
        
    class normalization( A1FieldCfg.normalization ):
        dof_pos_redundancy = 0.2
        clip_actions_method = "hard"
        clip_actions_low = []
        clip_actions_high = []
        for sdk_joint_name, sim_joint_name in zip(
            ["Hip", "Thigh", "Calf"] * 4,
            [ # in the order as simulation
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
        ):
            clip_actions_low.append( (go1_const_dof_range[sdk_joint_name + "_min"] + dof_pos_redundancy - A1FieldCfg.init_state.default_joint_angles[sim_joint_name]) / go1_action_scale )
            clip_actions_high.append( (go1_const_dof_range[sdk_joint_name + "_max"] - dof_pos_redundancy - A1FieldCfg.init_state.default_joint_angles[sim_joint_name]) / go1_action_scale )
        del dof_pos_redundancy, sdk_joint_name, sim_joint_name

    class sim( A1FieldCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1FieldCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2

    class policy( A1FieldCfgPPO.policy ):
        mu_activation = None # use action clip method by env

    class runner( A1FieldCfgPPO.runner ):
        experiment_name = "field_go1"
        resume = False
        load_run = None
        
        run_name = "".join(["WalkForward",
        ("_pEnergySubsteps" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.legs_energy_substeps, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "legs_energy_substeps", 0.0) != 0.0 else ""),
        ("_rTrackVel" + np.format_float_scientific(Go1FieldCfg.rewards.scales.tracking_world_vel, precision=1, exp_digits=1, trim="-") if getattr(Go1FieldCfg.rewards.scales, "tracking_world_vel", 0.0) != 0.0 else ""),
        ("_pWorldVel" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.world_vel_l2norm, precision=1, exp_digits=1, trim="-") if getattr(Go1FieldCfg.rewards.scales, "world_vel_l2norm", 0.0) != 0.0 else ""),
        ("_aScale{:d}{:d}{:d}".format(
                int(Go1FieldCfg.control.action_scale[0] * 10),
                int(Go1FieldCfg.control.action_scale[1] * 10),
                int(Go1FieldCfg.control.action_scale[2] * 10),
            ) if isinstance(Go1FieldCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(Go1FieldCfg.control.action_scale)
        ),
        ("_actionClip" + Go1FieldCfg.normalization.clip_actions_method if getattr(Go1FieldCfg.normalization, "clip_actions_method", None) is not None else ""),
        ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2]) if resume else "_noResume"),
        ])
        max_iterations = 20000
        save_interval = 500
