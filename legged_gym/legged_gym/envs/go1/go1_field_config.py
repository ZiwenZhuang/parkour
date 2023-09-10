import numpy as np
import os
from os import path as osp
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO

class Go1FieldCfg( A1FieldCfg ):

    class terrain( A1FieldCfg.terrain ):
        
        num_rows = 20
        num_cols = 50
        selected = "BarrierTrack"
        max_init_terrain_level = 0 # for climb, leap finetune
        border_size = 5
        slope_treshold = 100.

        curriculum = True # for tilt, crawl, climb, leap
        # curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options= [
                # "climb",
                # "crawl",
                "tilt",
                # "leap",
            ], # each race track will permute all the options
            track_width= 1.6, # for climb, crawl, tilt, walk
            # track_width= 1.0, # for leap
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= 0.01, # [m] for climb, crawl, tilt, walk
            # wall_height= -0.5, # for leap
            climb= dict(
                height= (0.2, 0.6),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
            ),
            crawl= dict(
                height= (0.28, 0.38),
                depth= (0.1, 0.5), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.25,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0., # for climb, crawl, tilt, walk
            # border_height= -0.5, # for leap
            virtual_terrain= False, # for climb, crawl, leap
            # virtual_terrain= True, # for tilt
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            curriculum_perlin= False,
            no_perlin_threshold= 0.0, # for crawl, tilt, walk
            # no_perlin_threshold= 0.05, # for leap
            # no_perlin_threshold= 0.06, # for climb
        )

        TerrainPerlin_kwargs = dict(
            # zScale= 0.1, # for crawl
            zScale= 0.12, # for tilt
            # zScale= [0.05, 0.1], # for climb
            # zScale= [0.04, 0.1], # for leap
            # zScale= [0.1, 0.15], # for walk
            frequency= 10,
        )

    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            # lin_vel_x = [0.0, 1.0] # for walk
            # lin_vel_x = [0.8, 1.5] # for climb
            # lin_vel_x = [1.0, 1.5] # for leap
            lin_vel_x = [0.3, 0.8] # for tilt, crawl
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]


    class control( A1FieldCfg.control ):
        stiffness = {'joint': 50.}
        damping = {'joint': 1.}
        # action_scale = [0.2, 0.4, 0.4] * 4 # for walk
        action_scale = 0.5 # for tilt, crawl, climb, leap
        # for climb, leap
        torque_limits = [20., 20., 25.] * 4
        computer_clip_torque = True
        motor_clip_torque = False

    class asset( A1FieldCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        penalize_contacts_on = ["base", "thigh"]
        terminate_after_contacts_on = ["base", "imu"] # for climb, leap, tilt, walk no-virtual

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll", # for tilt
            "pitch",
            "z_low",
            "z_high",
            "out_of_track", # for leap, walk
        ]

        roll_kwargs = dict(
            threshold= 0.8, # [rad] # for tilt
            tilt_threshold= 1.5, # for tilt (condition on engaging block)
        )
        pitch_kwargs = dict(
            threshold= 1.6,
            climb_threshold= 1.6,
            leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.08, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = True
        timeout_at_finished = True

    class domain_rand( A1FieldCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        
        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.1
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            exceed_torque_limits_i = -0.1
            alive = 2.
            lin_pos_y = -0.2
            yaw_abs = -0.5
            penetrate_depth = -5e-3
            penetrate_volume = -5e-3
        soft_dof_pos_limit = 0.01

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

    class curriculum:
        # chosen heuristically, please refer to `LeggedRobotField._get_terrain_curriculum_move` with fixed body_measure_points
        # for crawl (not updated)
        penetrate_volume_threshold_harder = 1500
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 10
        penetrate_depth_threshold_easier = 400
        # for tilt
        # penetrate_volume_threshold_harder = 2000
        # penetrate_volume_threshold_easier = 10000
        # penetrate_depth_threshold_harder = 20
        # penetrate_depth_threshold_easier = 300
        # for climb
        # penetrate_volume_threshold_harder = 6000
        # penetrate_volume_threshold_easier = 12000
        # penetrate_depth_threshold_harder = 600
        # penetrate_depth_threshold_easier = 1600
        # for leap
        # penetrate_volume_threshold_harder = 9000
        # penetrate_volume_threshold_easier = 10000
        # penetrate_depth_threshold_harder = 300
        # penetrate_depth_threshold_easier = 5000

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1FieldCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.01 # for walk
        clip_min_std = 1e-12 # for walk

    class runner( A1FieldCfgPPO.runner ):
        experiment_name = "field_go1"
        run_name = "".join(["Skills",
        ("_all" if len(Go1FieldCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else ("_" + Go1FieldCfg.terrain.BarrierTrack_kwargs["options"][0] if Go1FieldCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_pEnergySubsteps" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.legs_energy_substeps, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "legs_energy_substeps", 0.0) != 0.0 else ""),
        ("_pPenV" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.penetrate_volume, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "penetrate_volume", 0.) < 0. else ""),
        ("_pPenD" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.penetrate_depth, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "penetrate_depth", 0.) < 0. else ""),
        ("_rAlive{:d}".format(int(Go1FieldCfg.rewards.scales.alive)) if Go1FieldCfg.rewards.scales.alive != 2 else ""),
        ("_rAngVel{:.2f}".format(Go1FieldCfg.rewards.scales.tracking_ang_vel) if Go1FieldCfg.rewards.scales.tracking_ang_vel != 0.05 else ""),
        ("_pTorqueExceedIndicate" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.exceed_torque_limits_i, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "exceed_torque_limits_i", 0.) != 0. else ""),
        ("_pTorqueExceedSquare" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.exceed_torque_limits_square, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "exceed_torque_limits_square", 0.) != 0. else ""),
        ("_pYaw{:.2f}".format(-Go1FieldCfg.rewards.scales.yaw_abs) if Go1FieldCfg.rewards.scales.yaw_abs != 0. else ""),
        ("_noComputerTorqueClip" if not Go1FieldCfg.control.computer_clip_torque else ""),
        ("_virtualTerrain" if Go1FieldCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_aScale{:d}{:d}{:d}".format(
                int(Go1FieldCfg.control.action_scale[0] * 10),
                int(Go1FieldCfg.control.action_scale[1] * 10),
                int(Go1FieldCfg.control.action_scale[2] * 10),
            ) if isinstance(Go1FieldCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(Go1FieldCfg.control.action_scale)
        ),
        ("_torqueClip{:.0f}".format(Go1FieldCfg.control.torque_limits) if not isinstance(Go1FieldCfg.control.torque_limits, (tuple, list)) else (
            "_tClip{:d}{:d}{:d}".format(
                int(Go1FieldCfg.control.torque_limits[0]),
                int(Go1FieldCfg.control.torque_limits[1]),
                int(Go1FieldCfg.control.torque_limits[2]),
            )
        )),
        ])
        resume = False
        load_run = ""
        max_iterations = 10000
        save_interval = 500
