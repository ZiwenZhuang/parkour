import numpy as np
from os import path as osp
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1CrawlCfg( Go1FieldCfg ):
    class init_state( Go1FieldCfg.init_state ):
        pos = [0., 0., 0.45]

    #### uncomment this to train non-virtual terrain
    class sensor( Go1FieldCfg.sensor ):
        class proprioception( Go1FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( Go1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(Go1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "crawl",
            ],
            track_block_length= 1.6,
            crawl= dict(
                height= (0.28, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(Go1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.06,
        ))

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control( Go1FieldCfg.control ):
        computer_clip_torque = False
    
    class asset( Go1FieldCfg.asset ):
        terminate_after_contacts_on = ["base"]

    class termination( Go1FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]

    class domain_rand( Go1FieldCfg.domain_rand ):
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )
        init_base_vel_range = [-0.1, 0.1]

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -4e-5
            # alive = 5.
            # penetrate_depth = -6e-2
            # penetrate_volume = -6e-2
            # exceed_dof_pos_limits = -8e-1
            # exceed_torque_limits_l1norm = -8e-1
            # lin_pos_y = -0.1
            ###############################################
            tracking_ang_vel = 0.05
            tracking_world_vel = 5.
            # world_vel_l2norm = -1.
            # alive = 2.
            legs_energy_substeps = -1e-5
            # penetrate_depth = -6e-2 # comment this out if trianing non-virtual terrain
            # penetrate_volume = -6e-2 # comment this out if trianing non-virtual terrain
            exceed_dof_pos_limits = -8e-1
            # exceed_torque_limits_i = -2e-1
            exceed_torque_limits_l1norm = -1.
            # collision = -0.05
            # tilt_cond = 0.1
            torques = -1e-5
            yaw_abs = -0.1
            lin_pos_y = -0.1
        soft_dof_pos_limit = 0.7
        only_positive_rewards = False

    class curriculum( Go1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 1500
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 10
        penetrate_depth_threshold_easier = 400

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1CrawlCfgPPO( Go1FieldCfgPPO ):
    class algorithm( Go1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.1
    
    class runner( Go1FieldCfgPPO.runner ):
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "Sep20_03-37-32_SkillopensourcePlaneWalking_pEnergySubsteps1e-5_pTorqueExceedIndicate1e-1_aScale0.5_tClip202025"
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep26_14-30-24_Skills_crawl_propDelay0.04-0.05_pEnergy-2e-5_pDof8e-01_pTorqueL14e-01_rTilt5e-01_pCollision0.2_maxPushAng0.5_kp40_fromSep26_01-38-19")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-19-00_Skills_crawl_propDelay0.04-0.05_pEnergy-1e-5_pDof8e-01_pTorqueL14e-01_pPosY0.1_maxPushAng0.3_kp40_fromOct09_09-58-26")

        run_name = "".join(["Skills_",
        ("Multi" if len(Go1CrawlCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go1CrawlCfg.terrain.BarrierTrack_kwargs["options"][0] if Go1CrawlCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_pEnergy" + np.format_float_scientific(-Go1CrawlCfg.rewards.scales.legs_energy_substeps, precision=1, exp_digits=1)),
        ("_pDof" + np.format_float_scientific(-Go1CrawlCfg.rewards.scales.exceed_dof_pos_limits, precision=1, exp_digits=1)),
        ("_pTorqueL1" + np.format_float_scientific(-Go1CrawlCfg.rewards.scales.exceed_torque_limits_l1norm, precision=1, exp_digits=1)),
        ("_noComputerClip" if not Go1CrawlCfg.control.computer_clip_torque else ""),
        ("_noTanh"),
        ])
        max_iterations = 20000
        save_interval = 500