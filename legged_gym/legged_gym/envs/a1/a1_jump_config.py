from os import path as osp
import numpy as np
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1JumpCfg( A1FieldCfg ):

    class init_state( A1FieldCfg.init_state ):
        pos = [0., 0., 0.45]

    #### uncomment this to train non-virtual terrain
    # class sensor( A1FieldCfg.sensor ):
    #     class proprioception( A1FieldCfg.sensor.proprioception ):
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( A1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(A1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
            ],
            track_block_length= 1.6,
            jump= dict(
                height= (0.2, 0.6), # use this to train in virtual terrain
                # height= (0.1, 0.5), # use this to train in non-virtual terrain
                depth= (0.1, 0.2),
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.12,
            n_obstacles_per_track= 3,
        ))

        TerrainPerlin_kwargs = merge_dict(A1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.15],
        ))
    
    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [0.8, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( A1FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        z_low_kwargs = merge_dict(A1FieldCfg.termination.z_low_kwargs, dict(
            threshold= -1.,
        ))

    class domain_rand( A1FieldCfg.domain_rand ):
        class com_range( A1FieldCfg.domain_rand.com_range ):
            z = [-0.1, 0.1]
        
        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

        push_robots = False

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.1
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = -1e-2
            penetrate_volume = -1e-2
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
        soft_dof_pos_limit = 0.8
        max_contact_force = 100.0

    class curriculum( A1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 8000
        penetrate_volume_threshold_easier = 12000
        penetrate_depth_threshold_harder = 1000
        penetrate_depth_threshold_easier = 1600


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1JumpCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_a1"
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained jump model directory}"
        
        run_name = "".join(["Skills_",
        ("jump" if A1JumpCfg.terrain.BarrierTrack_kwargs["jump"]["jump_down_prob"] < 1. else "down"),
        ("_virtual" if A1JumpCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 500
    