from os import path as osp
import numpy as np
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1TiltCfg( Go1FieldCfg ):

    #### uncomment this to train non-virtual terrain
    # class sensor( Go1FieldCfg.sensor ):
    #     class proprioception( Go1FieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( Go1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(Go1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "tilt",
            ],
            tilt= dict(
                width= (0.24, 0.8),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            virtual_terrain= True, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(Go1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.6]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class termination( Go1FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        roll_kwargs = merge_dict(Go1FieldCfg.termination.roll_kwargs, dict(
            threshold= 0.4,
            leap_threshold= 0.4,
        ))
        z_high_kwargs = merge_dict(Go1FieldCfg.termination.z_high_kwargs, dict(
            threshold= 2.0,
        ))

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = -4e-3
            penetrate_volume = -4e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -1e-1

    class curriculum( Go1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000

class Go1TiltCfgPPO( Go1FieldCfgPPO ):    
    class runner( Go1FieldCfgPPO.runner ):
        run_name = "".join(["Skill",
        ("Multi" if len(Go1TiltCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go1TiltCfg.terrain.BarrierTrack_kwargs["options"][0] if Go1TiltCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_tiltMax{:.1f}".format(Go1TiltCfg.terrain.BarrierTrack_kwargs["tilt"]["width"][1]) if Go1TiltCfg.terrain.BarrierTrack_kwargs["tilt"]["width"][1] > 0.5 else ""),
        ])
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "Sep20_03-37-32_SkillopensourcePlaneWalking_pEnergySubsteps1e-5_pTorqueExceedIndicate1e-1_aScale0.5_tClip202025"
        max_iterations = 20000
        save_interval = 500

