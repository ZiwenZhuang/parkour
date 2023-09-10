import numpy as np
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1TiltCfg( A1FieldCfg ):

    #### uncomment this to train non-virtual terrain
    # class sensor( A1FieldCfg.sensor ):
    #     class proprioception( A1FieldCfg.sensor.proprioception ):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( A1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(A1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "tilt",
            ],
            tilt= dict(
                width= (0.24, 0.4),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(A1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))
    
    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.6]
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

    class domain_rand( A1FieldCfg.domain_rand ):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            alive = 2.
            penetrate_depth = -3e-3
            penetrate_volume = -3e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1

    class curriculum( A1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 4000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 300


class A1TiltCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_a1"
        run_name = "".join(["Skill",
        ("Multi" if len(A1TiltCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (A1TiltCfg.terrain.BarrierTrack_kwargs["options"][0] if A1TiltCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_propDelay{:.2f}-{:.2f}".format(
                A1TiltCfg.sensor.proprioception.latency_range[0],
                A1TiltCfg.sensor.proprioception.latency_range[1],
            ) if A1TiltCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_pPenV" + np.format_float_scientific(-A1TiltCfg.rewards.scales.penetrate_volume, trim= "-", exp_digits= 1) if getattr(A1TiltCfg.rewards.scales, "penetrate_volume", 0.) < 0. else ""),
        ("_pPenD" + np.format_float_scientific(-A1TiltCfg.rewards.scales.penetrate_depth, trim= "-", exp_digits= 1) if getattr(A1TiltCfg.rewards.scales, "penetrate_depth", 0.) < 0. else ""),
        ("_noPush" if not A1TiltCfg.domain_rand.push_robots else ""),
        ("_tiltMax{:.2f}".format(A1TiltCfg.terrain.BarrierTrack_kwargs["tilt"]["width"][1])),
        ("_virtual" if A1TiltCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ])
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtual terrain model directory}"
        load_run = "Aug17_11-13-14_WalkingBase_pEnergySubsteps2e-5_aScale0.5"
        load_run = "Aug23_22-03-41_Skilltilt_pPenV3e-3_pPenD3e-3_tiltMax0.40_virtual"
        max_iterations = 20000
        save_interval = 500
    