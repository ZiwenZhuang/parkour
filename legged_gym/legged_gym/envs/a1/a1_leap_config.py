import numpy as np
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1LeapCfg( A1FieldCfg ):

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
                "leap",
            ],
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(A1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.1],
        ))
    
    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [1.0, 1.5]
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
        roll_kwargs = merge_dict(A1FieldCfg.termination.roll_kwargs, dict(
            threshold= 0.4,
            leap_threshold= 0.4,
        ))
        z_high_kwargs = merge_dict(A1FieldCfg.termination.z_high_kwargs, dict(
            threshold= 2.0,
        ))

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-6
            alive = 2.
            penetrate_depth = -4e-3
            penetrate_volume = -4e-3
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1

    class curriculum( A1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000


class A1LeapCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_a1"
        run_name = "".join(["Skill",
        ("Multi" if len(A1LeapCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (A1LeapCfg.terrain.BarrierTrack_kwargs["options"][0] if A1LeapCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_propDelay{:.2f}-{:.2f}".format(
                A1LeapCfg.sensor.proprioception.latency_range[0],
                A1LeapCfg.sensor.proprioception.latency_range[1],
            ) if A1LeapCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_pEnergySubsteps{:.0e}".format(A1LeapCfg.rewards.scales.legs_energy_substeps) if A1LeapCfg.rewards.scales.legs_energy_substeps != -2e-6 else ""),
        ("_virtual" if A1LeapCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ])
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained leap model directory}"
        max_iterations = 20000
        save_interval = 500
    