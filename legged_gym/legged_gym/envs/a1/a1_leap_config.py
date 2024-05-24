import numpy as np
from os import path as osp
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1LeapCfg( A1FieldCfg ):

    ### uncomment this to train non-virtual terrain
    class sensor( A1FieldCfg.sensor ):
        class proprioception( A1FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
    ### uncomment the above to train non-virtual terrain
    
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
                length= (0.2, 0.8),
                depth= (0.4, 0.8),
                height= 0.12,
            ),
            virtual_terrain= True, # Change this to False for real terrain
            no_perlin_threshold= 0.1,
        ))

        TerrainPerlin_kwargs = merge_dict(A1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.15],
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

    class domain_rand( A1FieldCfg.domain_rand ):
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            # legs_energy_substeps = -8e-6
            alive = 2.
            penetrate_depth = -1e-2
            penetrate_volume = -1e-2
            exceed_dof_pos_limits = -4e-1
            exceed_torque_limits_l1norm = -8e-1
            # feet_contact_forces = -1e-2
            torques = -2e-5
            collision = -0.5
            lin_pos_y = -0.1
            yaw_abs = -0.1
        soft_dof_pos_limit = 0.5

    class curriculum( A1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 9000
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 300
        penetrate_depth_threshold_easier = 5000


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1LeapCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2
    
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_a1"
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained leap model directory}"
        # load_run = osp.join(logs_root, "field_a1_oracle/Jun04_01-03-59_Skills_leap_pEnergySubsteps2e-6_rAlive2_pPenV4e-3_pPenD4e-3_pPosY0.20_pYaw0.20_pTorqueExceedSquare1e-3_leapH0.2_propDelay0.04-0.05_noPerlinRate0.2_aScale0.5")
        # load_run = "Sep27_02-44-48_Skills_leap_propDelay0.04-0.05_pDofLimit8e-01_pCollision0.1_kp40_kd0.5fromJun04_01-03-59"
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep27_14-56-25_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pDofLimit8e-01_pCollision0.1_kp40_kd0.5fromSep27_02-44-48")
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct05_02-16-22_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pPenD8.e-3_pDofLimit4e-01_pCollision0.5_kp40_kd0.5fromSep27_14-56-25")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct09_09-51-58_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pPenD1.e-2_pDofLimit4e-01_pCollision0.5_kp40_kd0.5fromOct05_02-16-22")
        
        run_name = "".join(["Skills_",
        ("Multi" if len(A1LeapCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (A1LeapCfg.terrain.BarrierTrack_kwargs["options"][0] if A1LeapCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_comXRange{:.1f}-{:.1f}".format(A1LeapCfg.domain_rand.com_range.x[0], A1LeapCfg.domain_rand.com_range.x[1])),
        ("_noLinVel" if not A1LeapCfg.env.use_lin_vel else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                A1LeapCfg.sensor.proprioception.latency_range[0],
                A1LeapCfg.sensor.proprioception.latency_range[1],
            ) if A1LeapCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_pEnergySubsteps{:.0e}".format(A1LeapCfg.rewards.scales.legs_energy_substeps) if getattr(A1LeapCfg.rewards.scales, "legs_energy_substeps", -2e-6) != -2e-6 else ""),
        ("_pTorques" + np.format_float_scientific(-A1LeapCfg.rewards.scales.torques, precision=1, exp_digits=1) if getattr(A1LeapCfg.rewards.scales, "torques", 0.) != -0. else ""),
        # ("_pPenD" + np.format_float_scientific(-A1LeapCfg.rewards.scales.penetrate_depth, precision=1, exp_digits=1) if A1LeapCfg.rewards.scales.penetrate_depth != -4e-3 else ""),
        # ("_pYaw{:.1f}".format(-A1LeapCfg.rewards.scales.yaw_abs) if getattr(A1LeapCfg.rewards.scales, "yaw_abs", 0.) != 0. else ""),
        # ("_pDofLimit{:.0e}".format(-A1LeapCfg.rewards.scales.exceed_dof_pos_limits) if getattr(A1LeapCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0. else ""),
        # ("_pCollision{:.1f}".format(-A1LeapCfg.rewards.scales.collision) if getattr(A1LeapCfg.rewards.scales, "collision", 0.) != 0. else ""),
        ("_pContactForces" + np.format_float_scientific(-A1LeapCfg.rewards.scales.feet_contact_forces, precision=1, exp_digits=1) if getattr(A1LeapCfg.rewards.scales, "feet_contact_forces", 0.) != 0. else ""),
        ("_leapHeight{:.1f}".format(A1LeapCfg.terrain.BarrierTrack_kwargs["leap"]["height"]) if A1LeapCfg.terrain.BarrierTrack_kwargs["leap"]["height"] != 0.2 else ""),
        # ("_kp{:d}".format(int(A1LeapCfg.control.stiffness["joint"])) if A1LeapCfg.control.stiffness["joint"] != 50 else ""),
        # ("_kd{:.1f}".format(A1LeapCfg.control.damping["joint"]) if A1LeapCfg.control.damping["joint"] != 0. else ""),
        ("_noDelayActObs" if not A1LeapCfg.sensor.proprioception.delay_action_obs else ""),
        ("_noTanh"),
        ("_virtual" if A1LeapCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 500
    