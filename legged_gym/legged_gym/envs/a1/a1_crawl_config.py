import numpy as np
import os.path as osp
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1CrawlCfg( A1FieldCfg ):

    #### uncomment this to train non-virtual terrain
    class sensor( A1FieldCfg.sensor ):
        class proprioception( A1FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( A1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(A1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "crawl",
            ],
            track_block_length= 1.6,
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            virtual_terrain= False, # Change this to False for real terrain
        ))

        TerrainPerlin_kwargs = merge_dict(A1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= 0.12,
        ))
    
    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class asset( A1FieldCfg.asset ):
        terminate_after_contacts_on = ["base"]

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
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

    class rewards( A1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            alive = 2.
            # penetrate_depth = -6e-2 # comment this out if trianing non-virtual terrain
            # penetrate_volume = -6e-2 # comment this out if trianing non-virtual terrain
            exceed_dof_pos_limits = -8e-1
            # exceed_torque_limits_i = -2e-1
            exceed_torque_limits_l1norm = -4e-1
            # collision = -0.05
            # tilt_cond = 0.1
            torques = -1e-5
            yaw_abs = -0.1
            lin_pos_y = -0.1
        soft_dof_pos_limit = 0.9

    class curriculum( A1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 1500
        penetrate_volume_threshold_easier = 10000
        penetrate_depth_threshold_harder = 10
        penetrate_depth_threshold_easier = 400


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1CrawlCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.1
    
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_a1"
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained crawling model directory}"
        # load_run = "Aug21_06-12-58_Skillcrawl_propDelay0.00-0.05_virtual"
        # load_run = osp.join(logs_root, "field_a1_oracle/May21_05-25-19_Skills_crawl_pEnergy2e-5_rAlive1_pPenV6e-2_pPenD6e-2_pPosY0.2_kp50_noContactTerminate_aScale0.5")
        # load_run = osp.join(logs_root, "field_a1_oracle/Sep26_01-38-19_Skills_crawl_propDelay0.04-0.05_pEnergy-4e-5_pTorqueL13e-01_kp40_fromMay21_05-25-19")
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep26_14-30-24_Skills_crawl_propDelay0.04-0.05_pEnergy-2e-5_pDof8e-01_pTorqueL14e-01_rTilt5e-01_pCollision0.2_maxPushAng0.5_kp40_fromSep26_01-38-19")
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct09_09-58-26_Skills_crawl_propDelay0.04-0.05_pEnergy-1e-5_pDof8e-01_pTorqueL14e-01_maxPushAng0.0_kp40_fromSep26_14-30-24")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-19-00_Skills_crawl_propDelay0.04-0.05_pEnergy-1e-5_pDof8e-01_pTorqueL14e-01_pPosY0.1_maxPushAng0.3_kp40_fromOct09_09-58-26")

        run_name = "".join(["Skills_",
        ("Multi" if len(A1CrawlCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (A1CrawlCfg.terrain.BarrierTrack_kwargs["options"][0] if A1CrawlCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_comXRange{:.1f}-{:.1f}".format(A1CrawlCfg.domain_rand.com_range.x[0], A1CrawlCfg.domain_rand.com_range.x[1])),
        ("_noLinVel" if not A1CrawlCfg.env.use_lin_vel else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                A1CrawlCfg.sensor.proprioception.latency_range[0],
                A1CrawlCfg.sensor.proprioception.latency_range[1],
            ) if A1CrawlCfg.sensor.proprioception.delay_action_obs else ""
        ),
        # ("_pPenD{:.0e}".format(A1CrawlCfg.rewards.scales.penetrate_depth) if getattr(A1CrawlCfg.rewards.scales, "penetrate_depth", 0.) != 0. else ""),
        ("_pEnergySubsteps" + np.format_float_scientific(A1CrawlCfg.rewards.scales.legs_energy_substeps, precision= 1, exp_digits= 1, trim= "-") if getattr(A1CrawlCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),
        ("_pDof{:.0e}".format(-A1CrawlCfg.rewards.scales.exceed_dof_pos_limits) if getattr(A1CrawlCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0 else ""),
        ("_pTorque" + np.format_float_scientific(-A1CrawlCfg.rewards.scales.torques, precision= 1, exp_digits= 1, trim= "-") if getattr(A1CrawlCfg.rewards.scales, "torques", 0.) != 0 else ""),
        ("_pTorqueL1{:.0e}".format(-A1CrawlCfg.rewards.scales.exceed_torque_limits_l1norm) if getattr(A1CrawlCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0 else ""),
        # ("_rTilt{:.0e}".format(A1CrawlCfg.rewards.scales.tilt_cond) if getattr(A1CrawlCfg.rewards.scales, "tilt_cond", 0.) != 0 else ""),
        # ("_pYaw{:.1f}".format(-A1CrawlCfg.rewards.scales.yaw_abs) if getattr(A1CrawlCfg.rewards.scales, "yaw_abs", 0.) != 0 else ""),
        # ("_pPosY{:.1f}".format(-A1CrawlCfg.rewards.scales.lin_pos_y) if getattr(A1CrawlCfg.rewards.scales, "lin_pos_y", 0.) != 0 else ""),
        # ("_pCollision{:.1f}".format(-A1CrawlCfg.rewards.scales.collision) if getattr(A1CrawlCfg.rewards.scales, "collision", 0.) != 0 else ""),
        # ("_kp{:d}".format(int(A1CrawlCfg.control.stiffness["joint"])) if A1CrawlCfg.control.stiffness["joint"] != 50 else ""),
        ("_noDelayActObs" if not A1CrawlCfg.sensor.proprioception.delay_action_obs else ""),
        ("_noTanh"),
        ("_virtual" if A1CrawlCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 500
    