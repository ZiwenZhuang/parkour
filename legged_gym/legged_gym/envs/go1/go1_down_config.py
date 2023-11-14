from os import path as osp
import numpy as np
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1DownCfg( Go1FieldCfg ):
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
        curriculum = False

        BarrierTrack_kwargs = merge_dict(Go1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
            ],
            track_block_length= 1.6,
            jump= dict(
                height= (0.2, 0.45),
                depth= 0.3,
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 1., # probability of jumping down use it in non-virtual terrain
            ),
            virtual_terrain= True, # Change this to False for real terrain
            no_perlin_threshold= 0.6,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 3,
        ))

        TerrainPerlin_kwargs = merge_dict(Go1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.04, 0.15],
        ))

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [0.5, 1.]
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
        z_low_kwargs = merge_dict(Go1FieldCfg.termination.z_low_kwargs, dict(
            threshold= -3.,
        ))

    class domain_rand( Go1FieldCfg.domain_rand ):
        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )
        init_base_vel_range = merge_dict(Go1FieldCfg.domain_rand.init_base_vel_range, dict(
            x= [-0.1, 0.5],
        ))

        push_robots = True

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            action_rate = -0.1
            collision = -10.0
            delta_torques = -1e-07
            dof_acc = -1e-07
            down_cond = 0.04
            exceed_dof_pos_limits = -0.001
            exceed_torque_limits_l1norm = -0.2
            feet_contact_forces = -0.01
            hip_pos = -5.0
            legs_energy_substeps = -5e-06
            lin_pos_y = -0.3
            penetrate_depth = -0.002
            penetrate_volume = -0.002
            torques = -0.0001
            tracking_ang_vel = 0.1
            tracking_world_vel = 5.0
            yaw_abs = -0.
        soft_dof_pos_limit = 0.75
        max_contact_force = 200.
        only_positive_rewards = False

    class noise( Go1FieldCfg.noise ):
        add_noise = False

    class curriculum( Go1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 2000000
        penetrate_volume_threshold_easier = 4000000
        penetrate_depth_threshold_harder = 200000
        penetrate_depth_threshold_easier = 400000


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1DownCfgPPO( Go1FieldCfgPPO ):
    class algorithm( Go1FieldCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2

    class runner( Go1FieldCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_go1"
        resume = True
        load_run = "{Your trained climb model directory}"
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep28_14-59-53_Skills_down_pEnergySubsteps-8e-06_pPenD1e-02_pDofLimit-4e-01_pYawAbs_pTorqueL14e-01_pCollision-1e-1_downRedundant0.1_softDof0.8_pushRobot_propDelay0.04-0.05_kp40_kd0.5fromSep27_14-32-13")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct14_02-06-47_Skills_down_comXRange-0.2-0.2_pEnergySubsteps-4e-06_rAlive3_pPenD4e-03_pDHarder2.e+05_pushRobot_noTanh_virtualfromOct09_09-49-58")
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct25_10-48-54_down_pEnergySubsteps-4e-05_pDofLimit-4e-01_pYawAbs_pTorqueL14e-01_pActRate1.e-1_rDownCond3.e-1_kp40_kd0.5_virtualfromOct14_02-06-47")
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct26_08-45-02_Skills_down_pEnergySubsteps-4e-05_rTrackVel4._pZVel1e-01_pAngXYVel5e-02_pDTorques1e-07_rDownCond1.e-1_noTanh_virtualfromOct25_10-48-54")
        load_run = "Oct28_07-38-39_Skills_down_pEnergySubsteps-4e-05_rTrackVel5._pHipPos4e-01_rDownCond4.e-2_noTanh_virtualfromOct14_02-06-47"
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct28_13-08-21_Skills_down_pEnergySubsteps-4e-05_rTrackVel5._pHipPos8e-01_rDownCond4.e-2_withPropNoise_noTanh_virtualfromOct28_07-38-39")
        load_run = "Oct29_22-07-51_Skills_down_pEnergySubsteps-4e-05_rTrackVel5._pHipPos8e-01_rDownCond4.e-2_noComputerClip_noPropNoise_noTanh_virtualfromOct28_13-08-21"
        load_run = "Oct30_04-23-31_Skills_down_pEnergySubsteps-1e-05_rTrackVel5._pTorqueL14e-01_pHipPos5e+00_rDownCond4.e-2_allowNegativeReward_pushRobot_noComputerClip_noPropNoise_noTanh_virtual_fromOct29_22-07-51"

        run_name = "".join(["Skills_",
        ("down" if Go1DownCfg.terrain.BarrierTrack_kwargs["jump"]["jump_down_prob"] > 0. else "jump"),
        ("_pEnergySubsteps{:.0e}".format(Go1DownCfg.rewards.scales.legs_energy_substeps) if getattr(Go1DownCfg.rewards.scales, "legs_energy_substeps", 0.) != -1e-6 else ""),
        ("_rTrackVel" + np.format_float_positional(Go1DownCfg.rewards.scales.tracking_world_vel) if getattr(Go1DownCfg.rewards.scales, "tracking_world_vel", 0.) != 0. else ""),
        # ("_pLinVel" + np.format_float_positional(-Go1DownCfg.rewards.scales.world_vel_l2norm) if getattr(Go1DownCfg.rewards.scales, "world_vel_l2norm", 0.) != 0.0 else ""),
        # ("_rAlive" + np.format_float_positional(Go1DownCfg.rewards.scales.alive) if getattr(Go1DownCfg.rewards.scales, "alive", 0.) != 0.0 else ""),
        # ("_pDofLimit{:.0e}".format(Go1DownCfg.rewards.scales.exceed_dof_pos_limits) if getattr(Go1DownCfg.rewards.scales, "exceed_dof_pos_limits") != 0.0 else ""),
        # ("_pYawAbs" if getattr(Go1DownCfg.rewards.scales, "yaw_abs", 0.) != 0.1 else ""),
        ("_pTorqueL1{:.0e}".format(-Go1DownCfg.rewards.scales.exceed_torque_limits_l1norm) if getattr(Go1DownCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0 else ""),
        # ("_pActRate" + np.format_float_scientific(-Go1DownCfg.rewards.scales.action_rate, precision=1, exp_digits=1) if getattr(Go1DownCfg.rewards.scales, "action_rate", 0.) != 0.0 else ""),
        # ("_pZVel" + np.format_float_scientific(-Go1DownCfg.rewards.scales.lin_vel_z, precision=1, trim="-") if getattr(Go1DownCfg.rewards.scales, "lin_vel_z", 0.) != 0. else ""),
        # ("_pAngXYVel" + np.format_float_scientific(-Go1DownCfg.rewards.scales.ang_vel_xy, precision=1, trim="-") if getattr(Go1DownCfg.rewards.scales, "ang_vel_xy", 0.) != 0. else ""),
        # ("_pDTorques" + np.format_float_scientific(-Go1DownCfg.rewards.scales.delta_torques, precision=1, trim="-") if getattr(Go1DownCfg.rewards.scales, "delta_torques", 0.) != 0. else ""),
        # ("_pHipPos" + np.format_float_scientific(-Go1DownCfg.rewards.scales.hip_pos, precision=1, trim="-") if getattr(Go1DownCfg.rewards.scales, "hip_pos", 0.) != 0. else ""),
        # ("_rDownCond" + np.format_float_scientific(Go1DownCfg.rewards.scales.down_cond, precision=1, exp_digits=1) if getattr(Go1DownCfg.rewards.scales, "down_cond", 0.) != 0.0 else ""),
        ("_pCollision{:d}".format(int(Go1DownCfg.rewards.scales.collision)) if getattr(Go1DownCfg.rewards.scales, "collision", 0.) != 0.0 else ""),
        # ("_downRedundant{:.1f}".format(Go1DownCfg.terrain.BarrierTrack_kwargs["jump"]["down_forward_length"]) if (Go1DownCfg.terrain.BarrierTrack_kwargs["jump"].get("down_forward_length", 0.) > 0. and Go1DownCfg.terrain.BarrierTrack_kwargs["jump"].get("jump_down_prob", 0.) > 0.) else ""),
        ("_cmdRange{:.1f}-{:.1f}".format(*Go1DownCfg.commands.ranges.lin_vel_x)),
        ("_allowNegativeReward" if not Go1DownCfg.rewards.only_positive_rewards else ""),
        ("_pushRobot" if Go1DownCfg.domain_rand.push_robots else "_noPush"),
        ("_noComputerClip" if not Go1DownCfg.control.computer_clip_torque else ""),
        ("_noPropNoise" if not Go1DownCfg.noise.add_noise else "_withPropNoise"),
        # ("_kp{:d}".format(int(Go1DownCfg.control.stiffness["joint"])) if Go1DownCfg.control.stiffness["joint"] != 50 else ""),
        # ("_kd{:.1f}".format(Go1DownCfg.control.damping["joint"]) if Go1DownCfg.control.damping["joint"] != 1. else ""),
        ("_noTanh"),
        ("_virtual" if Go1DownCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        max_iterations = 20000
        save_interval = 200

