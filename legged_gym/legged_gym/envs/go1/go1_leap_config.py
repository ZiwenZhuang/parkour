from os import path as osp
import numpy as np
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1LeapCfg( Go1FieldCfg ):

    class init_state( Go1FieldCfg.init_state ):
        pos = [0., 0., 0.4]

    #### uncomment this to train non-virtual terrain
    class sensor( Go1FieldCfg.sensor ):
        class proprioception( Go1FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
            # latency_range = [0.06-0.005, 0.06+0.005]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( Go1FieldCfg.terrain ):
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(Go1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "leap",
            ],
            leap= dict(
                length= (0.3, 0.8),
                depth= (0.4, 0.8),
                height= 0.15,
            ),
            wall_height= -0.4,
            virtual_terrain= True, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
        ))

        TerrainPerlin_kwargs = merge_dict(Go1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.12],
        ))

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [1.8, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control( Go1FieldCfg.control ):
        computer_clip_torque = True

    class termination( Go1FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        # roll_kwargs = merge_dict(Go1FieldCfg.termination.roll_kwargs, dict(
        #     threshold= 0.7,
        #     leap_threshold= 0.7,
        # ))
        # pitch_kwargs = merge_dict(Go1FieldCfg.termination.pitch_kwargs, dict(
        #     threshold= 0.7,
        #     leap_threshold= 0.7,
        # ))
        z_low_kwargs = merge_dict(Go1FieldCfg.termination.z_low_kwargs, dict(
            threshold= -0.1,
        ))
        z_high_kwargs = merge_dict(Go1FieldCfg.termination.z_high_kwargs, dict(
            threshold= 2.0,
        ))

    class domain_rand( Go1FieldCfg.domain_rand ):
        class com_range( Go1FieldCfg.domain_rand.com_range ):
            z = [-0.2, 0.2]
        
        init_base_pos_range = dict(
            x= [0.2, 0.8],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

        push_robots = False

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            tracking_world_vel = 5.
            # leap_bonous_cond = 6.
            legs_energy_substeps = -7e-6 # 5e5 spike (not scaled)
            # alive = 3.
            penetrate_depth = -1e-2 # 8 spike (not scaled)
            penetrate_volume = -1e-4 # 100 spike (not scaled)
            exceed_dof_pos_limits = -8e-1
            exceed_torque_limits_l1norm = -1.
            # action_rate = -0.1
            delta_torques = -1e-7
            dof_acc = -1e-7
            torques = -4e-5 # 2000 spike (not scaled)
            yaw_abs = -0.2
            collision = -1.
            lin_pos_y = -0.4
            orientation = -0.1 # 0.3 segment (not scaled)
            hip_pos = -5. # 0.5 spike (not scaled)
            dof_error = -0.15 # 1.2 spike (not scaled)
        tracking_sigma = 0.35
        only_positive_rewards = False
        soft_dof_pos_limit = 0.7

    class curriculum( Go1FieldCfg.curriculum ):
        # penetrate_volume_threshold_harder = 9000
        # penetrate_volume_threshold_easier = 10000
        # penetrate_depth_threshold_harder = 300
        # penetrate_depth_threshold_easier = 5000
        ####### use extreme large value to disable penetrate curriculum
        penetrate_volume_threshold_harder = 900000
        penetrate_volume_threshold_easier = 1000000
        penetrate_depth_threshold_harder = 30000
        penetrate_depth_threshold_easier = 500000

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1LeapCfgPPO( Go1FieldCfgPPO ):

    class runner( Go1FieldCfgPPO.runner ):
        resume = True
        load_run = "{Your traind walking model directory}"
        # load_run = "Sep20_03-37-32_SkillopensourcePlaneWalking_pEnergySubsteps1e-5_pTorqueExceedIndicate1e-1_aScale0.5_tClip202025"
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep27_14-56-25_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pDofLimit8e-01_pCollision0.1_kp40_kd0.5fromSep27_02-44-48")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct09_09-51-58_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pPenD1.e-2_pDofLimit4e-01_pCollision0.5_kp40_kd0.5fromOct05_02-16-22")
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct25_11-39-12_Skills_leap_pEnertySubsteps4.e-5_pActRate2.e-1_pDofLimit8.e-1_pCollision5.e-1_noPropNoise_noTanh_zeroResetAction_actionCliphard_fromOct09_09-51-58")
        load_run = "Oct28_03-32-35_Skills_leap_pEnertySubsteps2.e-5_rTrackVel2._rAlive2.0_pActRate2.e-1_pHipPos5.e+0_noTanh_actionCliphard_virtual_fromOct25_11-39-12"
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct28_13-23-39_Skills_leap_pEnertySubsteps2.e-5_rTrackVel2._rAlive2.0_pSyncAllLegs6.e-1_pOrient6.e-1_pHipPos5.e+0_noTanh_actionCliphard_virtual_fromOct28_03-32-35")
        load_run = "Oct29_04-23-33_Skills_leap_pEnertySubsteps6.e-6_rTrackVel6._pTorques2.e-5_pHipPos5.e+0_pDorErrCond8.e-2_noComputerClip_noTanh_allowNegativeReward_actionCliphard_virtual_fromOct28_13-23-39"
        # load_run = "Oct29_08-30-23_Skills_leap_pEnertySubsteps6.e-6_rTrackVel6._pTorqueExceed2.e-1_pPosY2.e-1_pTorques2.e-5_pHipPos5.e+0_noComputerClip_noTanh_allowNegativeReward_actionCliphard_virtual_fromOct29_04-23-33"
        # load_run = "Oct29_15-46-30_Skills_leap_pEnertySubsteps6.e-6_rTrackVel6._pTorqueExceed2.e-1_pOrient3.e-1_pTorques2.e-5_noComputerClip_noTanh_allowNegativeReward_actionCliphard_virtual_fromOct29_08-30-23"
        load_run = "Oct30_02-07-26_Skills_leap_pEnertySubsteps6.e-6_rTrackVel6._pTorqueExceed1.e-1_pPosY4.e-1_pTorques2.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_allowNegativeReward_actionCliphard_virtual_fromOct29_04-23-33"
        load_run = "Oct30_04-10-12_Skills_leap_pEnertySubsteps4.e-6_rTrackVel5._pTorqueExceed3.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_02-07-26"
        load_run = "Oct30_05-07-24_Skills_leap_pEnertySubsteps5.e-6_rTrackVel5._pTorqueExceed4.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_04-10-12"
        
        load_run = "Oct30_05-08-03_Skills_leap_pEnertySubsteps6.e-6_rTrackVel5._pTorqueExceed4.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_04-10-12"
        load_run = "Oct30_05-08-20_Skills_leap_pEnertySubsteps6.e-6_rTrackVel5._pTorqueExceed5.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_04-10-12"
        load_run = "Oct30_07-32-10_Skills_leap_pEnertySubsteps6.e-6_rTrackVel5._pTorqueExceed8.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_pDorErr1.5e-1_noPenCurriculum_noCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_05-07-24"

        run_name = "".join(["Skills_",
        ("Multi" if len(Go1LeapCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go1LeapCfg.terrain.BarrierTrack_kwargs["options"][0] if Go1LeapCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_pEnertySubsteps" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.legs_energy_substeps, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "legs_energy_substeps", 0.0) != 0.0 else ""),
        ("_rTrackVel" + np.format_float_positional(Go1LeapCfg.rewards.scales.tracking_world_vel) if getattr(Go1LeapCfg.rewards.scales, "tracking_world_vel", 0.) != 0. else ""),
        ("_rAlive{:.1f}".format(Go1LeapCfg.rewards.scales.alive) if getattr(Go1LeapCfg.rewards.scales, "alive", 0.) != 0. else ""),
        # ("_pActRate" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.action_rate, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "action_rate", 0.) != 0.0 else ""),
        # ("_pDofAcc" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.dof_acc, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "dof_acc", 0.) != 0.0 else ""),
        # ("_pDofLimit" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.exceed_dof_pos_limits, precision=1, exp_digits=1) if Go1LeapCfg.rewards.scales.exceed_dof_pos_limits != 0.0 else ""),
        # ("_pCollision" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.collision, precision=1, exp_digits=1) if Go1LeapCfg.rewards.scales.collision != 0.0 else ""),
        # ("_leapBonousCond" if getattr(Go1LeapCfg.rewards.scales, "leap_bonous_cond", 0.) != 0.0 else ""),
        ("_pTorqueExceed" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.exceed_torque_limits_l1norm, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0.0 else ""),
        # ("_pSyncAllLegs" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.sync_all_legs, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "sync_all_legs", 0.) != 0.0 else ""),
        # ("_pOrient" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.orientation, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "orientation", 0.) != 0.0 else ""),
        # ("_pPenD" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.penetrate_depth, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "penetrate_depth", 0.) != 0.0 else ""),
        ("_pPosY" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.lin_pos_y, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "lin_pos_y", 0.) != 0.0 else ""),
        # ("_pPenV" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.penetrate_volume, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "penetrate_volume", 0.) != 0.0 else ""),
        ("_pTorques" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.torques, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "torques", 0.) != 0.0 else ""),
        ("_pHipPos" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.hip_pos, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "hip_pos", 0.) != 0.0 else ""),
        # ("_pDorErrCond" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.dof_error_cond, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "dof_error_cond", 0.) != 0.0 else ""),
        ("_pDorErr" + np.format_float_scientific(-Go1LeapCfg.rewards.scales.dof_error, precision=1, exp_digits=1) if getattr(Go1LeapCfg.rewards.scales, "dof_error", 0.) != 0.0 else ""),
        # ("_noPropNoise" if not Go1LeapCfg.noise.add_noise else ""),
        ("_noPenCurriculum" if Go1LeapCfg.curriculum.penetrate_volume_threshold_harder > 2e5 else ""),
        ("_noCurriculum" if not Go1LeapCfg.terrain.curriculum else ""),
        ("_noComputerClip" if not Go1LeapCfg.control.computer_clip_torque else ""),
        ("_noTanh"),
        ("_noPush" if not Go1LeapCfg.domain_rand.push_robots else "_pushRobot"),
        # ("_zeroResetAction" if Go1LeapCfg.init_state.zero_actions else ""),
        ("_allowNegativeReward" if not Go1LeapCfg.rewards.only_positive_rewards else ""),
        ("_actionClip" + Go1LeapCfg.normalization.clip_actions_method if getattr(Go1LeapCfg.normalization, "clip_actions_method", "") != "" else ""),
        ("_virtual" if Go1LeapCfg.terrain.BarrierTrack_kwargs["virtual_terrain"] else ""),
        ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2]) if resume else ""),
        ])
        max_iterations = 20000
        save_interval = 200

