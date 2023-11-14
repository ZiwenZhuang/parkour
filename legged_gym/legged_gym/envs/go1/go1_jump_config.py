from os import path as osp
import numpy as np
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1JumpCfg( Go1FieldCfg ):

    class init_state( Go1FieldCfg.init_state ):
        pos = [0., 0., 0.45]

    #### uncomment this to train non-virtual terrain
    class sensor( Go1FieldCfg.sensor ):
        class proprioception( Go1FieldCfg.sensor.proprioception ):
            latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain
    
    class terrain( Go1FieldCfg.terrain ):
        max_init_terrain_level = 10
        border_size = 5
        slope_treshold = 20.
        curriculum = True

        BarrierTrack_kwargs = merge_dict(Go1FieldCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                "jump",
            ],
            track_width= 1.6,
            track_block_length= 1.6,
            wall_thickness= 0.2,
            jump= dict(
                # height= (0.38, 0.46),
                height= (0.2, 0.46),
                depth= (0.1, 0.2),
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            virtual_terrain= False, # Change this to False for real terrain
            no_perlin_threshold= 0.06,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 3,
        ))

        TerrainPerlin_kwargs = merge_dict(Go1FieldCfg.terrain.TerrainPerlin_kwargs, dict(
            zScale= [0.05, 0.12],
        ))

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [0.5, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control( Go1FieldCfg.control ):
        computer_clip_torque = False

    class asset( Go1FieldCfg.asset ):
        penalize_contacts_on = ["base", "thigh", "calf", "imu"]
        terminate_after_contacts_on = ["base"]

    class termination( Go1FieldCfg.termination ):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_high",
            "out_of_track",
        ]
        timeout_at_finished = True
        timeout_at_border = True

    class domain_rand( Go1FieldCfg.domain_rand ):
        class com_range( Go1FieldCfg.domain_rand.com_range ):
            z = [-0.1, 0.1]
        
        init_base_pos_range = dict(
            x= [0.2, 0.8],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.1, 0.1],
            pitch= [-0.1, 0.1],
        )

        push_robots = True

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.08
            # alive = 10.
            tracking_world_vel = 5. ######## 1. always ########
            legs_energy_substeps = -6.e-5 # -6e-6
            jump_x_vel_cond = 0.1 ######## 1. segment ########
            # hip_pos = -5.
            front_hip_pos = -5.
            rear_hip_pos = -0.000001
            penetrate_depth = -6e-3
            penetrate_volume = -6e-4
            exceed_dof_pos_limits = -8e-4 # -8e-4
            exceed_torque_limits_l1norm = -2. # -8e-2 # -6e-2 
            action_rate = -0.1
            delta_torques = -1.e-5
            dof_acc = -1e-7
            torques = -4e-4 # 2e-3
            yaw_abs = -0.1
            lin_pos_y = -0.4
            collision = -1.
            sync_all_legs_cond = -0.3 # -0.6
            dof_error = -0.06
        only_positive_rewards = False # False
        soft_dof_pos_limit = 0.8
        max_contact_force = 100.
        tracking_sigma = 0.3

    class noise( Go1FieldCfg.noise ):
        add_noise = False

    class curriculum( Go1FieldCfg.curriculum ):
        penetrate_volume_threshold_harder = 6000
        penetrate_volume_threshold_easier = 12000
        penetrate_depth_threshold_harder = 600
        penetrate_depth_threshold_easier = 1200

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1JumpCfgPPO( Go1FieldCfgPPO ):
    class algorithm( Go1FieldCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.21
    
    class runner( Go1FieldCfgPPO.runner ):
        resume = True
        load_run = "{Your traind walking model directory}"
        # load_run = "Sep20_03-37-32_SkillopensourcePlaneWalking_pEnergySubsteps1e-5_pTorqueExceedIndicate1e-1_aScale0.5_tClip202025"
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep27_14-32-13_Skills_jump_pEnergySubsteps-8e-06_pDofLimit-8e-01_pTorqueL18e-01_pContact1e-02_pCollision0.4_pushRobot_propDelay0.04-0.05_kp40_kd0.5_fromSep27_03-36-02")
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct07_10-24-52_Skills_jump_pEnergy1e-06_pY-1e-02_pCollision-1e-02_noPush_noTanh_jumpRange0.2-0.6_fromSep27_14-32-13")
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct19_06-29-14_Skills_jump_comXRange-0.2-0.2_pLinVel1._pDofAcc5e-7_pTorque6e-05_jumpHeight0.10-0.46_noDelayActObs_noTanh_minStd0.1_fromOct18_15-57-31")
        # load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct16_10-55-09_Skills_jump_comXRange-0.2-0.2_pLinVel1.4_pTorque2e-06_pY0.1_pCollision0.1_pushRobot_jumpHeight0.10-0.45_propDelay0.04-0.05_noTanh_fromOct13_09-12-39")
        ######################################
        # load_run = "Oct20_16-09-47_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel2._propDelay0.04-0.05_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct19_06-29-14"
        # load_run = "Oct21_05-34-37_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel2._noPropNoise_propDelay0.04-0.05_pCollision-5e-01_noPush_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct20_16-09-47"
        # load_run = "Oct21_12-30-28_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel2._noPropNoise_propDelay0.04-0.05_pCollision-1e+00_noPush_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct21_05-34-37"
        # load_run = "Oct21_14-26-29_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel1.2_noPropNoise_propDelay0.04-0.05_pCollision-2e+00_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct21_12-30-28"
        # load_run = "Oct20_16-09-47_275k_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel2._propDelay0.04-0.05_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct19_06-29-14"
        # load_run = osp.join(logs_root, "field_go1_before20231030_tuneJump_noTanh", "Oct20_16-09-47_Skills_jump_pLinVel1._pDofLimit-8e-03_pDofAcc2.5e-07_pTorque1e-05_rJumpXVel2._propDelay0.04-0.05_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct19_06-29-14")
        # load_run = "Oct22_02-20-52_Skills_jump_pLinVel1._noPContact_pDofAcc2.5e-07_rJumpXVel1.4_noPropNoise_propDelay0.04-0.05_pCollision-1e+00_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct20_16-09-47"
        # load_run = "Oct22_08-47-56_Skills_jump_rTrackVel2._pJumpSameLegs0.8_noPContact_pDofAcc2.5e-07_rJumpXVel1._noPropNoise_propDelay0.04-0.05_pCollision-4e-01_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct22_02-20-52"
        # load_run = "Oct22_08-47-13_Skills_jump_rTrackVel2._pJumpSameLegs0.6_noPContact_pDofAcc2.5e-07_rJumpXVel1.2_noPropNoise_propDelay0.04-0.05_pCollision-6e-01_pushRobot_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct22_02-20-52"
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct22_15-19-26_Skills_jump_rTrackVel3._pJumpSameLegs0.6_noPContact_pDofAcc1.5e-07_rJumpXVel1.5_noPropNoise_propDelay0.04-0.05_pCollision-6e-01_noPush_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct22_08-47-13")
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct24_07-41-26_Skills_jump_rTrackVel4._pY-2e-01_pJumpSameLegs0.2_noPContact_pDofAcc2.5e-07_rJumpXVel1.5_noPropNoise_propDelay0.04-0.05_pCollision-1e-01_noPush_minStd0.2_noTanh_zeroResetAction_jumpRange0.4-0.5_fromOct20_16-09-47")
        # load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct25_11-36-16_Skills_jump_pEnergySubsteps3e-05_rTrackVel4._pY-3e-01_propDelay0.04-0.05_noPropNoise_noPush_noTanh_zeroResetAction_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct24_07-41-26")
        # load_run = "Oct27_08-20-20_Skills_jump_rTrackVel5._noPPen_pY-3e-01_pSyncSymLegs6e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct25_11-36-16"
        # load_run = "Oct27_09-17-23_Skills_jump_rTrackVel5._noPPen_pDofErr6e-02_pDofErrCond6e-01_pSyncSymLegs6e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_08-20-20"
        #### Force hip_pos and sync_legs
        # load_run = "Oct27_13-20-59_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._pDofErr4e-02_pDofErrCond4e-01_pHipPos5e-01_pCollision5e+00_pSyncSymLegs6e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_09-17-23"
        # load_run = "Oct27_13-24-58_Skills_jump_pEnergySubsteps3e-05_rTrackVel5._pDofErr4e-02_pDofErrCond3e-01_pHipPos3e-01_pCollision5e+00_pSyncSymLegs6e-01_propDelay0.04-0.05_withPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_09-17-23"
        #### Adding RL gamma for energy penalty
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct27_16-04-45_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._pPenD-6e-03_pDofErrCond6e-01_pHipPos5e-01_pCollision6e+00_pSyncSymLegs4e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_13-20-59")
        #### Wide spreading legs first
        # load_run = "Oct28_03-08-02_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._rAlive0.5_pDofErrCond8e-01_pHipPos5e-01_pSyncSymLegs2e-01_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_jumpRange0.4-0.5_fromOct27_16-04-45"
        # load_run = "Oct28_07-07-03_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._rAlive1.0_pPenD-1e-03_pDofErrCond8e-01_pHipPos3e+00_pSyncSymLegs2e-01_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_jumpRange0.4-0.5_fromOct27_16-04-45"
        load_run = "Oct28_08-54-23_Skills_jump_pEnergySubsteps1e-05_rTrackVel3._rAlive0.5_pYaw-1e-01_pHipPos5e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_jumpRange0.4-0.5_fromOct28_03-08-02"
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct28_11-05-56_Skills_jump_pEnergySubsteps2e-06_rTrackVel5._pYaw-1e-01_pHipPos5e+00_pCollision1e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_jumpRange0.4-0.5_allowNegativeReward_fromOct28_08-54-23")
        #### lowering max torques and energy
        load_run = "Oct29_03-21-19_Skills_jump_pEnergySubsteps6e-06_rTrackVel5._pHipPos5e+00_pTorque2e-05_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.4-0.5_allowNegativeReward_fromOct28_11-05-56"
        load_run = "Oct29_07-54-42_Skills_jump_pEnergySubsteps1e-05_rTrackVel5._pHipPos5e+00_pTorqueExceed6e-02_pTorque2e-03_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.4-0.5_allowNegativeReward_fromOct29_03-21-19"
        load_run = "Oct29_12-18-08_Skills_jump_pEnergySubsteps1e-05_rTrackVel6._pTorqueExceed6e-02_pTorque1e-03_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_07-54-42"
        load_run = "Oct29_15-18-02_Skills_jump_pEnergySubsteps1e-05_rTrackVel4._pTorqueExceed3e-02_pTorque1e-03_noJumpBonous_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_12-18-08"
        load_run = "Oct29_16-37-44_Skills_jump_pEnergySubsteps1.2e-05_rTrackVel4._pTorqueExceed6e-02_pTorque2e-03_noJumpBonous_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_15-18-02"
        load_run = "Oct29_20-17-31_Skills_jump_pEnergySubsteps5e-05_rTrackVel4._pTorqueExceed1e+00_pTorque4e-04_noJumpBonous_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_16-37-44"
        load_run = "Oct30_11-11-43_Skills_jump_pEnergySubsteps6e-05_rTrackVel5._pY-8e-01_pTorqueExceed1.2e+00_pTorque4e-04_pDTorques1e-06_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_20-17-31"
        load_run = osp.join(logs_root, "field_go1_noTanh_oracle", "Oct30_13-00-12_Skills_jump_pEnergySubsteps6e-05_rTrackVel5._pY-4e-01_pTorqueExceed1.8e+00_pTorque4e-04_pDTorques1e-05_propDelay0.04-0.05_noPropNoise_noPush_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_20-17-31")

        run_name = "".join(["Skills_",
        ("Multi" if len(Go1JumpCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else (Go1JumpCfg.terrain.BarrierTrack_kwargs["options"][0] if Go1JumpCfg.terrain.BarrierTrack_kwargs["options"] else "PlaneWalking")),
        ("_pEnergySubsteps" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.legs_energy_substeps, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "legs_energy_substeps", -2e-5) != -2e-5 else ""),
        # ("_rTrackVel" + np.format_float_positional(Go1JumpCfg.rewards.scales.tracking_world_vel) if getattr(Go1JumpCfg.rewards.scales, "tracking_world_vel", 0.) != 0. else ""),
        # ("_pLinVel" + np.format_float_positional(-Go1JumpCfg.rewards.scales.world_vel_l2norm) if getattr(Go1JumpCfg.rewards.scales, "world_vel_l2norm", 0.) != 0.0 else ""),
        # ("_rAlive{:.1f}".format(Go1JumpCfg.rewards.scales.alive) if getattr(Go1JumpCfg.rewards.scales, "alive", 0.) != 0. else ""),
        # ("_pPenD{:.0e}".format(Go1JumpCfg.rewards.scales.penetrate_depth) if getattr(Go1JumpCfg.rewards.scales, "penetrate_depth", 0.) < 0. else ""),
        # ("_pYaw{:.0e}".format(Go1JumpCfg.rewards.scales.yaw_abs) if getattr(Go1JumpCfg.rewards.scales, "yaw_abs", 0.) < 0. else ""),
        # ("_pY{:.0e}".format(Go1JumpCfg.rewards.scales.lin_pos_y) if getattr(Go1JumpCfg.rewards.scales, "lin_pos_y", 0.) < 0. else ""),
        # ("_pDofLimit{:.0e}".format(Go1JumpCfg.rewards.scales.exceed_dof_pos_limits) if getattr(Go1JumpCfg.rewards.scales, "exceed_dof_pos_limits", 0.) < 0. else ""),
        # ("_pDofErr" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.dof_error, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "dof_error", 0.) != 0. else ""),
        # ("_pDofErrCond" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.dof_error_cond, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "dof_error_cond", 0.) != 0. else ""),
        # ("_pHipPos" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.hip_pos, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "hip_pos", 0.) != 0. else ""),
        ("_pFHipPos" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.front_hip_pos, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "front_hip_pos", 0.) != 0. else ""),
        ("_pRHipPos" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.rear_hip_pos, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "rear_hip_pos", 0.) != 0. else ""),
        # ("_pTorqueExceedI" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.exceed_torque_limits_i, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "exceed_torque_limits_i", 0.) != 0. else ""),
        ("_pTorqueExceed" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.exceed_torque_limits_l1norm, precision=1, trim="-") if Go1JumpCfg.rewards.scales.exceed_torque_limits_l1norm != -8e-1 else ""),
        # ("_pContact" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.feet_contact_forces, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "feet_contact_forces", 0.) != 0. else ""),
        # ("_pJumpSameLegs{:.1f}".format(-Go1JumpCfg.rewards.scales.sync_legs_cond) if getattr(Go1JumpCfg.rewards.scales, "sync_legs_cond", 0.) != 0. else ""),
        # ("_pDofAcc" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.dof_acc, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "dof_acc", 0.) != 0. else ""),
        # ("_pTorque" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.torques, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "torques", 0.) != 0. else ""),
        # ("_rJumpXVel" + np.format_float_positional(Go1JumpCfg.rewards.scales.jump_x_vel_cond) if getattr(Go1JumpCfg.rewards.scales, "jump_x_vel_cond", 0.) != 0. else "_noJumpBonous"),
        # ("_pCollision" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.collision, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "collision", 0.) < 0. else ""),
        # ("_pSyncSymLegs" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.sync_all_legs_cond, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "sync_all_legs_cond", 0.) != 0. else ""),
        # ("_pZVel" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.lin_vel_z, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "lin_vel_z", 0.) != 0. else ""),
        # ("_pAngXYVel" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.ang_vel_xy, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "ang_vel_xy", 0.) != 0. else ""),
        # ("_pDTorques" + np.format_float_scientific(-Go1JumpCfg.rewards.scales.delta_torques, precision=1, trim="-") if getattr(Go1JumpCfg.rewards.scales, "delta_torques", 0.) != 0. else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
            Go1JumpCfg.sensor.proprioception.latency_range[0],
            Go1JumpCfg.sensor.proprioception.latency_range[1],
        )),
        # ("_softDof{:.1f}".format(Go1JumpCfg.rewards.soft_dof_pos_limit) if Go1JumpCfg.rewards.soft_dof_pos_limit != 0.9 else ""),
        # ("_timeoutFinished" if getattr(Go1JumpCfg.termination, "timeout_at_finished", False) else ""),
        # ("_noPitchTerm" if "pitch" not in Go1JumpCfg.termination.termination_terms else ""),
        ("_noPropNoise" if not Go1JumpCfg.noise.add_noise else "_withPropNoise"),
        ("_noPush" if not Go1JumpCfg.domain_rand.push_robots else "_pushRobot"),
        ("_minStd0.21"),
        ("_entropy0.01"),
        # ("_gamma0.999"),
        ("_noTanh"),
        # ("_zeroResetAction" if Go1JumpCfg.init_state.zero_actions else ""),
        # ("_actionClip" + Go1JumpCfg.normalization.clip_actions_method if getattr(Go1JumpCfg.normalization, "clip_actions_method", "") != "" else ""),
        # ("_noDelayActObs" if not Go1JumpCfg.sensor.proprioception.delay_action_obs else ""),
        ("_noComputerClip" if not Go1JumpCfg.control.computer_clip_torque else ""),
        ("_jumpRange{:.1f}-{:.1f}".format(*Go1JumpCfg.terrain.BarrierTrack_kwargs["jump"]["height"]) if Go1JumpCfg.terrain.BarrierTrack_kwargs["jump"]["height"] != (0.1, 0.6) else ""),
        # ("_noCurriculum" if not Go1JumpCfg.terrain.curriculum else ""),
        # ("_allowNegativeReward" if not Go1JumpCfg.rewards.only_positive_rewards else ""),
        ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2]) if resume else "_noResume"),
        ])
        max_iterations = 20000
        save_interval = 200

