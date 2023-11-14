from collections import OrderedDict
import os
from datetime import datetime
import os.path as osp
import numpy as np
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from legged_gym.envs.a1.a1_field_distill_config import A1FieldDistillCfg, A1FieldDistillCfgPPO
from legged_gym.utils.helpers import merge_dict

class Go1FieldDistillCfg( Go1FieldCfg ):
    class env( Go1FieldCfg.env ):
        num_envs = 256
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            "forward_depth",
        ]
        privileged_obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            # "forward_depth",
            "base_pose", # 6
            "robot_config", # 1 + 3 + 1 + 12
            "engaging_block", # 1 + (4 + 1) + 2
            "sidewall_distance", # 2
        ]
        use_lin_vel = False
        privileged_use_lin_vel = True
        # if True privileged_obs will not have noise based on implementations
        privileged_obs_gets_privilege = False # for jump temporarily

    class init_state( Go1FieldCfg.init_state ):
        pos = [0.0, 0.0, 0.43] # x,y,z [m]

    class sensor( Go1FieldCfg.sensor ):
        class forward_camera( Go1FieldCfg.sensor.forward_camera ):
            resolution = [int(240/4), int(424/4)]
            # position = dict(
            #     mean= [0.245, 0.0075, 0.072+0.018],
            #     std= [0.002, 0.002, 0.0005],
            # ) # position in base_link ##### small randomization
            # rotation = dict(
            #     lower= [0, 0, 0],
            #     upper= [0, 0, 0],
            # ) # rotation in base_link ##### small randomization
            ########## new camera extrinsics with 30degree down ####################
            position = dict(
                mean= [0.245+0.027, 0.0075, 0.072+0.02],
                std= [0.002, 0.002, 0.0002],
            ) # position in base_link ##### small randomization
            rotation = dict(
                lower= [0, 0.5, 0], # positive for pitch down
                upper= [0, 0.54, 0],
            ) # rotation in base_link ##### small randomization
            ########## new camera extrinsics with 30degree down  ####################
            ########## new camera extrinsics with 15degree down ####################
            # position = dict(
            #     mean= [0.245+0.027, 0.0075, 0.072+0.018],
            #     std= [0.003, 0.002, 0.0005],
            # ) # position in base_link ##### small randomization
            # rotation = dict(
            #     lower= [0, 0.24, 0], # positive for pitch down
            #     upper= [0, 0.28, 0],
            # ) # rotation in base_link ##### small randomization
            ########## new camera extrinsics with 15degree down  ####################
            resized_resolution= [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [85, 87] # measured around 87 degree
            # for go1, usb2.0, 480x640, d435i camera
            latency_range = [0.25, 0.30] # [s]
            # latency_range = [0.28, 0.36] # [s]
            latency_resample_time = 5.0 # [s]
            refresh_duration = 1/10 # [s] for (240, 424 option with onboard script fixed to no more than 20Hz)
            
            # config to simulate stero RGBD camera
            crop_top_bottom = [0, 0]
            crop_left_right = [int(60/4), int(46/4)]
            depth_range = [0.0, 2.0] # [m]
            
        class proprioception( Go1FieldCfg.sensor.proprioception ): # inherited from A1FieldCfg
            delay_action_obs = False
            delay_privileged_action_obs = False

    class terrain( Go1FieldCfg.terrain ):
        num_rows = 2
        # num_rows = 80
        num_cols = 1
        max_init_terrain_level = 1
        curriculum = False
        
        selected = "BarrierTrack"
        BarrierTrack_kwargs = dict(
            options= [
                # "tilt",
                "crawl",
                "jump",
                "jump",
                "leap",
            ], # each race track will permute all the options
            n_obstacles_per_track= 4,
            randomize_obstacle_order= True,
            track_width= 1.6,
            track_block_length= 2.0, # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.3), # [m]
            wall_height= (-0.5, 0.0), # [m]
            jump= dict(
                height= (0.40, 0.45),
                depth= (0.01, 0.01), # size along the forward axis
                fake_offset= 0.03, # [m] making the jump's height info greater than its physical height.
                jump_down_prob= 0.5,
            ),
            crawl= dict(
                height= (0.36, 0.5),
                depth= (0.1, 0.2), # size along the forward axis
                wall_height= 0.6,
                fake_depth= 0.4, # when track block length is 1.8m
            ),
            tilt= dict(
                width= (0.34, 0.36),
                depth= (0.4, 0.6), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                # length= (0.45, 0.7),
                length= (0.3, 0.74), # for parkour real-world env
                depth= (0.5, 0.6),
                height= 0.2,
                fake_offset= 0.1,
                follow_jump_ratio= 0.5, # when following jump, not drop down to ground suddenly.
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.3,
            check_skill_combinations= True,
            curriculum_perlin= False,
            no_perlin_threshold= 0.04, # for parkour real-world env
            walk_in_skill_gap= True, # for parkour real-world env
        )

        TerrainPerlin_kwargs = dict(
            zScale= [0.01, 0.1], # for parkour real-world env
            frequency= 10,
        )
        
    class commands( A1FieldDistillCfg.commands ):
        pass

    class control( Go1FieldCfg.control ):
        computer_clip_torque = False

    class asset( Go1FieldCfg.asset ):
        terminate_after_contacts_on = ["base"]

    class termination( A1FieldDistillCfg.termination ):
        out_of_track_kwargs = dict(
            threshold= 1.6,
        )

    class domain_rand( A1FieldDistillCfg.domain_rand ):
        randomize_com = True
        randomize_motor = True
        randomize_friction = True
        friction_range = [0.2, 2.0]
        randomize_base_mass = True
        push_robots = False
        init_dof_pos_ratio_range = [0.8, 1.2]
        init_dof_vel_range = [-2., 2.]
        # init_base_vel_range = [-0.0, 0.0] # use super class
        init_base_pos_range = dict(
            x= [0.4, 0.6],
            y= [-0.05, 0.05],
        )
        randomize_gravity_bias = True
        randomize_privileged_gravity_bias = False
        gravity_bias_range = dict(
            x= [-0.12, 0.12],
            y= [-0.12, 0.12],
            z= [-0.05, 0.05],
        )

    class rewards( A1FieldDistillCfg.rewards ):
        class scales:
            pass

    class noise( A1FieldDistillCfg.noise ):
        add_noise = True # This only account for proprioception noise and height measurements noise at most
        class noise_scales( A1FieldDistillCfg.noise.noise_scales ):
            ang_vel = 0.2 # measured in 0.02
            dof_pos = 0.0006 # measured in 0.0002
            dof_vel = 0.02 # measured in 0.015
            gravity = 0.06 # measured in 0.05

            # These aspects of obs should not have noise,
            # which are not used or set to zero or has its own noise mechanism
            height_measurements = 0.0
            forward_depth = 0.0
            base_pose = 0.0
            lin_vel = 0.0

        class forward_depth:
            stereo_min_distance = 0.12 # when using (240, 424) resolution
            stereo_far_distance = 2.
            stereo_far_noise_std = 0.08 # The noise std of pixels that are greater than stereo_far_noise_distance
            stereo_near_noise_std = 0.02 # The noise std of pixels that are less than stereo_far_noise_distance
            stereo_full_block_artifacts_prob = 0.004 # The probability of adding artifacts to pixels that are less than stereo_min_distance
            stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
            stereo_full_block_height_mean_std = [62, 1.5]
            stereo_full_block_width_mean_std = [3, 0.01]
            stereo_half_block_spark_prob = 0.02
            stereo_half_block_value = 3000 # to the maximum value directly
            sky_artifacts_prob = 0.0001
            sky_artifacts_far_distance = 2. # greater than this distance will be viewed as to the sky
            sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            sky_artifacts_height_mean_std = [2, 3.2]
            sky_artifacts_width_mean_std = [2, 3.2]

    class sim( Go1FieldCfg.sim ):
        no_camera = False

    class curriculum:
        # override base class attributes
        pass

# distill_target_ = "tanh"
distill_target_ = "l1"
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1FieldDistillCfgPPO( A1FieldDistillCfgPPO ):
    class algorithm( A1FieldDistillCfgPPO.algorithm ):
        entropy_coef = 0.0
        value_loss_coef = 0.0
        num_learning_epochs = 1 # 1 for finetuning, 2 for training from scratch
        num_mini_batches = 2
        teacher_act_prob = "exp"
        update_times_scale = 2000
        using_ppo = False
        distill_target = distill_target_
        buffer_dilation_ratio = 1.
        learning_rate = 1.e-4
        optimizer_class_name = "AdamW"

        teacher_policy_class_name = "ActorCriticClimbMutex"
        teacher_ac_path = None
        class teacher_policy (A1FieldDistillCfgPPO.policy ):
            # For loading teacher policy. No need to change for training student
            num_actor_obs = 81 # should equal to the sum of the obs_segments
            num_critic_obs = 81
            num_actions = 12
            obs_segments = OrderedDict(
                proprioception= (48,),
                base_pose= (6,),
                robot_config= (1 + 3 + 1 + 12,),
                engaging_block= (1 + (4 + 1) + 2,),
                sidewall_distance= (2,),
            )
            env_action_scale = Go1FieldCfg.control.action_scale
            action_smoothing_buffer_len = 3
            reset_non_selected = "when_skill"
            # reset_non_selected = True
            

            sub_policy_class_name = "ActorCriticRecurrent"
            
            sub_policy_paths = [
                # os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct10_00-03-12_WalkForward_pEnergySubsteps1e-5_maxPushAng0.5_pushInter3.0_noTanh_fromOct09_15-50-14"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct25_14-43-48_WalkForward_pEnergySubsteps2e-5_rTrackVel3e+0_rAlive3e+0_pTorqueL1norm8e-1_pDofLim8e-1_actionCliphard_fromOct24_09-00-16"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct27_20-22-36_WalkForward_pEnergySubsteps2e-5_rTrackVel3e+0_pYawAbs8e-1_pYPosAbs8e-01_pHipPos8e-01_noPropNoise_noTanh_actionCliphard_fromOct27_16-25-22"),
                os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-24-22_Skills_tilt_propDelay0.04-0.05_pEnergySubsteps1e-5_pPenD2e-3_pDofLimit8e-1_rTilt8e-03_pCollision0.1_PushRobot_kp40_kd0.5_tiltMax0.40fromSep27_13-59-27"),
                os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-19-00_Skills_crawl_propDelay0.04-0.05_pEnergy-1e-5_pDof8e-01_pTorqueL14e-01_pPosY0.1_maxPushAng0.3_kp40_fromOct09_09-58-26"),
                # os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct13_09-12-39_258k_Skills_jump_comXRange-0.2-0.2_pEnergySubsteps-4e-06_pushRobot_jumpHeight0.1-0.7_propDelay0.04-0.05_noTanh_fromOct09_10-13-26"),
                # os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct16_10-55-09_Skills_jump_comXRange-0.2-0.2_pLinVel1.4_pTorque2e-06_pY0.1_pCollision0.1_pushRobot_jumpHeight0.10-0.45_propDelay0.04-0.05_noTanh_fromOct13_09-12-39"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct22_15-19-26_Skills_jump_rTrackVel3._pJumpSameLegs0.6_noPContact_pDofAcc1.5e-07_rJumpXVel1.5_noPropNoise_propDelay0.04-0.05_pCollision-6e-01_noPush_minStd0.2_noTanh_jumpRange0.4-0.5_fromOct22_08-47-13"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct24_07-41-26_Skills_jump_rTrackVel4._pY-2e-01_pJumpSameLegs0.2_noPContact_pDofAcc2.5e-07_rJumpXVel1.5_noPropNoise_propDelay0.04-0.05_pCollision-1e-01_noPush_minStd0.2_noTanh_zeroResetAction_jumpRange0.4-0.5_fromOct20_16-09-47"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct25_11-36-16_Skills_jump_pEnergySubsteps3e-05_rTrackVel4._pY-3e-01_propDelay0.04-0.05_noPropNoise_noPush_noTanh_zeroResetAction_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct24_07-41-26"),
                # os.path.join(logs_root, "field_go1", "Oct27_16-03-48_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._pDofErrCond4e-01_pHipPos5e-01_pCollision3e+00_pSyncSymLegs4e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_13-20-59"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct27_16-04-45_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._pPenD-6e-03_pDofErrCond6e-01_pHipPos5e-01_pCollision6e+00_pSyncSymLegs4e-01_propDelay0.04-0.05_noPropNoise_pushRobot_noTanh_actionCliphard_noDelayActObs_jumpRange0.4-0.5_fromOct27_13-20-59"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct28_11-05-56_Skills_jump_pEnergySubsteps2e-06_rTrackVel5._pYaw-1e-01_pHipPos5e+00_pCollision1e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_jumpRange0.4-0.5_allowNegativeReward_fromOct28_08-54-23"),
                # os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct09_09-51-58_Skills_leap_propDelay0.04-0.05_pEnergySubsteps-8e-06_pPenD1.e-2_pDofLimit4e-01_pCollision0.5_kp40_kd0.5fromOct05_02-16-22"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct25_11-39-12_Skills_leap_pEnertySubsteps4.e-5_pActRate2.e-1_pDofLimit8.e-1_pCollision5.e-1_noPropNoise_noTanh_zeroResetAction_actionCliphard_fromOct09_09-51-58"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct28_13-23-39_Skills_leap_pEnertySubsteps2.e-5_rTrackVel2._rAlive2.0_pSyncAllLegs6.e-1_pOrient6.e-1_pHipPos5.e+0_noTanh_actionCliphard_virtual_fromOct28_03-32-35"),
            ]
            # jump_down_policy_path = logs_root + "/field_a1_noTanh_oracle/Oct09_09-49-58_Skills_down_pEnergySubsteps-4e-06_rAlive3_pPenD1e-02_pDofLimit-4e-01_pTorqueL14e-01_maxForce200_downRedundant0.2_softDof0.8_pushRobot_kp40_kd0.5fromSep28_14-59-53"
            # jump_down_policy_path = os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct14_02-06-47_Skills_down_comXRange-0.2-0.2_pEnergySubsteps-4e-06_rAlive3_pPenD4e-03_pDHarder2.e+05_pushRobot_noTanh_virtualfromOct09_09-49-58")
            # jump_down_policy_path = os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct25_10-48-54_down_pEnergySubsteps-4e-05_pDofLimit-4e-01_pYawAbs_pTorqueL14e-01_pActRate1.e-1_rDownCond3.e-1_kp40_kd0.5_virtualfromOct14_02-06-47")
            #tested OK, tends to stop
            # jump_down_policy_path = os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct25_11-36-31_down_pEnergySubsteps-4e-05_pDofLimit-4e-01_pYawAbs_pTorqueL14e-01_pActRate1.e-1_rDownCond1.e-2_kp40_kd0.5_virtualfromOct14_02-06-47")
            # jump_down_policy_path = os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct26_08-45-02_Skills_down_pEnergySubsteps-4e-05_rTrackVel4._pZVel1e-01_pAngXYVel5e-02_pDTorques1e-07_rDownCond1.e-1_noTanh_virtualfromOct25_10-48-54")
            jump_down_policy_path = os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct28_13-08-21_Skills_down_pEnergySubsteps-4e-05_rTrackVel5._pHipPos8e-01_rDownCond4.e-2_withPropNoise_noTanh_virtualfromOct28_07-38-39")
        
            # sub_policy_paths = [ # must in the order of obstacle ID
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct18_12-37-48_WalkForward_comXRange-0.2-0.2_noDelayActObs_pLinY0.05_noTanh_fromOct10_00-03-12"), # a little pitch up, accetable
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-24-22_Skills_tilt_propDelay0.04-0.05_pEnergySubsteps1e-5_pPenD2e-3_pDofLimit8e-1_rTilt8e-03_pCollision0.1_PushRobot_kp40_kd0.5_tiltMax0.40fromSep27_13-59-27"), # a little pitch up, accetable
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct18_12-32-28_Skills_crawl_comXRange-0.2-0.2_pEnergySubsteps-1e-5_pDof8e-01_pTorque1e-5_pTorqueL14e-01_noDelayActObs_noTanh_fromOct11_12-19-00"),
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct19_06-29-14_Skills_jump_comXRange-0.2-0.2_pLinVel1._pDofAcc5e-7_pTorque6e-05_jumpHeight0.10-0.46_noDelayActObs_noTanh_minStd0.1_fromOct18_15-57-31"),
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct19_01-25-22_Skills_leap_comXRange-0.2-0.2_pTorques2.e-5_pContactForces1.e-2_leapHeight0.1_noDelayActObs_noTanh_virtualfromOct09_09-51-58"),
            # ]
            # jump_down_policy_path = \
            #     os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct19_07-01-54_Skills_down_comXRange-0.2-0.2_pXVelL21.3_rDown0.3_pPenD2e-03_pDHarder2e+5_pTorque2e-5_pushRobot_noDelayActObs_noTanh_virtualfromOct18_00-51-27")

            # The group of ckpt that used for no Computer Clip and no Motor Clip
            sub_policy_paths = [
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct27_20-22-36_WalkForward_pEnergySubsteps2e-5_rTrackVel3e+0_pYawAbs8e-1_pYPosAbs8e-01_pHipPos8e-01_noPropNoise_noTanh_actionCliphard_fromOct27_16-25-22"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct27_20-22-36_WalkForward_pEnergySubsteps2e-5_rTrackVel3e+0_pYawAbs8e-1_pYPosAbs8e-01_pHipPos8e-01_noPropNoise_noTanh_actionCliphard_fromOct27_16-25-22"),
                os.path.join(logs_root, "field_a1_noTanh_oracle", "Oct11_12-19-00_Skills_crawl_propDelay0.04-0.05_pEnergy-1e-5_pDof8e-01_pTorqueL14e-01_pPosY0.1_maxPushAng0.3_kp40_fromOct09_09-58-26"),
                # os.path.join(logs_root, "field_go1", "Oct30_03-46-42_Skills_crawl_pEnergy1.e-5_pDof8.e-1_pTorqueL11.e+0_noComputerClip_noTanh"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct29_22-31-44_Skills_jump_pEnergySubsteps4e-05_rTrackVel4._pY-4e-01_pTorque4e-04_noJumpBonous_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_20-17-31"),
                # os.path.join(logs_root, "field_go1", "Oct30_11-11-43_Skills_jump_pEnergySubsteps6e-05_rTrackVel5._pY-8e-01_pTorqueExceed1.2e+00_pTorque4e-04_pDTorques1e-06_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_20-17-31"),
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct30_13-00-12_Skills_jump_pEnergySubsteps6e-05_rTrackVel5._pY-4e-01_pTorqueExceed1.8e+00_pTorque4e-04_pDTorques1e-05_propDelay0.04-0.05_noPropNoise_noPush_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct29_20-17-31"),
                ### New jump oracle with more spreaded rear hip.
                # os.path.join(logs_root, "field_go1_noTanh_oracle", "Nov02_11-07-49_Skills_jump_pEnergySubsteps5e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed1.8e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Nov02_13-53-48_Skills_jump_pEnergySubsteps5e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed1.5e+00_propDelay0.04-0.05_noPropNoise_pushRobot_minStd0.21_entropy0.01_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12"),
                # os.path.join(logs_root, "field_go1", "Oct30_05-08-03_Skills_leap_pEnertySubsteps6.e-6_rTrackVel5._pTorqueExceed4.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_noPenCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_04-10-12"),
                os.path.join(logs_root, "field_go1_noTanh_oracle", "Oct30_07-32-10_Skills_leap_pEnertySubsteps6.e-6_rTrackVel5._pTorqueExceed8.e-1_pPosY4.e-1_pTorques4.e-5_pHipPos5.e+0_pDorErr1.5e-1_noPenCurriculum_noCurriculum_noComputerClip_noTanh_noPush_allowNegativeReward_actionCliphard_virtual_fromOct30_05-07-24"),
            ]
            jump_down_policy_path = os.path.join(logs_root, "field_go1_noTanh_oracle", 
                "Oct30_03-42-56_Skills_down_pEnergySubsteps-1e-05_rTrackVel5._pTorqueL12e-01_pHipPos5e+00_rDownCond4.e-2_allowNegativeReward_pushRobot_noComputerClip_noPropNoise_noTanh_virtualfromOct29_22-07-51",
            )
            # jump_down_policy_path = os.path.join(logs_root, "field_go1", 
            #     "Oct30_04-23-31_Skills_down_pEnergySubsteps-1e-05_rTrackVel5._pTorqueL14e-01_pHipPos5e+00_rDownCond4.e-2_allowNegativeReward_pushRobot_noComputerClip_noPropNoise_noTanh_virtual_fromOct29_22-07-51",
            # )
            
            jump_down_vel = 1.0
            cmd_vel_mapping = {
                0: 1.0,
                1: 0.6,
                2: 1.0,
                3: 1.0,
                4: 1.5,
            }

    class policy( A1FieldDistillCfgPPO.policy ):
        pass

    runner_class_name = "TwoStageRunner"
    class runner( A1FieldDistillCfgPPO.runner ):
        experiment_name = "distill_go1"
        num_steps_per_env = 32

        class pretrain_dataset:
            # data_dir = [
            #     "/localdata_ssd/isaac_ziwenz_tmp/distill_go1_dagger/20231021_camPitch0.52_jumpA1Oct16_10-55-09/" + dir_ \
            #     for dir_ in os.listdir("/localdata_ssd/isaac_ziwenz_tmp/distill_go1_dagger/20231021_camPitch0.52_jumpA1Oct16_10-55-09")
            # ]
            scan_dir = "".join([
                "/localdata_ssd/isaac_ziwenz_tmp/distill_go1_dagger/", datetime.now().strftime('%b%d_%H-%M-%S'), "_",
                "".join(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]),
                "_vDelay{:.2f}-{:.2f}".format(
                    Go1FieldDistillCfg.sensor.forward_camera.latency_range[0],
                    Go1FieldDistillCfg.sensor.forward_camera.latency_range[1],
                ),
                "_pDelay{:.2f}-{:.2f}".format(
                    Go1FieldDistillCfg.sensor.proprioception.latency_range[0],
                    Go1FieldDistillCfg.sensor.proprioception.latency_range[1],
                ),
                ("_camPitch{:.2f}".format(
                    (Go1FieldDistillCfg.sensor.forward_camera.rotation["lower"][1] + Go1FieldDistillCfg.sensor.forward_camera.rotation["upper"][1]) / 2
                )),
                ("_depthMax{:.1f}".format(Go1FieldDistillCfg.sensor.forward_camera.depth_range[1])),
                # ("_zeroResetAction" if Go1FieldDistillCfg.init_state.zero_actions else ""),
                # ("_actionClip" + Go1FieldDistillCfg.normalization.clip_actions_method if getattr(Go1FieldDistillCfg.normalization, "clip_actions_method", None) else ""),
                ("_jumpOffset{:.2f}".format(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["jump"]["fake_offset"])),
                ("_noComputerClip" if not Go1FieldDistillCfg.control.computer_clip_torque else ""),
                ("_randOrder" if Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"] else ""),
            ])
            dataset_loops = -1
            random_shuffle_traj_order = True
            keep_latest_ratio = 1.0
            keep_latest_n_trajs = 3000
            starting_frame_range = [0, 100]

        resume = True
        # load_run = "/home/zzw/Data/legged_gym_logs/distill_a1/Jun06_00-15-50_distill_crawljumpleap_vDelay0.20-0.26_jump0.40-0.45_vFps10_leap0.55-0.62_fromJun04_20-18-56_06-3308JumpOracle"
        # load_run = "/home/zzw/Data/legged_gym_logs/distill_go1/Jul11_16-21-59_distill_crawljumpleap"
        # load_run = "Jul17_08-47-39_distill_jump_vDelay0.25-0.30"
        # load_run = "Oct08_02-51-14_distill_crawljumpjumpleap_vDelay0.25-0.30_noTanh_fromJul17_08-47-39"
        # load_run = "Oct08_14-51-22_distill_crawljumpjumpleap_vDelay0.25-0.30_noTanh_fromOct08_02-51-14"
        load_run = "Oct20_05-35-58_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_jumpA1Oct19_06-29-14_noTanh_fromOct08_14-51-22"
        load_run = "Oct21_09-01-45_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_jumpA1Oct16_10-55-09_noTanh_fromOct20_05-35-58"
        load_run = "Oct21_16-21-33_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_depthMax2.0_jumpA1Oct16_10-55-09_noTanh_fromOct21_09-01-45"
        load_run = "Oct22_02-25-42_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_depthMax2.0_jumpA1Oct16_10-55-09_noTanh_fromOct21_16-21-33"
        load_run = "Oct24_15-43-34_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_depthMax2.0_zeroResetAction_actionCliphard_jumpOct24_07-41-26_noTanh_fromOct22_02-25-42"
        load_run = "Oct25_17-10-09_distill_crawljumpjumpleap_withDown_vDelay0.25-0.30_camPitch0.52_depthMax2.0_actionCliphard_jumpOct25_11-36-16_downOct25_11-36-31_noTanh_fromOct24_15-43-34"
        #### From wrong walk and jump skill policy
        # load_run = "Oct26_15-38-28_distill_crawljumpjumpleap_vDelay0.25-0.30_camPitch0.52_depthMax2.0_noPropNoise_gravityBias_actionCliphard_jumpOct25_11-36-16_downOct25_11-36-31_noTanh_fromOct25_17-10-09"
        # load_run = "Oct27_06-00-07_distill_crawljumpjumpleap_vDelay0.25-0.30_camPitch0.52_depthMax2.0_noPropNoise_gravityBias_actionCliphard_jumpOct25_11-36-16_downOctOct26_08-45-02_fricMax2.0_noTanh_fromOct26_15-38-28"
        load_run = "Oct28_16-16-58_distill_crawljumpjumpleap_vDelay0.25-0.30_camPitch0.52_depthMax2.0_addPropNoise_gravityBias_actionCliphard_jumpOct28_11-05-56_fricMax2.0_noTanh_fromOct25_17-10-09"
        ######## Continue tuning policy to fix jump and leap.
        ## This ckpt looks good. But the jump is not good enough.
        load_run = "Oct30_16-31-09_distill_crawljumpjumpleap_vDelay0.25-0.30_camPitch0.52_depthMax2.0_addPropNoise_gravityBias_actionCliphard_noComputerClip_oracleResetWhenSkill_fricMax2.0_noTanh_fromOct28_16-16-58"

        max_iterations = 100000
        save_interval = 2000
        log_interval = 100
        run_name = "".join(["distill_",
            "".join(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]),
            "_vDelay{:.2f}-{:.2f}".format(
                Go1FieldDistillCfg.sensor.forward_camera.latency_range[0],
                Go1FieldDistillCfg.sensor.forward_camera.latency_range[1],
            ),
            ("_camPitch{:.2f}".format(
                (Go1FieldDistillCfg.sensor.forward_camera.rotation["lower"][1] + Go1FieldDistillCfg.sensor.forward_camera.rotation["upper"][1]) / 2
            )),
            ("_depthMax{:.1f}".format(Go1FieldDistillCfg.sensor.forward_camera.depth_range[1])),
            # ("_addPropNoise" if Go1FieldDistillCfg.noise.add_noise else "_noPropNoise"),
            # ("_gravityBias" if Go1FieldDistillCfg.domain_rand.randomize_gravity_bias else "_noGravityBias"),
            # ("_actionClip" + Go1FieldDistillCfg.normalization.clip_actions_method if getattr(Go1FieldDistillCfg.normalization, "clip_actions_method", None) else ""),
            # ("_noComputerClip" if not Go1FieldDistillCfg.control.computer_clip_torque else ""),
            ("_jumpNov02_13-53-48"),
            ("_leapVel1.5"),
            ("_jumpOffset{:.2f}".format(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["jump"]["fake_offset"])),
            ("_oracleResetWhenSkill"),
            ("_fricMax{:.1f}".format(Go1FieldDistillCfg.domain_rand.friction_range[1])),
            ("_noTanh"),
            ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2]) if resume else "_noResume"),
        ])

        # Checking variants:
        """
        1. Faster leap (1.5m/s)
            higher offset 0.1 for jump up
        2. Longer leap offset 0.2, faster leap (1.5m/s)
            higher offset 0.1 for jump up
        3. Faster leap (1.5m/s)
            higher offset 0.1 for jump up
            from older ckpt
        4. Faster leap (1.5m/s)
            faster jump (1.2m/s) and higher offset 0.1 for jump up
            from older ckpt
            smaller learning rate 1e-4
        ---- 20231101 after testing preious ckpts ----
        5. Faster leap (1.5m/s)
            normal 0.05 offset for jump up, normal (1.0m/s) jump
            from older ckpt (Oct28_16-16-58)
            same learning rate as Oct30_16-31-09
        6. Faster leap (1.5m/s)
            slight higher 0.07 offset for jump up, faster (1.2m/s) jump
            from older ckpt (Oct28_16-16-58)
            lower learning rate as 1e-4
        ---- 20231102 after testing preious ckpts ----
        New jump oracle to test:
            * Nov02_11-07-49_Skills_jump_pEnergySubsteps5e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed1.8e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12
            Nov02_11-09-03_Skills_jump_pEnergySubsteps4e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed2e+00_propDelay0.04-0.05_noPropNoise_pushRobot_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12
            Nov02_13-53-13_Skills_jump_pEnergySubsteps5e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed1.5e+00_propDelay0.04-0.05_noPropNoise_pushRobot_minStd0.22_entropy0.01_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12
            * Nov02_13-53-48_Skills_jump_pEnergySubsteps5e-05_rTrackVel5._pFHipPos5e+00_pTorqueExceed1.5e+00_propDelay0.04-0.05_noPropNoise_pushRobot_minStd0.21_entropy0.01_gamma0.999_noTanh_noComputerClip_jumpRange0.2-0.5_allowNegativeReward_fromOct30_13-00-12
        7. Faster leap (1.5m/s)
            lower 0.05 offset for jump up, normal (1.0m/s) jump
            using new jump oracle
            same older ckpt (Oct28_16-16-58)
            same learning rate as Oct30_16-31-09 (1.5e-4)
        8. Faster leap (1.5m/s)
            lower 0.03 offset for jump up, normal (1.0m/s) jump
            using new jump oracle
            new ckpt (Oct30_16-31-09)
            lower learning rate as 1e-4
        """

                
    
