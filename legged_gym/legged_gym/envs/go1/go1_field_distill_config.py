from collections import OrderedDict
import os
import datetime
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
            "forward_depth",
        ]
        privileged_obs_components = [
            "proprioception", # 48
            "base_pose",
            "robot_config",
            "engaging_block",
            "sidewall_distance",
        ]
        use_lin_vel = False
        privileged_use_lin_vel = True
        # if True privileged_obs will not have noise based on implementations
        privileged_obs_gets_privilege = False # for climb temporarily

    class init_state( Go1FieldCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]

    class sensor( Go1FieldCfg.sensor ):
        class forward_camera( Go1FieldCfg.sensor.forward_camera ):
            resolution = [int(240/4), int(424/4)]
            position = dict(
                mean= [0.245, 0.0075, 0.072+0.018],
                std= [0.002, 0.002, 0.0005],
            ) # position in base_link ##### small randomization
            rotation = dict(
                lower= [0, 0, 0],
                upper= [0, 0, 0],
            ) # rotation in base_link ##### small randomization
            resized_resolution= [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [85, 87] # measured around 87 degree
            # for go1, usb2.0, 480x640, d435i camera
            latency_range = [0.25, 0.30]
            latency_resample_time = 5.0 # [s]
            refresh_duration = 1/10 # [s] for (240, 424 option with onboard script fixed to no more than 20Hz)
            
            # config to simulate stero RGBD camera
            crop_top_bottom = [0, 0]
            crop_left_right = [int(60/4), int(46/4)]
            depth_range = [0.0, 1.5] # [m]
            
        # class proprioception( Go1FieldCfg.sensor.proprioception ): # inherited from A1FieldCfg

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
                "climb",
                "leap",
            ], # each race track will permute all the options
            one_obstacle_per_track= True,
            track_width= 1.6,
            track_block_length= 1.8, # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= 0.0, # [m]
            climb= dict(
                height= (0.40, 0.45),
                depth= (0.5, 1.0), # size along the forward axis
                fake_offset= 0.05, # [m] making the climb's height info greater than its physical height.
            ),
            crawl= dict(
                height= (0.28, 0.38),
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
                length= (0.5, 0.7), # for parkour real-world env
                depth= (0.5, 0.6),
                height= 0.2,
                fake_offset= 0.2,
                follow_climb_ratio= 0.5, # when following climb, not drop down to ground suddenly.
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            check_skill_combinations= True,
            curriculum_perlin= False,
            no_perlin_threshold= 0.04, # for parkour real-world env
            walk_in_skill_gap= True, # for parkour real-world env
        )

        TerrainPerlin_kwargs = dict(
            zScale= [0.0, 0.05], # for parkour real-world env
            frequency= 10,
        )
        
    class commands( A1FieldDistillCfg.commands ):
        pass

    class termination( A1FieldDistillCfg.termination ):
        pass

    class domain_rand( A1FieldDistillCfg.domain_rand ):
        randomize_com = True
        randomize_motor = True
        randomize_friction = True
        friction_range = [0.0, 0.8]
        randomize_base_mass = True
        push_robots = False
        init_dof_pos_ratio_range = [0.9, 1.1]
        init_base_vel_range = [-0.0, 0.0]
        init_base_pos_range = dict(
            x= [0.4, 0.6],
            y= [-0.05, 0.05],
        )

    class noise( A1FieldDistillCfg.noise ):
        class noise_scales( A1FieldDistillCfg.noise.noise_scales ):
            forward_depth = 0.0
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

distill_target_ = "tanh"
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
        learning_rate = 3e-4
        optimizer_class_name = "AdamW"

        teacher_policy_class_name = "ActorCriticFieldMutex"
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

            sub_policy_class_name = "ActorCriticRecurrent"
            sub_policy_paths = [ # must in the order of obstacle ID
                logs_root + "/field_go1/{your go1 walking policy}",
                logs_root + "/field_go1/{your go1 tilting policy}",
                logs_root + "/field_go1/{your go1 crawling policy}",
                logs_root + "/field_go1/{your go1 climbing policy}",
                logs_root + "/field_go1/{your go1 leaping policy}",
            ]
            cmd_vel_mapping = {
                0: 1.0,
                1: 0.5,
                2: 0.8,
                3: 1.2,
                4: 1.5,
            }

    class policy( A1FieldDistillCfgPPO.policy ):
        pass

    runner_class_name = "TwoStageRunner"
    class runner( A1FieldDistillCfgPPO.runner ):
        experiment_name = "distill_go1"

        class pretrain_dataset:
            # data_dir = [
            #     "logs/distill_a1_dagger/" + dir_ \
            #     for dir_ in os.listdir("logs/distill_a1_dagger")
            # ]
            scan_dir = "".join([
                "logs/distill_go1_dagger/{}_".format(datetime.datetime.now().strftime("%b%d")),
                "".join(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]),
                "_vDelay{:.2f}-{:.2f}".format(
                    Go1FieldDistillCfg.sensor.forward_camera.latency_range[0],
                    Go1FieldDistillCfg.sensor.forward_camera.latency_range[1],
                ),
            ])
            dataset_loops = -1
            random_shuffle_traj_order = True
            keep_latest_ratio = 1.0
            keep_latest_n_trajs = 4000
            starting_frame_range = [0, 100]

        resume = False
        load_run = ""

        max_iterations = 80000
        save_interval = 2000
        run_name = "".join(["distill_",
            "".join(Go1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]),
            "_vDelay{:.2f}-{:.2f}".format(
                Go1FieldDistillCfg.sensor.forward_camera.latency_range[0],
                Go1FieldDistillCfg.sensor.forward_camera.latency_range[1],
            ),
        ])

                
    
