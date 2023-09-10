from collections import OrderedDict
import os
import os.path as osp
from datetime import datetime
import numpy as np
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1FieldDistillCfg( A1FieldCfg ):
    class env( A1FieldCfg.env ):
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
        privileged_obs_gets_privilege = False

    class init_state( A1FieldCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]

    class sensor( A1FieldCfg.sensor ):
        class forward_camera( A1FieldCfg.sensor.forward_camera ):
            resolution = [int(240/4), int(424/4)]
            position = dict(
                mean= [0.27, 0.0075, 0.033],
                std= [0.01, 0.0025, 0.0005],
            ) # position in base_link ##### small randomization
            rotation = dict(
                lower= [0, 0, 0],
                upper= [0, 5 * np.pi / 180, 0],
            ) # rotation in base_link ##### small randomization
            resized_resolution= [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [85, 88]

            # adding randomized latency
            latency_range = [0.2, 0.26] # for [16, 32, 32] -> 128 -> 128 visual model in (240, 424 option)
            latency_resample_time = 5.0 # [s]
            refresh_duration = 1/10 # [s] for (240, 424 option with onboard script fixed to no more than 20Hz)

            # config to simulate stero RGBD camera
            crop_top_bottom = [0, 0]
            crop_left_right = [int(60/4), int(46/4)]
            depth_range = [0.0, 1.5] # [m]

        class proprioception:
            delay_action_obs = True
            latency_range = [0.04-0.0025, 0.04+0.0075] # [min, max] in seconds
            latency_resample_time = 2.0 # [s]

    class terrain( A1FieldCfg.terrain ):
        num_rows = 2
        # num_rows = 80
        num_cols = 1
        max_init_terrain_level = 1
        curriculum = False
        
        selected = "BarrierTrack"
        BarrierTrack_kwargs = dict(
            options= [
                "tilt",
                "crawl",
                "climb",
                "climb",
                "leap",
            ], # each race track will permute all the options
            n_obstacles_per_track= 5,
            randomize_obstacle_order= True,
            track_width= 3.0,
            track_block_length= 1.8, # the x-axis distance from the env origin point
            wall_thickness= (0.2, 1.), # [m]
            wall_height= (-0.5, 0.5), # [m]
            climb= dict(
                height= (0.2, 0.6),
                # height= (0.1, 0.5),
                depth= (0.1, 0.2), # size along the forward axis
                fake_offset= 0.0, # [m] making the climb's height info greater than its physical height.
                climb_down_prob= 0.5,
            ),
            crawl= dict(
                # height= (0.28, 0.4),
                height= (0.3, 0.5),
                depth= (0.1, 0.4), # size along the forward axis
                wall_height= (0.1, 0.6),
            ),
            tilt= dict(
                width= (0.305, 0.4),
                depth= (0.4, 0.6), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.3, 0.7), # for parkour real-world env
                depth= (0.5, 0.6),
                height= 0.2,
                fake_offset= 0.2,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= -0.5,
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
        
    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            # student not react to command input
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control( A1FieldCfg.control ):
        stiffness = {'joint': 50.}
        damping = {'joint': 1.}
        action_scale = 0.5
        torque_limits = 25.
        computer_clip_torque = True # for climb, walk temporarily
        motor_clip_torque = False # for climb, walk temporarily

    class termination( A1FieldCfg.termination ):
        termination_terms = [
            "roll",
            "pitch",
            "out_of_track",
        ]
        roll_kwargs = merge_dict(A1FieldCfg.termination.pitch_kwargs, dict(
            threshold= 0.8, # [rad] # for tilt
            crawl_threshold= 0.4,
            climb_threshold= 0.8,
            leap_threshold= 0.6,
            tilt_threshold= 1.0, # for tilt (condition on engaging block)
        ))
        pitch_kwargs = merge_dict(A1FieldCfg.termination.pitch_kwargs, dict(
            threshold= 1.5,
            crawl_threshold= 0.7,
            climb_threshold= 1.5,
            leap_threshold= 0.7,
            tilt_threshold= 0.5,
        ))
        out_of_track_kwargs = dict(
            threshold= 1., # [m] NOTE: change according to skill demonstration
        )
        timeout_at_border = True
        timeout_at_finished = True

    class domain_rand( A1FieldCfg.domain_rand ):
        randomize_com = True
        class com_range( A1FieldCfg.domain_rand.com_range ):
            x = [0.05, 0.15]
        randomize_motor = True
        randomize_friction = True
        friction_range = [0.0, 0.8]
        randomize_base_mass = True
        # added_mass_range = [-1.0, 1.0]
        push_robots = False
        init_dof_pos_ratio_range = [0.9, 1.1]
        init_base_vel_range = [-0.0, 0.0]
        init_base_pos_range = dict(
            x= [0.4, 0.6],
            y= [-0.05, 0.05],
        )

    # not used in the algorithm, only for visualizing
    class rewards( A1FieldCfg.rewards ):
        class scales:
            pass

    class noise( A1FieldCfg.noise ):
        class noise_scales( A1FieldCfg.noise.noise_scales ):
            forward_depth = 0.0
        class forward_depth:
            stereo_min_distance = 0.12 # when using (240, 424) resolution
            stereo_far_distance = 2.
            stereo_far_noise_std = 0.08 # The noise std of pixels that are greater than stereo_far_noise_distance
            stereo_near_noise_std = 0.02 # The noise std of pixels that are less than stereo_far_noise_distance
            stereo_full_block_artifacts_prob = 0.008 # The probability of adding artifacts to pixels that are less than stereo_min_distance
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

    class viewer( A1FieldCfg.viewer ):
        pos = [16, 5, 3.2]  # [m]
        lookat = [16, 8., 2.]  # [m]

    class sim( A1FieldCfg.sim ):
        no_camera = False

    class curriculum:
        # override base class attributes
        pass

distill_target_ = "tanh"
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1FieldDistillCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
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

        teacher_policy_class_name = "ActorCriticClimbMutex"
        teacher_ac_path = None
        class teacher_policy (A1FieldCfgPPO.policy ):
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
            env_action_scale = A1FieldCfg.control.action_scale

            sub_policy_class_name = "ActorCriticRecurrent"
            sub_policy_paths = [ # must in the order of obstacle ID, Replace the folder name with your own training logdir
                os.path.join(logs_root, "field_a1/{your walking policy}"),
                os.path.join(logs_root, "field_a1/{your tilting policy}"),
                os.path.join(logs_root, "field_a1/{your crawling policy}"),
                os.path.join(logs_root, "field_a1/{your climbing policy}"),
                os.path.join(logs_root, "field_a1/{your leaping policy}"),
            ]
            climb_down_policy_path = os.path.join(logs_root, "field_a1/{your climbing down policy}")
            cmd_vel_mapping = {
                0: 1.0,
                1: 0.5,
                2: 0.8,
                3: 1.2,
                4: 1.5,
            }
    
    class policy( A1FieldCfgPPO.policy ):
        class visual_kwargs:
            channels = [16, 32, 32]
            kernel_sizes = [5, 4, 3]
            strides = [2, 2, 1]
            hidden_sizes = [128]
            use_maxpool = True
            nonlinearity = "LeakyReLU"
        visual_latent_size = 128
        init_noise_std = 0.05

    runner_class_name = "TwoStageRunner"
    class runner( A1FieldCfgPPO.runner ):
        policy_class_name = "VisualDeterministicRecurrent"
        algorithm_class_name = "TPPO"
        experiment_name = "distill_a1"
        num_steps_per_env = 48

        # configs for training using collected dataset
        pretrain_iterations = -1 # negative value for infinite training
        class pretrain_dataset:
            scan_dir = "".join([
                "logs/distill_a1_dagger/", datetime.now().strftime('%b%d_%H-%M-%S'), "_",
                "".join(A1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]),
                "_vDelay{:.2f}-{:.2f}".format(
                    A1FieldDistillCfg.sensor.forward_camera.latency_range[0],
                    A1FieldDistillCfg.sensor.forward_camera.latency_range[1],
                ),
                "_pDelay{:.2f}-{:.2f}".format(
                    A1FieldDistillCfg.sensor.proprioception.latency_range[0],
                    A1FieldDistillCfg.sensor.proprioception.latency_range[1],
                ),
                ("_randOrder" if A1FieldDistillCfg.terrain.BarrierTrack_kwargs["randomize_obstacle_order"] else ""),
                ("_noPerlinRate{:.1f}".format(
                    (A1FieldDistillCfg.terrain.BarrierTrack_kwargs["no_perlin_threshold"] - A1FieldDistillCfg.terrain.TerrainPerlin_kwargs["zScale"][0]) / \
                    (A1FieldDistillCfg.terrain.TerrainPerlin_kwargs["zScale"][1] - A1FieldDistillCfg.terrain.TerrainPerlin_kwargs["zScale"][0])
                )),
            ])
            dataset_loops = -1 # negative value for infinite dataset loops
            
            random_shuffle_traj_order = True
            keep_latest_ratio = 1.0
            keep_latest_n_trajs = 2000
            starting_frame_range = [0, 100]

        resume = False
        load_run = None

        max_iterations = 80000
        save_interval = 2000

        run_name = "".join(["distill", #"_opensource",
        ("_" + ("".join(A1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]) if len(A1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"]) > 1 else A1FieldDistillCfg.terrain.BarrierTrack_kwargs["options"][0])),
        ("_vDelay{:.2f}-{:.2f}".format(*A1FieldDistillCfg.sensor.forward_camera.latency_range)),
        ("_pDelay{:.2f}-{:.2f}".format(*A1FieldDistillCfg.sensor.proprioception.latency_range)),
        ("_randOrder" if A1FieldDistillCfg.terrain.BarrierTrack_kwargs["randomize_obstacle_order"] else ""),
        ("_from" + "_".join(load_run.split("_")[:2]) if resume else ""),
        ])
