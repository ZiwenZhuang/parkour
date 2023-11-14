""" The script to collect demonstrations for the legged robot """
import isaacgym
from collections import OrderedDict
import torch
from datetime import datetime
import numpy as np
import os
import json
import os.path as osp

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, update_class_from_dict
from legged_gym.debugger import break_into_debugger

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.dagger_saver import DemonstrationSaver, DaggerSaver

def main(args):
    RunnerCls = DaggerSaver
    # RunnerCls = DemonstrationSaver
    success_traj_only = False
    teacher_act_prob = 0.1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if RunnerCls == DaggerSaver:
        with open(os.path.join("logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
            d = json.load(f, object_pairs_hook= OrderedDict)
            update_class_from_dict(env_cfg, d, strict= True)
            update_class_from_dict(train_cfg, d, strict= True)

    # Some custom settings
    # ####### customized option to increase data distribution #######
    action_sample_std = 0.0
    ############# some predefined options #############
    if len(env_cfg.terrain.BarrierTrack_kwargs["options"]) == 1:
        env_cfg.terrain.num_rows = 20; env_cfg.terrain.num_cols = 30
    else: # for parkour env
        # >>> option 1
        env_cfg.terrain.BarrierTrack_kwargs["track_block_length"] = 2.8
        env_cfg.terrain.BarrierTrack_kwargs["track_width"] = 2.4
        env_cfg.terrain.BarrierTrack_kwargs["wall_thickness"] = (0.0, 0.6)
        env_cfg.domain_rand.init_base_pos_range["x"] = (0.4, 1.8)
        env_cfg.terrain.num_rows = 12; env_cfg.terrain.num_cols = 10
        # >>> option 2
        # env_cfg.terrain.BarrierTrack_kwargs["track_block_length"] = 3.
        # env_cfg.terrain.BarrierTrack_kwargs["track_width"] = 4.0
        # env_cfg.terrain.BarrierTrack_kwargs["wall_height"] = (-0.5, -0.2)
        # env_cfg.terrain.BarrierTrack_kwargs["wall_thickness"] = (0.0, 1.4)
        # env_cfg.domain_rand.init_base_pos_range["x"] = (1.6, 2.0)
        # env_cfg.terrain.num_rows = 16; env_cfg.terrain.num_cols = 5
        # >>> option 3
        # env_cfg.terrain.BarrierTrack_kwargs["track_block_length"] = 1.6
        # env_cfg.terrain.BarrierTrack_kwargs["track_width"] = 2.2
        # env_cfg.terrain.BarrierTrack_kwargs["wall_height"] = (-0.5, 0.1)
        # env_cfg.terrain.BarrierTrack_kwargs["wall_thickness"] = (0.0, 0.5)
        # env_cfg.domain_rand.init_base_pos_range["x"] = (0.2, 0.9)
        # env_cfg.terrain.BarrierTrack_kwargs["n_obstacles_per_track"] = 1
        # action_sample_std = 0.1
        # env_cfg.terrain.num_rows = 22; env_cfg.terrain.num_cols = 16
        pass
    if (env_cfg.terrain.BarrierTrack_kwargs["options"][0] == "leap") and all(i  == env_cfg.terrain.BarrierTrack_kwargs["options"][0] for i in env_cfg.terrain.BarrierTrack_kwargs["options"]):
        ######### For leap, because the platform is usually higher than the ground.
        env_cfg.terrain.num_rows = 80
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.BarrierTrack_kwargs["track_width"] = 1.6
        env_cfg.terrain.BarrierTrack_kwargs["wall_thickness"] = (0.01, 0.5)
        env_cfg.terrain.BarrierTrack_kwargs["wall_height"] = (-0.4, 0.2) # randomize incase of terrain that have side walls
        env_cfg.terrain.BarrierTrack_kwargs["border_height"] = -0.4
    # Done custom settings

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    config = class_to_dict(train_cfg)
    config.update(class_to_dict(env_cfg))

    # create teacher policy
    policy = build_actor_critic(
        env,
        train_cfg.algorithm.teacher_policy_class_name,
        config["algorithm"]["teacher_policy"],
    ).to(env.device)
    # load the policy is possible
    if train_cfg.algorithm.teacher_ac_path is not None:
        state_dict = torch.load(train_cfg.algorithm.teacher_ac_path, map_location= "cpu")
        teacher_actor_critic_state_dict = state_dict["model_state_dict"]
        policy.load_state_dict(teacher_actor_critic_state_dict)

    # build runner
    track_header = "".join(env_cfg.terrain.BarrierTrack_kwargs["options"])
    if env_cfg.commands.ranges.lin_vel_x[1] > 0.0:
        cmd_vel = "_cmd{:.1f}-{:.1f}".format(env_cfg.commands.ranges.lin_vel_x[0], env_cfg.commands.ranges.lin_vel_x[1])
    elif env_cfg.commands.ranges.lin_vel_x[1] == 0. and len(env_cfg.terrain.BarrierTrack_kwargs["options"]) == 1 \
        or (env_cfg.terrain.BarrierTrack_kwargs["options"][0] == env_cfg.terrain.BarrierTrack_kwargs["options"][1]):
        obstacle_id = env.terrain.track_options_id_dict[env_cfg.terrain.BarrierTrack_kwargs["options"][0]]
        try:
            overrided_vel = train_cfg.algorithm.teacher_policy.cmd_vel_mapping[obstacle_id]
        except:
            overrided_vel = train_cfg.algorithm.teacher_policy.cmd_vel_mapping[str(obstacle_id)]
        cmd_vel = "_cmdOverride{:.1f}".format(overrided_vel)
    else:
        cmd_vel = "_cmdMutex"

    datadir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "logs")
    runner_kwargs = dict(
        env= env,
        policy= policy,
        save_dir= osp.join(
            train_cfg.runner.pretrain_dataset.scan_dir if RunnerCls == DaggerSaver else osp.join(datadir, "distill_{}_dagger".format(args.task.split("_")[0])),
            datetime.now().strftime('%b%d_%H-%M-%S') + "_" + "".join([
                track_header,
                cmd_vel,
                "_lowBorder" if env_cfg.terrain.BarrierTrack_kwargs["border_height"] < 0 else "",
                "_addMassMin{:.1f}".format(env_cfg.domain_rand.added_mass_range[0]) if env_cfg.domain_rand.added_mass_range[0] > 1. else "",
                "_comMean{:.2f}".format((env_cfg.domain_rand.com_range.x[0] + env_cfg.domain_rand.com_range.x[1])/2),
                "_1cols" if env_cfg.terrain.num_cols == 1 else "",
                "_randOrder" if env_cfg.terrain.BarrierTrack_kwargs.get("randomize_obstacle_order", False) else "",
                ("_noPerlinRate{:.1f}".format(
                    (env_cfg.terrain.BarrierTrack_kwargs["no_perlin_threshold"] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0]) / \
                    (env_cfg.terrain.TerrainPerlin_kwargs["zScale"][1] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0])
                )),
                ("_fric{:.1f}-{:.1f}".format(*env_cfg.domain_rand.friction_range)),
                "_successOnly" if success_traj_only else "",
                "_aStd{:.2f}".format(action_sample_std) if (action_sample_std > 0. and RunnerCls == DaggerSaver) else "",
            ] + ([] if RunnerCls == DemonstrationSaver else ["_" + "_".join(args.load_run.split("_")[:2])])
            ),
        ),
        rollout_storage_length= 256,
        min_timesteps= 1e6, # 1e6,
        min_episodes= 2e4 if RunnerCls == DaggerSaver else 2e-3,
        use_critic_obs= True,
        success_traj_only= success_traj_only,
        obs_disassemble_mapping= dict(
            forward_depth= "normalized_image",
        ),
    )
    if RunnerCls == DaggerSaver:
        # kwargs for dagger saver
        runner_kwargs.update(dict(
            training_policy_logdir= osp.join(
                "logs",
                train_cfg.runner.experiment_name,
                args.load_run,
            ),
            teacher_act_prob= teacher_act_prob,
            update_times_scale= config["algorithm"]["update_times_scale"],
            action_sample_std= action_sample_std,
        ))
    runner = RunnerCls(**runner_kwargs)
    runner.collect_and_save(config= config)

if __name__ == "__main__":
    args = get_args()
    main(args)
