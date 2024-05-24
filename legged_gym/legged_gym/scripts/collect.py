""" The script to collect demonstrations for the legged robot """
import isaacgym
from collections import OrderedDict
import torch
from datetime import datetime
import numpy as np
import os
import json
import os.path as osp

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, update_class_from_dict
from legged_gym.debugger import break_into_debugger

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.dagger_saver import DemonstrationSaver, DaggerSaver

def main(args):
    RunnerCls = DaggerSaver if args.load_run else DemonstrationSaver
    success_traj_only = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if RunnerCls == DaggerSaver:
        with open(os.path.join("logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
            d = json.load(f, object_pairs_hook= OrderedDict)
            update_class_from_dict(env_cfg, d, strict= True)
            update_class_from_dict(train_cfg, d, strict= True)
    
    ####### customized option to increase data distribution #######
    # env_cfg.env.num_envs = 6
    # env_cfg.terrain.curriculum = True
    # env_cfg.terrain.max_init_terrain_level = 0
    # env_cfg.terrain.border_size = 1.
    ############# some predefined options #############
    env_cfg.terrain.num_rows = 8; env_cfg.terrain.num_cols = 40
    # Done custom settings

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    config = class_to_dict(train_cfg)
    config.update(class_to_dict(env_cfg))
    teacher_act_prob = config["algorithm"]["teacher_act_prob"] if args.teacher_prob is None else args.teacher_prob
    action_std = config["policy"]["init_noise_std"] if args.action_std is None else args.action_std

    # create teacher policy
    policy = build_actor_critic(
        env,
        config["algorithm"]["teacher_policy_class_name"],
        config["algorithm"]["teacher_policy"],
    ).to(env.device)
    # load the policy is possible
    if config["algorithm"]["teacher_ac_path"] is not None:
        if "{LEGGED_GYM_ROOT_DIR}" in config["algorithm"]["teacher_ac_path"]:
            config["algorithm"]["teacher_ac_path"] = config["algorithm"]["teacher_ac_path"].format(LEGGED_GYM_ROOT_DIR= LEGGED_GYM_ROOT_DIR)
        state_dict = torch.load(config["algorithm"]["teacher_ac_path"], map_location= "cpu")
        teacher_actor_critic_state_dict = state_dict["model_state_dict"]
        policy.load_state_dict(teacher_actor_critic_state_dict)

    # build runner
    track_header = "".join(env_cfg.terrain.BarrierTrack_kwargs["options"])
    datadir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "logs")
    runner_kwargs = dict(
        env= env,
        policy= policy,
        save_dir= osp.join(
            config["runner"]["pretrain_dataset"]["data_dir"] if RunnerCls == DaggerSaver else osp.join("/localdata_ssd/zzw/athletic-isaac_tmp", "{}_dagger".format(config["runner"]["experiment_name"])),
            datetime.now().strftime('%b%d_%H-%M-%S') + "_" + "".join([
                track_header,
                "_lowBorder" if env_cfg.terrain.BarrierTrack_kwargs["border_height"] < 0 else "",
                "_trackWidth{:.1f}".format(env_cfg.terrain.BarrierTrack_kwargs["track_width"]) if env_cfg.terrain.BarrierTrack_kwargs["track_width"] < 1.8 else "",
                "_blockLength{:.1f}".format(env_cfg.terrain.BarrierTrack_kwargs["track_block_length"]) if env_cfg.terrain.BarrierTrack_kwargs["track_block_length"] > 1.6 else "",
                "_addMassMin{:.1f}".format(env_cfg.domain_rand.added_mass_range[0]) if env_cfg.domain_rand.added_mass_range[0] > 1. else "",
                "_teacherProb{:.1f}".format(teacher_act_prob),
                "_randOrder" if env_cfg.terrain.BarrierTrack_kwargs.get("randomize_obstacle_order", False) else "",
                ("_noPerlinRate{:.1f}".format(
                    (env_cfg.terrain.BarrierTrack_kwargs["no_perlin_threshold"] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0]) / \
                    (env_cfg.terrain.TerrainPerlin_kwargs["zScale"][1] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0])
                ) if isinstance(env_cfg.terrain.TerrainPerlin_kwargs["zScale"], (tuple, list)) else ""),
                ("_fric{:.1f}-{:.1f}".format(*env_cfg.domain_rand.friction_range)),
                "_successOnly" if success_traj_only else "",
                "_aStd{:.2f}".format(action_std) if (action_std > 0. and RunnerCls == DaggerSaver) else "",
            ] + ([] if RunnerCls == DemonstrationSaver else ["_" + "_".join(args.load_run.split("_")[:2])])
            ),
        ),
        rollout_storage_length= 256,
        min_timesteps= 1e9, # 1e6,
        min_episodes= 1e6 if RunnerCls == DaggerSaver else 2e5,
        use_critic_obs= True,
        success_traj_only= success_traj_only,
        obs_disassemble_mapping= dict(
            forward_depth= "normalized_image",
        ),
        demo_by_sample= config["algorithm"].get("action_labels_from_sample", False),
    )
    if RunnerCls == DaggerSaver:
        # kwargs for dagger saver
        runner_kwargs.update(dict(
            training_policy_logdir= osp.join(
                "logs",
                config["runner"]["experiment_name"],
                args.load_run,
            ),
            teacher_act_prob= teacher_act_prob,
            update_times_scale= config["algorithm"].get("update_times_scale", 1e5),
            action_sample_std= action_std,
            log_to_tensorboard= args.log,
        ))
    runner = RunnerCls(**runner_kwargs)
    runner.collect_and_save(config= config)

if __name__ == "__main__":
    args = get_args(
        custom_args= [
            {"name": "--teacher_prob", "type": float, "default": None, "help": "probability of using teacher's action"},
            {"name": "--action_std", "type": float, "default": None, "help": "override the action sample std during rollout. None for using model's std"},
            {"name": "--log", "action": "store_true", "help": "log the data to tensorboard"},
        ],
    )
    main(args)
