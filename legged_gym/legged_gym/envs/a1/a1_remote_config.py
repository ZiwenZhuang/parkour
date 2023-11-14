import numpy as np
import os.path as osp
from legged_gym.envs.a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from legged_gym.utils.helpers import merge_dict

class A1RemoteCfg( A1FieldCfg ):
    class env( A1FieldCfg.env ):
        num_envs = 4096
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            # "forward_depth",
            # "base_pose",
            # "robot_config",
            # "engaging_block",
            # "sidewall_distance",
        ]
        privileged_obs_components = [
            "proprioception",
            # "height_measurements",
            # "forward_depth",
            "robot_config",
        ]
        use_lin_vel = False
        privileged_use_lin_vel = True

    class terrain( A1FieldCfg.terrain ):
        num_rows = 4
        num_cols = 4
        selected = "TerrainPerlin"
        TerrainPerlin_kwargs = dict(
            zScale= 0.15,
            frequency= 10,
        )

    class commands( A1FieldCfg.commands ):
        class ranges( A1FieldCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-1., 1.]

    class domain_rand( A1FieldCfg.domain_rand ):
        class com_range( A1FieldCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]
        max_push_vel_ang = 0.5
        push_robots = True

    class rewards( A1FieldCfg.rewards ):
        class scales:
            ###### hacker from Field
            tracking_ang_vel = 0.5
            lin_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -4e-2
            exceed_torque_limits_l1norm = -4e-1
            # penalty for walking gait, probably no need
            collision = -0.1
            orientation = -0.1
            feet_contact_forces = -1e-3
        soft_dof_pos_limit = 0.5
        max_contact_force = 60.0

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1RemoteCfgPPO( A1FieldCfgPPO ):
    class algorithm( A1FieldCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.05

    class runner( A1FieldCfgPPO.runner ):
        resume = False
        load_run = None

        run_name = "".join(["WalkByRemote",
        ("_rLinTrack{:.1f}".format(A1RemoteCfg.rewards.scales.tracking_lin_vel) if getattr(A1RemoteCfg.rewards.scales, "tracking_lin_vel", 0.) != 0. else ""),
        ("_pLin{:.1f}".format(-A1RemoteCfg.rewards.scales.lin_vel_l2norm) if getattr(A1RemoteCfg.rewards.scales, "lin_vel_l2norm", 0.) != 0. else ""),
        ("_rAng{:.1f}".format(A1RemoteCfg.rewards.scales.tracking_ang_vel) if A1RemoteCfg.rewards.scales.tracking_ang_vel != 0.2 else ""),
        ("_rAlive{:.1f}".format(A1RemoteCfg.rewards.scales.alive) if getattr(A1RemoteCfg.rewards.scales, "alive", 2.) != 2. else ""),
        ("_pEnergySubsteps{:.0e}".format(A1RemoteCfg.rewards.scales.legs_energy_substeps) if getattr(A1RemoteCfg.rewards.scales, "legs_energy_substeps", -2e-5) != -2e-5 else "_nopEnergy"),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])