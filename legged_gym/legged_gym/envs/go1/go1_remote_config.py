import numpy as np
import os.path as osp
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO

class Go1RemoteCfg( Go1FieldCfg ):
    class env( Go1FieldCfg.env ):
        num_envs = 4096
        obs_components = [
            "proprioception", # 48
        ]
        privileged_obs_components = [
            "proprioception",
            "robot_config",
        ]
        use_lin_vel = False
        privileged_use_lin_vel = True

    class init_state( Go1FieldCfg.init_state ):
        pos = [0., 0., 0.42]

    class terrain( Go1FieldCfg.terrain ):
        num_rows = 6
        num_cols = 6
        selected = "TerrainPerlin"
        TerrainPerlin_kwargs = dict(
            zScale= 0.15,
            frequency= 10,
        )

    class commands( Go1FieldCfg.commands ):
        class ranges( Go1FieldCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-1., 1.]

    class domain_rand( Go1FieldCfg.domain_rand ):
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_base_rot_range = dict(
            roll= [-0.4, 0.4],
            pitch= [-0.4, 0.4],
        )
        init_base_vel_range = merge_dict(Go1FieldCfg.domain_rand.init_base_vel_range, dict(
            x= [-0.8, 1.5],
            y= [-0.8, 0.8],
        ))

    class rewards( Go1FieldCfg.rewards ):
        class scales:
            ###### hacker from Field
            tracking_ang_vel = 1.
            tracking_lin_vel = 3.
            # lin_vel_l2norm = -1.
            alive = 1.
            legs_energy_substeps = -6e-7
            # penalty for hardware safety
            # exceed_dof_pos_limits = -4e-2
            exceed_torque_limits_l1norm = -1e-2
            # penalty for walking gait, probably no need
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -4.
            dof_acc = -1.e-7
            collision = -10.
            action_rate = -0.01
            delta_torques = -1e-7
            torques = -1.e-5
            hip_pos = -0.4
            dof_error = -0.04
            stand_still = -0.6
        only_positive_rewards = False
        soft_dof_pos_limit = 0.6

    class normalization( Go1FieldCfg.normalization ):
        clip_actions_method = "hard"

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1RemoteCfgPPO( Go1FieldCfgPPO ):
    class algorithm( Go1FieldCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.02

    class runner( Go1FieldCfgPPO.runner ):
        resume = True
        # load_run = osp.join(logs_root, "field_a1/Sep26_15-33-45_WalkByRemote_pEnergySubsteps2e-5_pTorqueL18e-01_pCollision0.2_propDelay0.04-0.05_aScale0.5_kp40_kd0.5_maxPushAng0.5_noTanh_hardActClip_noResume")
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Sep28_11-53-30_WalkByRemote_pEnergySubsteps-4e-05_kp40_kd0.5_noTanh_fromSep26_15-33-45")
        load_run = "Oct04_16-15-17_WalkByRemote_pEnergySubsteps-1e-04_softDof0.5_pStand0.0_pDofAcc5e-7_kp40_kd0.5_noTanh_fromSep28_11-53-30"
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct12_16-01-06_WalkByRemote_pLin1.0_pFootForce1.e-02_xComMin-0.2_maxForce200_propDelay0.04-0.05_entropyCoef0.01_noTanh_fromOct11_04-34-39")
        load_run = "Oct23_03-33-29_WalkByRemote_pEnergySubsteps-1e-05_rLinTrack3.0_softDof0.6_pStand1.0_kp40_kd0.5_noTanh_fromOct12_16-01-06"
        load_run = "Oct23_11-01-47_WalkByRemote_pEnergySubsteps-1e-05_rLinTrack5.0_pCollide1e-1_pOrient1e-1_pStand2e-1_softDof0.6_kp40_kd0.5_noTanh_fromOct23_03-33-29"
        load_run = "Oct23_13-14-42_WalkByRemote_pEnergySubsteps1e-05_rLinTrack5.0_pDofAcc2.5e-7_pCollide1e-1_pOrient1e-1_pExceedDof8e-1_pStand4e-1_softDof0.6_noTanh_EntropyCoef0.01_fromOct23_11-01-47"
        load_run = osp.join(logs_root, "field_a1_noTanh_oracle", "Oct12_16-00-55_WalkByRemote_pLin1.0_pFootForce1.e-02_xComMin-0.2_maxForce200_propDelay0.05-0.07_entropyCoef0.01_noTanh_fromOct11_04-34-39")
        load_run = "Oct24_15-05-23_WalkByRemote_rLinTrack3.0_pActRate1e-1_softDof0.6_noTanh_zeroResetAction_hackReset_EntropyCoef0.01_actionCliphard_fromOct12_16-00-55"

        run_name = "".join(["WalkByRemote",
        ("_pEnergySubsteps{:.0e}".format(-Go1RemoteCfg.rewards.scales.legs_energy_substeps) if getattr(Go1RemoteCfg.rewards.scales, "legs_energy_substeps", 0.) != 0. else ""),
        ("_rLinTrack{:.1f}".format(Go1RemoteCfg.rewards.scales.tracking_lin_vel) if getattr(Go1RemoteCfg.rewards.scales, "tracking_lin_vel", 0.) != 0. else ""),
        ("_pLinVelL2{:.1f}".format(-Go1RemoteCfg.rewards.scales.lin_vel_l2norm) if getattr(Go1RemoteCfg.rewards.scales, "lin_vel_l2norm", 0.) != 0. else ""),
        ("_rAlive{:.1f}".format(Go1RemoteCfg.rewards.scales.alive) if getattr(Go1RemoteCfg.rewards.scales, "alive", 0.) != 0. else ""),
        # ("_pDofAcc" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.dof_acc, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "dof_acc", 0.) != 0. else ""),
        # ("_pCollide" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.collision, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "collision", 0.) != 0. else ""),
        # ("_pOrient" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.orientation, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "orientation", 0.) != 0. else ""),
        # ("_pExceedDof" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.exceed_dof_pos_limits, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "exceed_dof_pos_limits", 0.) != 0. else ""),
        ("_pExceedTorqueL1" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.exceed_torque_limits_l1norm, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "exceed_torque_limits_l1norm", 0.) != 0. else ""),
        # ("_pStand" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.stand_still, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "stand_still", 0.) != 0. else ""),
        ("_pActRate" + np.format_float_scientific(-Go1RemoteCfg.rewards.scales.action_rate, trim= "-", exp_digits= 1) if getattr(Go1RemoteCfg.rewards.scales, "action_rate", 0.) != 0. else ""),
        ("_softDof{:.1f}".format(Go1RemoteCfg.rewards.soft_dof_pos_limit) if Go1RemoteCfg.rewards.soft_dof_pos_limit != 0.9 else ""),
        # ("_kp{:d}".format(int(Go1RemoteCfg.control.stiffness["joint"])) if Go1RemoteCfg.control.stiffness["joint"] != 50 else ""),
        # ("_kd{:.1f}".format(Go1RemoteCfg.control.damping["joint"]) if Go1RemoteCfg.control.damping["joint"] != 1. else ""),
        ("_noTanh"),
        # ("_zeroResetAction" if Go1RemoteCfg.init_state.zero_actions else ""),
        ("_EntropyCoef0.01"),
        ("_actionClip" + Go1RemoteCfg.normalization.clip_actions_method if getattr(Go1RemoteCfg.normalization, "clip_actions_method", None) is not None else ""),
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

    
