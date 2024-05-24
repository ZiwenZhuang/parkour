import os
import os.path as osp
import json
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tabulate import tabulate

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.demonstration import DemonstrationSaver
from rsl_rl.algorithms.tppo import GET_PROB_FUNC

class DaggerSaver(DemonstrationSaver):
    """ This demonstration saver will rollout the trajectory by running the student policy
    (with a probablity) and label the trajectory by running the teacher policy
    """
    def __init__(self,
            *args,
            training_policy_logdir,
            teacher_act_prob= "exp",
            update_times_scale= 5000,
            action_sample_std= 0.0, # if > 0, add Gaussian noise to the action in effort.
            log_to_tensorboard= False, # if True, log the rollout episode info to tensorboard
            **kwargs,
        ):
        """
        Args:
            teacher_act_prob: The same as in TPPO, but the iteration will be get from the model checkpoint
        """
        super().__init__(*args, **kwargs)
        self.training_policy_logdir = training_policy_logdir
        self.teacher_act_prob = teacher_act_prob
        self.update_times_scale = update_times_scale
        self.action_sample_std = action_sample_std
        self.log_to_tensorboard = log_to_tensorboard
        if self.log_to_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir= osp.join(
                    self.training_policy_logdir,
                    "_".join(["collector", *(osp.basename(self.save_dir).split("_")[:2])]),
                ),
                flush_secs= 10,
            )

        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_PROB_FUNC(self.teacher_act_prob, update_times_scale)
        else:
            self.__teacher_act_prob = self.teacher_act_prob
            self.teacher_act_prob = lambda x: self.__teacher_act_prob

    def init_traj_handlers(self):
        return_ = super().init_traj_handlers()
        self.metadata["training_policy_logdir"] = self.training_policy_logdir
        self.metadata["update_times_scale"] = self.update_times_scale
        self.metadata["action_sample_std"] = self.action_sample_std
        self.build_training_policy()
        return return_

    def init_storage_buffer(self):
        return_ = super().init_storage_buffer()
        self.rollout_episode_infos = []
        return return_

    def build_training_policy(self):
        """ Load the latest training policy model. """
        with open(osp.join(self.training_policy_logdir, "config.json"), "r") as f:
            config = json.load(f)
        training_policy = build_actor_critic(
            self.env,
            config["runner"]["policy_class_name"],
            config["policy"],
        ).to(self.env.device)
        self.training_policy = training_policy
        self.training_policy_iteration = 0
    
    def load_latest_training_policy(self):
        assert hasattr(self, "training_policy"), "Please build the training policy first."
        models = [file for file in os.listdir(self.training_policy_logdir) if 'model' in file]
        if len(models) == 0:
            print("No model found in the training policy logdir. Make sure you don't need to load the training policy or stop the program.")
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model_f = models[-1]
        model_f_iter = int((model_f.split(".")[0]).split("_")[1])
        if model_f_iter > self.training_policy_iteration:
            loaded_dict = None
            while loaded_dict is None:
                try:
                    loaded_dict = torch.load(osp.join(self.training_policy_logdir, model_f))
                except RuntimeError:
                    print("Failed to load model state dict file, wait 0.1s")
                    time.sleep(0.1)
            self.training_policy.load_state_dict(loaded_dict["model_state_dict"])
            self.training_policy_iteration = loaded_dict["iter"]
            # override the action std in self.training_policy
            with torch.no_grad():
                if self.action_sample_std > 0:
                    self.training_policy.std[:] = self.action_sample_std
            print("Training policy iteration: {}".format(self.training_policy_iteration))
        self.use_teacher_act_mask = torch.rand(self.env.num_envs) < self.teacher_act_prob(self.training_policy_iteration)

    def get_transition(self):
        teacher_actions = self.get_policy_actions()
        actions = self.training_policy.act(self.obs)
        actions[self.use_teacher_act_mask] = teacher_actions[self.use_teacher_act_mask]
        n_obs, n_critic_obs, rewards, dones, infos = self.env.step(actions)
        # Use teacher actions to label the trajectory, no matter what the student policy does
        return teacher_actions, rewards, dones, infos, n_obs, n_critic_obs

    def add_transition(self, step_i, infos):
        return_ = super().add_transition(step_i, infos)
        if "episode" in infos:
            self.rollout_episode_infos.append(infos["episode"])
        return return_
    
    def policy_reset(self, dones):
        return_ = super().policy_reset(dones)
        if dones.any():
            self.training_policy.reset(dones)
        return return_
        
    def check_stop(self):
        """ Also check whether need to load the latest training policy model. """
        self.load_latest_training_policy()
        return super().check_stop()
    
    def print_log(self):
        # Copy from runner logging mechanism. TODO: optimize these implementation into one.
        ep_table = []
        for key in self.rollout_episode_infos[0].keys():
            infotensor = torch.tensor([], device= self.env.device)
            for ep_info in self.rollout_episode_infos:
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(self.env.device)))
            if "_max" in key:
                infotensor = infotensor[~infotensor.isnan()]
                value = torch.max(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
            elif "_min" in key:
                infotensor = infotensor[~infotensor.isnan()]
                value = torch.min(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
            else:
                value = torch.nanmean(infotensor)
            if self.log_to_tensorboard:
                self.tb_writer.add_scalar('Episode/' + key, value, self.training_policy_iteration)
            ep_table.append(("Episode/" + key, value.detach().cpu().item()))
        # NOTE: assuming dagger trainner's iteration is always faster than collector's iteration
        # Otherwise, the training_policy will not be updated.
        self.training_policy_iteration += 1
        print("Sampling saved for training policy iteration: {}".format(self.training_policy_iteration))
        print(tabulate(ep_table))
        self.rollout_episode_infos = []
        return super().print_log()
