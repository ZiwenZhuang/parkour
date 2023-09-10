import os
import os.path as osp
import json
import pickle
import time

import numpy as np
import torch

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.demonstration import DemonstrationSaver
from rsl_rl.algorithms.tppo import GET_TEACHER_ACT_PROB_FUNC

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

        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_TEACHER_ACT_PROB_FUNC(self.teacher_act_prob, update_times_scale)
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
            print("Training policy iteration: {}".format(self.training_policy_iteration))
        self.use_teacher_act_mask = torch.rand(self.env.num_envs) < self.teacher_act_prob(self.training_policy_iteration)

    def get_transition(self):
        if self.use_critic_obs:
            teacher_actions = self.policy.act_inference(self.critic_obs)
        else:
            teacher_actions = self.policy.act_inference(self.obs)
        actions = self.training_policy.act(self.obs)
        if self.action_sample_std > 0:
            actions += torch.randn_like(actions) * self.action_sample_std
        actions[self.use_teacher_act_mask] = teacher_actions[self.use_teacher_act_mask]
        n_obs, n_critic_obs, rewards, dones, infos = self.env.step(actions)
        # Use teacher actions to label the trajectory, no matter what the student policy does
        return teacher_actions, rewards, dones, infos, n_obs, n_critic_obs
    
    def policy_reset(self, dones):
        return_ = super().policy_reset(dones)
        if dones.any():
            self.training_policy.reset(dones)
        return return_
        
    def check_stop(self):
        """ Also check whether need to load the latest training policy model. """
        self.load_latest_training_policy()
        return super().check_stop()
