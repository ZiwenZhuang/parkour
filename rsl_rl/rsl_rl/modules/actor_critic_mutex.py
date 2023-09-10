import os
import os.path as osp
import json
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from rsl_rl.utils.utils import get_obs_slice
from rsl_rl import modules

class ActorCriticMutex(modules.ActorCritic):
    def __init__(self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            sub_policy_class_name,
            sub_policy_paths,
            obs_segments= None,
            privileged_obs_segments= None, # No need
            env_action_scale = 0.5,
            **kwargs,
        ):
        if kwargs:
            print("ActorCriticMutex.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        if privileged_obs_segments is None:
            privileged_obs_segments = obs_segments
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.obs_segments = obs_segments
        self.privileged_obs_segments = privileged_obs_segments
        self.sub_policy_paths = sub_policy_paths
        nn.Module.__init__(self) # This implementation is incompatible with the default ActorCritic implementation
        if isinstance(env_action_scale, (tuple, list)):
            print("Warning: a list of env action scale is applied, check if it is what you need.")
            self.register_buffer("env_action_scale", torch.tensor(env_action_scale))
        else:
            self.env_action_scale = env_action_scale

        self.submodules = nn.ModuleList()
        self.is_recurrent = False
        if not sub_policy_paths:
            print("ActorCriticMutex Warning: No sub policy snapshot path provided. No sub policy is available")
        for subpolicy_idx, sub_path in enumerate(sub_policy_paths):
            # get the config
            with open(osp.join(sub_path, "config.json"), "r") as f:
                run_kwargs = json.load(f, object_pairs_hook= OrderedDict)
                policy_kwargs = run_kwargs["policy"]
            # initiate the policy instance
            self.submodules.append(getattr(modules, sub_policy_class_name)(
                num_actor_obs,
                num_critic_obs,
                num_actions,
                obs_segments= obs_segments,
                privileged_obs_segments= privileged_obs_segments,
                **policy_kwargs,
            ))
            if self.submodules[-1].is_recurrent: self.is_recurrent = True
            self.register_buffer(
                "subpolicy_action_scale_{:d}".format(subpolicy_idx),
                torch.tensor(run_kwargs["control"]["action_scale"]) \
                if isinstance(run_kwargs["control"]["action_scale"], (tuple, list)) \
                else torch.tensor([run_kwargs["control"]["action_scale"]])[0]
            )
            # acquire the snapshot and load the weights
            fmodels = [f for f in os.listdir(sub_path) if 'model' in f]
            fmodels.sort(key=lambda m: '{0:0>15}'.format(m))
            fmodel = fmodels[-1]
            ep_snapshot = torch.load(osp.join(sub_path, fmodel), map_location= "cpu")
            self.submodules[-1].load_state_dict(ep_snapshot["model_state_dict"])
        if len(self.submodules) > 0:
            print("ActorCriticMutex Info: {} policy loaded".format(len(sub_policy_paths)))

    def reset(self, dones=None):
        for module in self.submodules:
            module.reset(dones)

    def act(self, observations, **kwargs):
        raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")

    def act_inference(self, observations):
        raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")

    # Some other methods are temporary not used. May be added later.
    