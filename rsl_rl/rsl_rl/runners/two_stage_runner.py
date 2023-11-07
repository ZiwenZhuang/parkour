import os

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage.rollout_dataset import RolloutDataset

class TwoStageRunner(OnPolicyRunner):
    """ A runner that have a pretrain stage which is used to collect demonstration data """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load some configs and their default values
        self.pretrain_iterations = self.cfg.get("pretrain_iterations", 0)
        self.log_interval = self.cfg.get("log_interval", 50)
        assert "pretrain_dataset" in self.cfg, "pretrain_dataset is not defined in the runner cfg object"
        self.rollout_dataset = RolloutDataset(
            **self.cfg["pretrain_dataset"],
            num_envs= self.env.num_envs,
            rl_device= self.alg.device,
        )

    def rollout_step(self, obs, critic_obs):
        # check if within the pretrain stage or RL stage
        if self.pretrain_iterations < 0 or self.current_learning_iteration < self.pretrain_iterations:
            transition, infos = self.rollout_dataset.get_transition_batch()
            if infos:
                for k, v in infos.items():
                    if not k in ["time_outs"]:
                        self.writer.add_scalar("Perf/dataset_" + k, v, self.current_learning_iteration)
            if not transition is None:
                self.alg.collect_transition_from_dataset(transition, infos)
                return (
                    transition.observation,
                    transition.privileged_observation,
                    transition.reward,
                    transition.done,
                    infos,
                )
            else:
                obs = self.env.get_observations()
                critic_obs = self.env.get_privileged_observations()
        return super().rollout_step(obs, critic_obs)
