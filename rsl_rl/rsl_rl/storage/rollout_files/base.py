from abc import ABC, abstractmethod
import os.path as osp
from collections import namedtuple, OrderedDict
import numpy as np
import random
import torch
from torch.utils.data import IterableDataset

class RolloutFileBase(IterableDataset):
    """ A unified interface design to load a batch of trajectory/sequence from file.
    """

    def __init__(self, data_dir, num_envs, device= "cuda"
        ):
        """
        Args:
            data_dir: directory to load the data
            num_envs: number of envs to fill the buffers, fixed during training.
        """
        self.data_dir = data_dir
        self.num_envs = num_envs
        self.device = device
        self.__initialized = False
        # The inherited class should handle reset() call by itself when first initialized.
        # e.g. rollout_dataset needs the collector to fill the trajectories. But the collectors
        # needs the run directory to be initialized and get proper training configs. But the
        # run directory is not available before everything is initialized. (in task_registry.py)
        self.all_env_ids = torch.arange(num_envs, device= self.device)

    def reset(self, env_ids= None):
        """ Reset the handlers. Make sure values in env_ids are unique.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            self.reset_all()
        else:
            self.refresh_handlers(env_ids)
        # for env_idx in range(self.num_envs):
        #     self._refresh_traj_handler(env_idx)

    def get_batch(self, num_transitions_per_env= None):
        """ Filling in the buffer starting with the current frame_cursors. The assigned motion will
        be refreshed when one motion is exhausted.
        ----------
        Args:
        - num_transitions_per_env: int, number of transitions to fill for each env.
            The returned buffer have no time dimension if num_transitions_per_env is None.
        ----------
        Returns:
        - buffer: (num_envs, *transition_shape) if num_transitions_per_env is None or
            (num_transitions_per_env, num_envs, *transition_shape) otherwise.
        """
        if not self.__initialized:
            self.reset()
            self.__initialized = True
        buffer = self.get_buffer(num_transitions_per_env)
        if num_transitions_per_env is None:
            self.fill_transition(buffer)
        else:
            for time_idx in range(num_transitions_per_env):
                self.fill_transition(buffer[time_idx])
            
        return buffer
    
    def get_transition_batch(self):
        """ A temporal interface to simulate the env rollout, which produces (s, a, r, d, info) in
        each timestep.
        """
        buffer = self.get_batch()
        if "timeout" in buffer:
            # TODO: Due to the inconsistency between rsl_rl and legged_gym, timeout and time_outs
            # are mixed
            return buffer, {"time_outs": buffer.timeout}
        else:
            return buffer, {}
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        return self.get_batch()

    @abstractmethod
    def reset_all(self):
        """ build the handlers, reset them and sample identifiers for all envs to the initial state. """
        self.traj_identifiers = None
        pass

    @abstractmethod
    def refresh_handlers(self, env_ids= None):
        """ Given the env_ids, refresh the traj handlers of these env.
        Could be called externally when the env is reset.
        """
        pass

    @abstractmethod
    def get_buffer(self, num_transitions_per_env: int = None):
        """ Build and return the buffer to return the transition batch. """
        pass

    @abstractmethod
    def fill_transition(self, buffer, env_ids= None):
        """ Fill the transition for the given env_ids (batchwise) to buffer.
        The buffer does not requries further indexing.
        Also, advance the cursor for one step and maintain its handlers.
        NOTE: Each trainsition stored in the file should be a tuple of (s, a, r, d, ...) for each
        frame. But the output transition should also fill the next observation to the output buffer.
        """
        pass
