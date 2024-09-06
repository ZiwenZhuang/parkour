import os
import torch
import numpy as np
import pickle
import json
import time
from collections import OrderedDict

from rsl_rl.utils.collections import namedarraytuple
import rsl_rl.utils.data_compresser as compresser
from rsl_rl.storage.rollout_files.base import RolloutFileBase

class RolloutDataset(RolloutFileBase):
    Transition = namedarraytuple("Transition", [
        "observation",
        "privileged_observation",
        "action",
        "reward",
        "done",
        "timeout",
        "next_observation",
        "next_privileged_observation",
    ])
    def __init__(self, data_dir, num_envs,
            dataset_loops: int = 1,
            random_shuffle_traj_order= False,
            keep_latest_n_trajs= 0, # If > 0 and more than n_trajectories, ignores keep_latest_ratio and keeps the latest n trajectories.
            starting_frame_range= [0, 1], # if set, the starting timestep will be uniformly chose from this, when each new trajectory is loaded.
                # if sampled starting frame is bigger than the trajectory length, starting frame will be 0
            device= "cuda",
        ):
        super().__init__(data_dir, num_envs, device= device)
        self.dataset_loops = dataset_loops
        self.random_shuffle_traj_order = random_shuffle_traj_order
        self.keep_latest_n_trajs = keep_latest_n_trajs
        self.starting_frame_range = starting_frame_range
        
        self.num_dataset_looped = 0

    @staticmethod
    def get_frame_range(filename: str) -> tuple:
        """ Get the frame range from the filename. Return a tuple [start, end). (end is exclusive)
        """
        return (
            int(filename.split(".")[0].split("_")[1]),
            int(filename.split(".")[0].split("_")[2]),
        )

    def read_dataset_directory(self):
        """ Refresh file-related information by scanning the directory. All traj_handlers must be
        updated from attributes here.
        """
        if isinstance(self.data_dir, str):
            self.data_dirs = [self.data_dir]
        elif isinstance(self.data_dir, (tuple, list)):
            self.data_dirs = self.data_dir
        else:
            raise ValueError("data_dir should be a string or a list of strings.")
        self.all_available_trajectory_dirs = []
        self.metadata = None # reset metadata to ensure only one metadata is used
        metadata_repeated = False
        total_timesteps = 0
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
                print("RolloutDataset: {} not found, created...".format(data_dir))
            for root, dirs, _ in os.walk(data_dir):
                for d in dirs:
                    if d.startswith("trajectory_") and len(os.listdir(os.path.join(root, d))) > 0:
                        self.all_available_trajectory_dirs.append(os.path.join(root, d))
                        trajectory_files = os.listdir(os.path.join(root, d))
                        trajectory_files.sort(key= lambda x: self.get_frame_range(x)[1])
                        total_timesteps += self.get_frame_range(trajectory_files[-1])[1]
                try:
                    with open(os.path.join(root, "metadata.json"), "r") as f:
                        if self.metadata is None:
                            self.metadata = json.load(f, object_pairs_hook= OrderedDict)
                        elif not metadata_repeated:
                            print("RolloutDataset: multiple metadata files found, using the first one.")
                            metadata_repeated = True
                except FileNotFoundError:
                    pass # skip
        # making sure all trajectories are sorted by modification time
        self.all_available_trajectory_dirs.sort(key= lambda x: os.path.getmtime(x))
        print("RolloutDataset: {} trajectories found. {} timesteps in total.".format(
            len(self.all_available_trajectory_dirs),
            total_timesteps,
        ))
        if len(self.all_available_trajectory_dirs) < self.keep_latest_n_trajs:
            return False
        else:
            self.all_available_trajectory_dirs = self.all_available_trajectory_dirs[-self.keep_latest_n_trajs:]
        self.unused_trajectory_idxs = list(range(len(self.all_available_trajectory_dirs)))
        if self.random_shuffle_traj_order:
            self.unused_trajectory_idxs = np.random.permutation(self.unused_trajectory_idxs)
        return True

    def assemble_obs_components(self, traj_data):
        assert "obs_segments" in self.metadata, "Corrupted metadata, obs_segments not found in metadata"
        observations = []
        for component_name in self.metadata["obs_segments"].keys():
            obs_component = traj_data.pop("obs_" + component_name)
            if component_name in self.metadata["obs_disassemble_mapping"]:
                obs_component = getattr(
                    compresser,
                    "decompress_" + self.metadata["obs_disassemble_mapping"][component_name],
                )(obs_component)
            observations.append(obs_component)
        traj_data["observations"] = np.concatenate(observations, axis= -1) # (n_steps, d_obs)
        return traj_data

    def reset_all(self):
        """ Reset and defines the handlers. Usually called in reset() to initialize the handlers.
        All handlers that identify which trajectory(file) is currently loaded for each env appear
        here.
        """
        while not self.read_dataset_directory():
            print("RolloutDataset: trajectory not enough, need {} at least, waiting for 15 minutes...".format(self.keep_latest_n_trajs))
            time.sleep(60 * 15)
        # use trajectory index to identify the trajectory in all_available_trajectory_dirs
        self.traj_identifiers = self.unused_trajectory_idxs[:self.num_envs]
        self.unused_trajectory_idxs = [i for i in self.unused_trajectory_idxs if i not in self.traj_identifiers]
        self.traj_file_names = [[] for _ in range(self.num_envs)]
        self.traj_lengths = [None for _ in range(self.num_envs)]
        self.traj_file_idxs = [None for _ in range(self.num_envs)] # in ascending order
        self.traj_datas = [None for _ in range(self.num_envs)]
        self.traj_cursors = np.zeros(self.num_envs, dtype= int)

        self.refresh_handlers()

    def _refresh_traj_data(self, env_idx):
        """ refresh `self.traj_data` based on current traj_file_idxs[env_idx]. usually called
        after refreshing traj_handler or updated traj_file_idxs[env_idx]
        """
        traj_dir = self.all_available_trajectory_dirs[self.traj_identifiers[env_idx]]
        try:
            with open(os.path.join(traj_dir, self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]]), "rb") as f:
                traj_data = pickle.load(f)
        except:
            raise RuntimeError("RolloutDataset: failed to load trajectory data from {}".format(
                os.path.join(traj_dir, self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])
            ))
        if "obs_disassemble_mapping" in self.metadata.keys():
            traj_data = self.assemble_obs_components(traj_data)
        for k, v in traj_data.items():
            traj_data[k] = torch.from_numpy(v).to(self.device)
        self.traj_datas[env_idx] = traj_data

    def _refresh_traj_handler(self, env_idx):
        """ update traj_handler for the given env and load the first traj_data. It does not update
        the traj_identifiers.
        """
        traj_dir = self.all_available_trajectory_dirs[self.traj_identifiers[env_idx]]
        trajectory_files = os.listdir(traj_dir)
        trajectory_files.sort(key= lambda x: self.get_frame_range(x)[1])
        self.traj_cursors[env_idx] = np.random.randint(
            min(self.starting_frame_range[0], self.get_frame_range(trajectory_files[-1])[0]),
            min(self.starting_frame_range[1], self.get_frame_range(trajectory_files[-1])[1]),
        )
        self.traj_file_names[env_idx] = trajectory_files
        self.traj_lengths[env_idx] = self.get_frame_range(trajectory_files[-1])[1]
        self.traj_file_idxs[env_idx] = 0
        while (self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0] > self.traj_cursors[env_idx] \
            or self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[1] <= self.traj_cursors[env_idx]):
            self.traj_file_idxs[env_idx] += 1 \
                if self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[1] <= self.traj_cursors[env_idx] else -1
        self._refresh_traj_data(env_idx)
        self.traj_datas[env_idx]["dones"][0] = True # set the first frame as done

    def refresh_handlers(self, env_ids= None):
        if env_ids is None: env_ids = self.all_env_ids

        for env_idx in env_ids:
            self.traj_identifiers[env_idx] = self.unused_trajectory_idxs.pop(0)
            self._refresh_traj_handler(env_idx)

    def _maintain_handler(self, env_idx):
        """ Maintain traj_handler and update traj_data if needed. Return whether a new trajectory
        is loaded.
        """
        try:
            if self.traj_cursors[env_idx] >= self.traj_lengths[env_idx]:
                # load a new trajectory
                # NOTE: self.unused_trajectory_idxs should be shuffled during read_dataset_directory if needed
                if len(self.unused_trajectory_idxs) == 0:
                    raise StopIteration
                self.traj_identifiers[env_idx] = self.unused_trajectory_idxs.pop(0)
                self._refresh_traj_handler(env_idx)
                return True
            traj_cursor_range = self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])
            if self.traj_cursors[env_idx] < traj_cursor_range[0] or self.traj_cursors[env_idx] >= traj_cursor_range[1]:
                # load new traj_data from the same trajectory
                self.traj_file_idxs[env_idx] += 1
                self._refresh_traj_data(env_idx)
                return False
        except StopIteration:
            if self.dataset_loops < 1 or self.num_dataset_looped < self.dataset_loops:
                # loop the dataset
                self.reset()
                return True
            else:
                raise StopIteration
        return False

    def get_buffer(self, num_transitions_per_env= None):
        leading_dims = ([] if num_transitions_per_env is None else [num_transitions_per_env]) + [self.num_envs]
        if not hasattr(self, "_output_transition_buffer") or self._output_transition_buffer_leading_dims != leading_dims:
            observations = torch.empty(
                leading_dims + list(self.traj_datas[0]["observations"].shape[1:]),
                dtype= self.traj_datas[0]["observations"].dtype,
                device= self.device,
            )
            privileged_observations = torch.empty(
                leading_dims + list(self.traj_datas[0]["privileged_observations"].shape[1:]),
                dtype= self.traj_datas[0]["privileged_observations"].dtype,
                device= self.device,
            )
            actions = torch.empty(
                leading_dims + list(self.traj_datas[0]["actions"].shape[1:]),
                dtype= self.traj_datas[0]["actions"].dtype,
                device= self.device,
            )
            rewards = torch.empty(
                leading_dims,
                dtype= self.traj_datas[0]["rewards"].dtype,
                device= self.device,
            )
            dones = torch.empty(
                leading_dims,
                dtype= bool,
                device= self.device,
            )
            timeouts = torch.empty(
                leading_dims,
                dtype= self.traj_datas[0]["timeouts"].dtype,
                device= self.device,
            )
            next_observations = torch.empty(
                leading_dims + list(self.traj_datas[0]["observations"].shape[1:]),
                dtype= self.traj_datas[0]["observations"].dtype,
                device= self.device,
            )
            next_privileged_observations = torch.empty(
                leading_dims + list(self.traj_datas[0]["privileged_observations"].shape[1:]),
                dtype= self.traj_datas[0]["privileged_observations"].dtype,
                device= self.device,
            )
            self._output_transition_buffer_leading_dims = leading_dims
            self._output_transition_buffer = self.Transition(
                observation= observations,
                privileged_observation= privileged_observations,
                action= actions,
                reward= rewards,
                done= dones,
                timeout= timeouts,
                next_observation= next_observations,
                next_privileged_observation= next_privileged_observations,
            )
        return self._output_transition_buffer
    
    def _fill_transition_per_env(self, buffer, env_idx: int):
        traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
        buffer.observation.copy_(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
        buffer.privileged_observation.copy_(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])
        buffer.action.copy_(self.traj_datas[env_idx]["actions"][traj_cursor_in_file])
        buffer.reward.copy_(self.traj_datas[env_idx]["rewards"][traj_cursor_in_file].squeeze())
        buffer.done.copy_(self.traj_datas[env_idx]["dones"][traj_cursor_in_file].squeeze())
        if "timeout" in self.traj_datas[env_idx].keys():
            buffer.timeout.copy_(self.traj_datas[env_idx]["timeouts"][traj_cursor_in_file].squeeze())
        self.traj_cursors[env_idx] += 1
        if self._maintain_handler(env_idx):
            if not buffer.done.any():
                buffer.timeout.copy_(torch.tensor([True], device= self.device).squeeze())
            buffer.done.copy_(torch.tensor([True], device= self.device).squeeze())
        traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
        buffer.next_observation.copy_(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
        buffer.next_privileged_observation.copy_(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])

    def fill_transition(self, buffer, env_ids= None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device= self.device)
        for env_idx in env_ids:
            self._fill_transition_per_env(buffer[env_idx], env_idx)
