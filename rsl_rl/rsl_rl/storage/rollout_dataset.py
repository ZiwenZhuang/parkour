import os
import os.path as osp
import pickle
from collections import namedtuple, OrderedDict
import json
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import rsl_rl.utils.data_compresser as compresser

class RolloutDataset(IterableDataset):
    Transitions = namedtuple("Transitions", [
        "observation", "privileged_observation", "action", "reward", "done",
    ])

    def __init__(self,
            data_dir= None,
            scan_dir= None,
            num_envs= 1,
            dataset_loops: int= 1,
            subset_traj= None, # (start_idx, end_idx) as a slice
            random_shuffle_traj_order= False, # If True, the traj_data will be loaded directoy to rl_device in a random order
            keep_latest_ratio= 1.0, # If < 1., only keeps a certain ratio of the latest trajectories
            keep_latest_n_trajs= 0, # If > 0 and more than n_trajectories, ignores keep_latest_ratio and keeps the latest n trajectories. 
            starting_frame_range= [0, 1], # if set, the starting timestep will be uniformly chose from this, when each new trajectory is loaded.
                # if sampled starting frame is bigger than the trajectory length, starting frame will be 0
            load_data_to_device= True, # If True, the traj_data will be loaded directoy to rl_device rather than np array
            rl_device= "cpu",
        ):
        """ choose data_dir or scan_dir, but not both. If scan_dir is chosen, the dataset will scan the
        directory and treat each direct subdirectory as a dataset everytime it is initialized.
        """
        self.data_dir = data_dir
        self.scan_dir = scan_dir
        self.num_envs = num_envs
        self.max_loops = dataset_loops
        self.subset_traj = subset_traj
        self.random_shuffle_traj_order = random_shuffle_traj_order
        self.keep_latest_ratio = keep_latest_ratio
        self.keep_latest_n_trajs = keep_latest_n_trajs
        self.starting_frame_range = starting_frame_range
        self.load_data_to_device = load_data_to_device
        self.rl_device = rl_device

        # check arguments
        assert not (self.data_dir is None and self.scan_dir is None), "data_dir and scan_dir cannot be both None"

        self.num_looped = 0

    def initialize(self):
        self.load_dataset_directory()
        if self.subset_traj is not None:
            self.unused_traj_dirs = self.unused_traj_dirs[self.subset_traj[0]: self.subset_traj[1]]
        if self.keep_latest_ratio < 1. or self.keep_latest_n_trajs > 0:
            self.unused_traj_dirs = sorted(
                self.unused_traj_dirs,
                key= lambda x: os.stat(x).st_ctime,
            )
            if self.keep_latest_n_trajs > 0:
                self.unused_traj_dirs = self.unused_traj_dirs[-self.keep_latest_n_trajs:]
            else:
                self.unused_traj_dirs = self.unused_traj_dirs[int(len(self.unused_traj_dirs) * self.keep_latest_ratio):]
            print("Using a subset of trajectories, total number of trajectories: ", len(self.unused_traj_dirs))
        if self.random_shuffle_traj_order:
            random.shuffle(self.unused_traj_dirs)

        # attributes that handles trajectory files for each env
        self.current_traj_dirs = [None for _ in range(self.num_envs)]
        self.trajectory_files = [[] for _ in range(self.num_envs)]
        self.traj_file_idxs = np.zeros(self.num_envs, dtype= np.int32)
        self.traj_step_idxs = np.zeros(self.num_envs, dtype= np.int32)
        self.traj_datas = [None for _ in range(self.num_envs)]

        env_idx = 0
        while env_idx < self.num_envs:
            if len(self.unused_traj_dirs) == 0:
                print("Not enough trajectories, waiting to re-initialize. Press Enter to continue....")
                input()
                self.initialize()
                return
            starting_frame = torch.randint(self.starting_frame_range[0], self.starting_frame_range[1], (1,)).item()
            update_result = self.update_traj_handle(env_idx, self.unused_traj_dirs.pop(0), starting_frame)
            if update_result:
                env_idx += 1
        
        self.dataset_drained = False

    def update_traj_handle(self, env_idx, traj_dir, starting_step_idx= 0):
        """ Load and update the trajectory handle for a given env_idx.
        Also update traj_step_idxs.
        Return whether the trajectory is successfully loaded
        """
        self.current_traj_dirs[env_idx] = traj_dir
        try:
            self.trajectory_files[env_idx] = sorted(
                os.listdir(self.current_traj_dirs[env_idx]),
                key= lambda x: int(x.split("_")[1]),
            )
            self.traj_file_idxs[env_idx] = 0
        except:
            self.nullify_traj_handles(env_idx)
            return False
        self.traj_datas[env_idx] = self.load_traj_data(
            env_idx,
            self.traj_file_idxs[env_idx],
            new_episode= True,
        )
        if self.traj_datas[env_idx] is None:
            self.nullify_traj_handles(env_idx)
            return False
        
        # The number in the file name is the timestep slice
        current_file_max_timestep = int(self.trajectory_files[env_idx][self.traj_file_idxs[env_idx]].split(".")[0].split("_")[2]) - 1
        while current_file_max_timestep < starting_step_idx:
            self.traj_file_idxs[env_idx] += 1
            if self.traj_file_idxs[env_idx] >= len(self.trajectory_files[env_idx]):
                # trajectory length is shorter than starting_step_idx, set starting_step_idx to 0
                starting_step_idx = 0
                self.traj_file_idxs[env_idx] = 0
                break
            current_file_max_timestep = int(self.trajectory_files[env_idx][self.traj_file_idxs[env_idx]].split(".")[0].split("_")[2]) - 1
            
        current_file_min_step = int(self.trajectory_files[env_idx][self.traj_file_idxs[env_idx]].split(".")[0].split("_")[1])
        self.traj_step_idxs[env_idx] = starting_step_idx - current_file_min_step
        if self.traj_file_idxs[env_idx] > 0:
            # reload the traj_data because traj_file_idxs is updated
            self.traj_datas[env_idx] = self.load_traj_data(
                env_idx,
                self.traj_file_idxs[env_idx],
                new_episode= True,
            )
            if self.traj_datas[env_idx] is None:
                self.nullify_traj_handles(env_idx)
                return False
        return True

    def nullify_traj_handles(self, env_idx):
        self.current_traj_dirs[env_idx] = ""
        self.trajectory_files[env_idx] = []
        self.traj_file_idxs[env_idx] = 0
        self.traj_step_idxs[env_idx] = 0
        self.traj_datas[env_idx] = None

    def load_dataset_directory(self):
        if self.scan_dir is not None:
            if not osp.isdir(self.scan_dir):
                print("RolloutDataset: scan_dir {} does not exist, creating...".format(self.scan_dir))
                os.makedirs(self.scan_dir)
            self.data_dir = sorted([
                osp.join(self.scan_dir, x) \
                for x in os.listdir(self.scan_dir) \
                if osp.isdir(osp.join(self.scan_dir, x)) and osp.isfile(osp.join(self.scan_dir, x, "metadata.json"))
            ])
        if isinstance(self.data_dir, list):
            total_timesteps = 0
            self.unused_traj_dirs = []
            for data_dir in self.data_dir:
                try:
                    new_trajectories = sorted([
                        osp.join(data_dir, x) \
                        for x in os.listdir(data_dir) \
                        if x.startswith("trajectory_") and len(os.listdir(osp.join(data_dir, x))) > 0
                    ], key= lambda x: int(x.split("_")[-1]))
                except:
                    continue
                self.unused_traj_dirs.extend(new_trajectories)
                try:    
                    with open(osp.join(data_dir, "metadata.json"), "r") as f:
                        self.metadata = json.load(f, object_pairs_hook= OrderedDict)
                    total_timesteps += self.metadata["total_timesteps"]
                except:
                    pass # skip
            print("RolloutDataset: Loaded data from multiple directories. The metadata is from the last directory.")
            print("RolloutDataset: Total number of timesteps: ", total_timesteps)
            print("RolloutDataset: Total number of trajectories: ", len(self.unused_traj_dirs))
        else:
            self.unused_traj_dirs = sorted([
                osp.join(self.data_dir, x) \
                for x in os.listdir(self.data_dir) \
                if x.startswith("trajectory_") and len(os.listdir(osp.join(self.data_dir, x))) > 0
            ], key= lambda x: int(x.split("_")[-1]))
            with open(osp.join(self.data_dir, "metadata.json"), "r") as f:
                self.metadata = json.load(f, object_pairs_hook= OrderedDict)
        
        # check if this dataset is initialized in worker process
        worker_info = get_worker_info()
        if worker_info is not None:
            self.dataset_loops = 1 # Let the sampler handle the loops
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            trajs_per_worker = len(self.unused_traj_dirs) // num_workers
            self.unused_traj_dirs = self.unused_traj_dirs[worker_id * trajs_per_worker: (worker_id + 1) * trajs_per_worker]
            if worker_id == num_workers - 1:
                self.unused_traj_dirs.extend(self.unused_traj_dirs[:(len(self.unused_traj_dirs) % num_workers)])
            print("RolloutDataset: Worker {} of {} initialized with {} trajectories".format(
                worker_id, num_workers, len(self.unused_traj_dirs)
            ))

    def assmeble_obs_components(self, traj_data):
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

    def load_traj_data(self, env_idx, traj_file_idx, new_episode= False):
        """ If new_episode, set the 0-th frame to done, making sure the agent is reset.
        """
        traj_dir = self.current_traj_dirs[env_idx]
        try:
            with open(osp.join(traj_dir, self.trajectory_files[env_idx][traj_file_idx]), "rb") as f:
                traj_data = pickle.load(f)
        except:
            try:
                traj_file = osp.join(traj_dir, self.trajectory_files[env_idx][traj_file_idx])
                print("Failed to load", traj_file)
            except:
                print("Failed to load file")
            # The caller will know that the file is abscent, then switch to a new trajectory
            return None
        # connect the observation components if they are disassambled in pickle files
        if "obs_disassemble_mapping" in self.metadata:
            traj_data = self.assmeble_obs_components(traj_data)
        if self.load_data_to_device:
            for data_key, data_val in traj_data.items():
                traj_data[data_key] = torch.from_numpy(data_val).to(self.rl_device)
        if new_episode:
            # add done flag to the 0-th step of newly loaded trajectory
            traj_data["dones"][0] = True
        return traj_data

    def get_transition_batch(self):
        if not hasattr(self, "dataset_drained"):
            # initialize the dataset if it is not used as a iterator
            self.initialize()
        observations = []
        privileged_observations = []
        actions = []
        rewards = []
        dones = []
        time_outs = []
        if self.dataset_drained:
            return None, None
        for env_idx in range(self.num_envs):
            traj_data = self.traj_datas[env_idx]
            traj_step_idx = self.traj_step_idxs[env_idx]
            observations.append(traj_data["observations"][traj_step_idx])
            privileged_observations.append(traj_data["privileged_observations"][traj_step_idx])
            actions.append(traj_data["actions"][traj_step_idx])
            rewards.append(traj_data["rewards"][traj_step_idx])
            dones.append(traj_data["dones"][traj_step_idx])
            if "timeouts" in traj_data: time_outs.append(traj_data["timeouts"][traj_step_idx])
            self.traj_step_idxs[env_idx] += 1
            traj_update_result = self.update_traj_data_if_needed(env_idx)
            if traj_update_result == "drained":
                self.dataset_drained = True
                return None, None
            elif traj_update_result == "new_traj":
                dones[-1][:] = True
        if torch.is_tensor(observations[0]):
            observations = torch.stack(observations)
        else:
            observations = torch.from_numpy(np.stack(observations)).to(self.rl_device)
        if torch.is_tensor(privileged_observations[0]):
            privileged_observations = torch.stack(privileged_observations)
        else:
            privileged_observations = torch.from_numpy(np.stack(privileged_observations)).to(self.rl_device)
        if torch.is_tensor(actions[0]):
            actions = torch.stack(actions)
        else:
            actions = torch.from_numpy(np.stack(actions)).to(self.rl_device)
        if torch.is_tensor(rewards[0]):
            rewards = torch.stack(rewards).squeeze(-1) # to remove the last dimension as the simulator env
        else:
            rewards = torch.from_numpy(np.stack(rewards)).to(self.rl_device).squeeze(-1)
        if torch.is_tensor(dones[0]):
            dones = torch.stack(dones).to(bool).squeeze(-1)
        else:
            dones = torch.from_numpy(np.stack(dones)).to(self.rl_device).to(bool).squeeze(-1)
        infos = dict()
        if time_outs:
            if torch.is_tensor(time_outs[0]):
                infos["time_outs"] = torch.stack(time_outs)
            else:
                infos["time_outs"] = torch.from_numpy(np.stack(time_outs)).to(self.rl_device)
        infos["num_looped"] = self.num_looped
        return self.Transitions(
            observation= observations,
            privileged_observation= privileged_observations,
            action= actions,
            reward= rewards,
            done= dones,
        ), infos
    
    def update_traj_data_if_needed(self, env_idx):
        """ Return 'new_file', 'new_traj', 'drained', or None
        """
        traj_data = self.traj_datas[env_idx]
        if self.traj_step_idxs[env_idx] >= len(traj_data["rewards"]):
            # to next file
            self.traj_file_idxs[env_idx] += 1
            self.traj_step_idxs[env_idx] = 0
            traj_data = None
            new_episode = False
            while traj_data is None:
                if self.traj_file_idxs[env_idx] >= len(self.trajectory_files[env_idx]):
                    # to next trajectory
                    if len(self.unused_traj_dirs) == 0 or not osp.isdir(self.unused_traj_dirs[0]):
                        if self.max_loops > 0 and self.num_looped >= self.max_loops:
                            return 'drained'
                        else:
                            self.num_looped += 1
                            self.initialize()
                            return 'new_traj'
                    starting_frame = torch.randint(self.starting_frame_range[0], self.starting_frame_range[1], (1,)).item()
                    self.update_traj_handle(env_idx, self.unused_traj_dirs.pop(0), starting_frame)
                    traj_data = self.traj_datas[env_idx]
                else:
                    traj_data = self.load_traj_data(
                        env_idx,
                        self.traj_file_idxs[env_idx],
                        new_episode= new_episode,
                    )
                    if traj_data is None:
                        self.nullify_traj_handles(env_idx)
                    else:
                        self.traj_datas[env_idx] = traj_data
                        return 'new_file'
        return None
    
    def set_traj_idx(self, traj_idx, env_idx= 0):
        """ Allow users to select a specific trajectory to start from """
        self.current_traj_dirs[env_idx] = self.unused_traj_dirs[traj_idx]
        self.traj_file_idxs[env_idx] = 0
        self.traj_step_idxs[env_idx] = 0
        self.trajectory_files[env_idx] = sorted(
            os.listdir(self.current_traj_dirs[env_idx]),
            key= lambda x: int(x.split("_")[1]),
        )
        self.traj_datas[env_idx] = self.load_traj_data(env_idx, self.traj_file_idxs[env_idx])
        self.dataset_drained = False

    ##### Interfaces for the IterableDataset #####
    def __iter__(self):
        self.initialize()
        transition_batch, infos = self.get_transition_batch()
        while transition_batch is not None:
            yield transition_batch, infos
            transition_batch, infos = self.get_transition_batch()
