import numpy as np
import torch
from copy import copy

from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import convert_heightfield_to_trimesh
from legged_gym.utils import trimesh
from legged_gym.utils.terrain.perlin import TerrainPerlin
from legged_gym.utils.console import colorize

class BarrierTrack:
    # default kwargs
    track_kwargs = dict(
            options= [
                "jump",
                "crawl",
                "tilt",
            ], # each race track will permute all the options
            randomize_obstacle_order= True, # if True, will randomize the order of the obstacles instead of the order in options
            n_obstacles_per_track= 1, # number of obstacles per track, only used when randomize_obstacle_order is True
            track_width= 1.6,
            track_block_length= 1.2, # the x-axis distance from the env origin point
            wall_thickness= 0.04, # [m]
            wall_height= 0.5, # [m]
            jump= dict(
                height= 0.3,
                depth= 0.04, # size along the forward axis
                fake_offset= 0.0, # [m] fake offset will make jump's height info greater than its physical height.
                jump_down_prob= 0.0, # if > 0, will have a chance to jump down from the obstacle
            ),
            crawl= dict(
                height= 0.32,
                depth= 0.04, # size along the forward axis
                wall_height= 0.8,
                no_perlin_at_obstacle= False, # if True, will set the heightfield to zero at the obstacle
            ),
            tilt= dict(
                width= 0.18,
                depth= 0.04, # size along the forward axis
                opening_angle= 0.3, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.8,
            ),
            leap= dict(
                length= 0.8,
                depth= 0.5,
                height= 0.1, # expected leap height over the gap
            ),
            # If True, will add perlin noise to each surface which will be step on. And please
            # provide self.cfg.TerrainPerlin_kwargs for generating Perlin noise
            add_perlin_noise= False,
            border_perlin_noise= False,
            border_height= 0., # Incase we want the surrounding plane to be lower than the track
            virtual_terrain= False,
            draw_virtual_terrain= False, # set True for visualization
            check_skill_combinations= False, # check if some specific skills are connected, if set. e.g. jump -> leap
            engaging_next_threshold= 0., # if > 0, engaging_next is based on this threshold instead of track_block_length/2. Make sure the obstacle is not too long.
            engaging_finish_threshold= 0., # an obstacle is considered finished only if the last volume point is this amount away from the block origin.
            curriculum_perlin= True, # If True, perlin noise scale will be depends on the difficulty if possible.
            no_perlin_threshold= 0.02, # If the perlin noise is too small, clip it to zero.
            walk_in_skill_gap= False, # If True, obstacle ID will be walk when the distance to the obstacle does not reach engaging_next_threshold
        )
    max_track_options = 4 # ("tilt", "crawl", "jump", "dynamic") at most
    track_options_id_dict = {
        "tilt": 1,
        "crawl": 2,
        "jump": 3,
        "leap": 4,
     } # track_id are aranged in this order
    def __init__(self, cfg, num_robots: int) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        
        assert self.cfg.mesh_type == "trimesh", "Not implemented for mesh_type other than trimesh, get {}".format(self.cfg.mesh_type)
        assert getattr(self.cfg, "BarrierTrack_kwargs", None) is not None, "Must provide BarrierTrack_kwargs in cfg.terrain"

        self.track_kwargs.update(self.cfg.BarrierTrack_kwargs)
        if self.track_kwargs["add_perlin_noise"] and not hasattr(self.cfg, "TerrainPerlin_kwargs"):
            print(colorize(
                "Warning: Please provide cfg.terrain.TerrainPerlin to configure perlin noise for all surface to step on.",
                color= "yellow",
            ))

        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= np.float32)

    def initialize_track_info_buffer(self):
        """ Build buffers to store oracle info for each track blocks so that it is faster to compute
        oracle observation. Some dimensions are predefined.
        """
        # For each track block (n_options + 1 in total), 3 parameters are enabled:
        # - track_id: int, starting track is 0, other numbers depends on the options order.
        # - obstacle_depth: float,
        # - obstacle_critical_params: e.g. tilt width, crawl height, jump height

        # num_rows + 1 incase the robot finish the entire row of tracks
        self.track_info_map = torch.zeros(
            (self.cfg.num_rows + 1, self.cfg.num_cols, self.n_blocks_per_track, 3),
            dtype= torch.float32,
            device= self.device,
        )
        self.track_width_map = torch.zeros(
            (self.cfg.num_rows, self.cfg.num_cols),
            dtype= torch.float32,
            device= self.device,
        )
        self.block_starting_height_map = torch.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.n_blocks_per_track),
            dtype= torch.float32,
            device= self.device,
        ) # height [m] related to the world coordinate system

    def initialize_track(self):
        """ All track blocks are defined as follows
            +-----------------------+
            |xxxxxxxxxxxxxxxxxxxxxxx|track wall
            |xxxxxxxxxxxxxxxxxxxxxxx|
            |xxxxxxxxxxxxxxxxxxxxxxx|
            |                       |
            |                       |
            |                       |
            |                       |
            * (env origin)          |
            |                       |
            |                       | ^+y
            |                       | |
            |xxxxxxxxxxxxxxxxxxxxxxx| |
            |xxxxxxxxxxxxxxxxxxxxxxx| |
            |xxxxxxxxxxxxxxxxxxxxxxx| |         +x
            +-----------------------+ +---------->
        
        """
        self.track_block_resolution = (
            np.ceil(self.track_kwargs["track_block_length"] / self.cfg.horizontal_scale).astype(int),
            np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int),
        )
        self.n_blocks_per_track = (self.track_kwargs["n_obstacles_per_track"] + 1) if (self.track_kwargs["randomize_obstacle_order"] and len(self.track_kwargs["options"]) > 0) else (len(self.track_kwargs["options"]) + 1)
        self.track_resolution = (
            np.ceil(self.track_kwargs["track_block_length"] * self.n_blocks_per_track / self.cfg.horizontal_scale).astype(int),
            np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int),
        ) # a track consist of a connected track_blocks
        self.env_block_length = self.track_kwargs["track_block_length"]
        self.env_length = self.track_kwargs["track_block_length"] * self.n_blocks_per_track
        self.env_width = self.track_kwargs["track_width"]

    ##### methods that generate models and critical parameters of each track block #####
    def fill_heightfield_to_scale(self, heightfield):
        """ Due to the rasterization of the heightfield, the trimesh size does not match the 
        heightfield_resolution * horizontal_scale, so we need to fill enlarge heightfield to
        meet this scale.
        """
        assert len(heightfield.shape) == 2, "heightfield must be 2D"
        heightfield_x_fill = np.concatenate([
            heightfield,
            heightfield[-2:, :],
        ], axis= 0)
        heightfield_y_fill = np.concatenate([
            heightfield_x_fill,
            heightfield_x_fill[:, -2:],
        ], axis= 1)
        return heightfield_y_fill

    def get_starting_track(self, wall_thickness):
        track_heighfield_template = np.zeros(self.track_block_resolution, dtype= np.float32)
        track_heighfield_template[:, :np.ceil(
            wall_thickness / self.cfg.horizontal_scale
        ).astype(int)] += ( \
            np.random.uniform(*self.track_kwargs["wall_height"]) \
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
            else self.track_kwargs["wall_height"] \
        ) / self.cfg.vertical_scale
        track_heighfield_template[:, -np.ceil(
            wall_thickness / self.cfg.horizontal_scale
        ).astype(int):] += ( \
            np.random.uniform(*self.track_kwargs["wall_height"]) \
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
            else self.track_kwargs["wall_height"] \
        ) / self.cfg.vertical_scale

        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heighfield_template),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        track_heightfield = track_heighfield_template
        block_info = torch.tensor([
            0., # obstacle depth (along x-axis)
            0., # critical parameter for each obstacle
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px

    def get_jump_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        if isinstance(self.track_kwargs["jump"]["depth"], (tuple, list)):
            if not virtual:
                jump_depth = min(*self.track_kwargs["jump"]["depth"])
            else:
                jump_depth = np.random.uniform(*self.track_kwargs["jump"]["depth"])
        else:
            jump_depth = self.track_kwargs["jump"]["depth"]
        if isinstance(self.track_kwargs["jump"]["height"], (tuple, list)):
            if difficulty is None:
                jump_height = np.random.uniform(*self.track_kwargs["jump"]["height"])
            else:
                jump_height = (1-difficulty) * self.track_kwargs["jump"]["height"][0] + difficulty * self.track_kwargs["jump"]["height"][1]
        else:
            jump_height = self.track_kwargs["jump"]["height"]
        if self.track_kwargs["jump"].get("jump_down_prob", 0.) > 0.:
            if np.random.uniform() < self.track_kwargs["jump"]["jump_down_prob"]:
                jump_height = -jump_height
        depth_px = int(jump_depth / self.cfg.horizontal_scale)
        height_value = jump_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        if not virtual and height_value > 0.:
            track_heightfield[
                1:,
                wall_thickness_px: -wall_thickness_px,
            ] += height_value
        if height_value < 0.:
            track_heightfield[
                (0 if virtual else depth_px):,
                max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
            ] += height_value
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        assert not (
            self.track_kwargs["jump"].get("fake_offset", 0.) != 0. and self.track_kwargs["jump"].get("fake_height", 0.) != 0.), \
            "fake_offset and fake_height cannot be both non-zero"
        jump_height_ = jump_height + (
            self.track_kwargs["jump"].get("fake_offset", 0.) \
            if jump_height > 0. \
            else 0.)
        block_info = torch.tensor([
            jump_depth,
            jump_height_,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = height_value if not virtual else min(height_value, 0)
        return track_trimesh, track_heightfield, block_info, height_offset_px

    def get_tilt_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        tilt_depth = np.random.uniform(*self.track_kwargs["tilt"]["depth"]) if isinstance(self.track_kwargs["tilt"]["depth"], (tuple, list)) else self.track_kwargs["tilt"]["depth"]
        tilt_wall_height = np.random.uniform(*self.track_kwargs["tilt"]["wall_height"]) if isinstance(self.track_kwargs["tilt"]["wall_height"], (tuple, list)) else self.track_kwargs["tilt"]["wall_height"]
        tilt_opening_angle = np.random.uniform(*self.track_kwargs["tilt"]["opening_angle"]) if isinstance(self.track_kwargs["tilt"].get("opening_angle", 0.), (tuple, list)) else self.track_kwargs["tilt"].get("opening_angle", 0.)
        if isinstance(self.track_kwargs["tilt"]["width"], (tuple, list)):
            if difficulty is None:
                tilt_width = np.random.uniform(*self.track_kwargs["tilt"]["width"])
            else:
                tilt_width = difficulty * self.track_kwargs["tilt"]["width"][0] + (1-difficulty) * self.track_kwargs["tilt"]["width"][1]
        else:
            tilt_width = self.track_kwargs["tilt"]["width"]
        depth_px = int(tilt_depth / self.cfg.horizontal_scale)
        height_value = tilt_wall_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        wall_gap_px = int(tilt_width / self.cfg.horizontal_scale / 2)

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        # If index out of limit error occured, it might because of the too large tilt width
        if virtual:
            pass # no modification on the heightfield (so as the trimesh)
        elif tilt_opening_angle == 0:
            track_heightfield[
                1: depth_px+1,
                wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px),
            ] = height_value
            track_heightfield[
                1: depth_px+1,
                int(self.track_block_resolution[1] / 2 + wall_gap_px): -wall_thickness_px,
            ] = height_value
        else:
            for depth_i in range(1, depth_px + 1):
                wall_gap_px_row = wall_gap_px + (depth_px - depth_i) * np.tan(tilt_opening_angle)
                track_heightfield[
                    depth_i,
                    wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px_row),
                ] = height_value
                track_heightfield[
                    depth_i,
                    int(self.track_block_resolution[1] / 2 + wall_gap_px_row): -wall_thickness_px,
                ] = height_value
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        block_info = torch.tensor([
            tilt_depth,
            tilt_width,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px

    def get_crawl_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        crawl_depth = np.random.uniform(*self.track_kwargs["crawl"]["depth"]) if isinstance(self.track_kwargs["crawl"]["depth"], (tuple, list)) else self.track_kwargs["crawl"]["depth"]
        if isinstance(self.track_kwargs["crawl"]["height"], (tuple, list)):
            if difficulty is None:
                crawl_height = np.random.uniform(*self.track_kwargs["crawl"]["height"])
            else:
                crawl_height = difficulty * self.track_kwargs["crawl"]["height"][0] + (1-difficulty) * self.track_kwargs["crawl"]["height"][1]
        else:
            crawl_height = self.track_kwargs["crawl"]["height"]
        crawl_wall_height = np.random.uniform(*self.track_kwargs["crawl"]["wall_height"]) if isinstance(self.track_kwargs["crawl"]["wall_height"], (tuple, list)) else self.track_kwargs["crawl"]["wall_height"]
        
        if not heightfield_noise is None:
            if self.track_kwargs["crawl"].get("no_perlin_at_obstacle", False):
                depth_px = int(crawl_depth / self.cfg.horizontal_scale)
                wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
                heightfield_template = heightfield_template.copy()
                heightfield_template[
                    1: depth_px+1,
                    :wall_thickness_px,
                ] += heightfield_noise[1: depth_px+1, :wall_thickness_px]
                heightfield_template[
                    1: depth_px+1,
                    -max(wall_thickness_px, 1):,
                ] += heightfield_noise[1: depth_px+1, -max(wall_thickness_px, 1):]
                heightfield_template[
                    depth_px+1:,
                ] += heightfield_noise[depth_px+1:]
            else:
                heightfield_template = heightfield_template + heightfield_noise
            trimesh_template = convert_heightfield_to_trimesh(
                self.fill_heightfield_to_scale(heightfield_template),
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
        
        upper_bar_trimesh = trimesh.box_trimesh(
            np.array([
                crawl_depth,
                self.track_kwargs["track_width"] - wall_thickness*2,
                crawl_wall_height,
            ], dtype= np.float32),
            np.array([
                crawl_depth / 2,
                self.track_kwargs["track_width"] / 2,
                crawl_height + crawl_wall_height / 2,
            ], dtype= np.float32),
        )
        if not virtual:
            track_trimesh = trimesh.combine_trimeshes(
                trimesh_template,
                upper_bar_trimesh,
            )
        else:
            track_trimesh = trimesh_template
        block_info = torch.tensor([
            crawl_depth if self.track_kwargs["crawl"].get("fake_depth", 0.) <= 0 else self.track_kwargs["crawl"]["fake_depth"],
            crawl_height,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, heightfield_template, block_info, height_offset_px

    def get_leap_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        leap_depth = np.random.uniform(*self.track_kwargs["leap"]["depth"]) if isinstance(self.track_kwargs["leap"]["depth"], (tuple, list)) else self.track_kwargs["leap"]["depth"]
        if isinstance(self.track_kwargs["leap"]["length"], (tuple, list)):
            if difficulty is None:
                leap_length = np.random.uniform(*self.track_kwargs["leap"]["length"])
            else:
                leap_length = (1-difficulty) * self.track_kwargs["leap"]["length"][0] + difficulty * self.track_kwargs["leap"]["length"][1]
        else:
            leap_length = self.track_kwargs["leap"]["length"]
        length_px = int(leap_length / self.cfg.horizontal_scale)
        depth_value = leap_depth / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        start_px = int(self.track_kwargs["leap"].get("fake_offset", 0.) / self.cfg.horizontal_scale) + 1 if not virtual else 1
        track_heightfield[
            start_px: length_px+1,
            max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
        ] -= depth_value
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        block_info = torch.tensor([
            leap_length + self.track_kwargs["leap"].get("fake_offset", 0.), # along x(forward)-axis
            leap_depth, # along z(downward)-axis
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px

    ##### initialize and building tracks then entire terrain #####
    def build_heightfield_raw(self):
        self.border = int(self.cfg.border_size / self.cfg.horizontal_scale)
        map_x_size = int(self.cfg.num_rows * self.track_resolution[0]) + 2 * self.border
        map_y_size = int(self.cfg.num_cols * self.track_resolution[1]) + 2 * self.border
        self.tot_rows = map_x_size
        self.tot_cols = map_y_size
        self.heightfield_raw = np.zeros((map_x_size, map_y_size), dtype= np.float32)
        if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            for k, v in self.cfg.TerrainPerlin_kwargs.items():
                if isinstance(v, (tuple, list)):
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                    TerrainPerlin_kwargs[k] = v[0]
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(
                xSize= self.env_length * self.cfg.num_rows + 2 * self.cfg.border_size,
                ySize= self.env_width * self.cfg.num_cols + 2 * self.cfg.border_size,
                xSamples= map_x_size,
                ySamples= map_y_size,
                **TerrainPerlin_kwargs,
            ) / self.cfg.vertical_scale
            self.heightfield_raw += heightfield_noise
            self.heightfield_raw[self.border:-self.border, self.border:-self.border] = 0.
            if self.track_kwargs["border_height"] != 0.:
                # self.heightfield_raw[:self.border, :] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                # self.heightfield_raw[-self.border:, :] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                self.heightfield_raw[:, :self.border] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                self.heightfield_raw[:, -self.border:] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
        self.heightsamples = self.heightfield_raw

    def add_trimesh_to_sim(self, trimesh, trimesh_origin):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = trimesh_origin[2]
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_track_to_sim(self, track_origin_px, row_idx= None, col_idx= None):
        """ add heighfield value and add trimesh to sim for one certain race track """
        # adding trimesh and heighfields
        if "one_obstacle_per_track" in self.track_kwargs.keys():
            print("Warning: one_obstacle_per_track is deprecated, use n_obstacles_per_track instead.")
        if self.track_kwargs["randomize_obstacle_order"] and len(self.track_kwargs["options"]) > 0:
            obstacle_order = np.random.choice(
                len(self.track_kwargs["options"]),
                size= self.track_kwargs.get("n_obstacles_per_track", 1),
                replace= True,
            )
        else:
            obstacle_order = np.arange(len(self.track_kwargs["options"]))
        difficulties = self.get_difficulty(row_idx, col_idx)
        difficulty, virtual_track = difficulties[:2]

        if self.track_kwargs["add_perlin_noise"]:
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            for k, v in self.cfg.TerrainPerlin_kwargs.items():
                if isinstance(v, (tuple, list)):
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                    if difficulty is None or (not self.track_kwargs["curriculum_perlin"]):
                        TerrainPerlin_kwargs[k] = np.random.uniform(*v)
                    else:
                        TerrainPerlin_kwargs[k] = v[0] * (1 - difficulty) + v[1] * difficulty
                    if self.track_kwargs["no_perlin_threshold"] > TerrainPerlin_kwargs[k]:
                        TerrainPerlin_kwargs[k] = 0.
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(
                xSize= self.env_length,
                ySize= self.env_width,
                xSamples= self.track_resolution[0],
                ySamples= self.track_resolution[1],
                **TerrainPerlin_kwargs,
            ) / self.cfg.vertical_scale

        block_starting_height_px = track_origin_px[2]
        wall_thickness = np.random.uniform(*self.track_kwargs["wall_thickness"]) if isinstance(self.track_kwargs["wall_thickness"], (tuple, list)) else self.track_kwargs["wall_thickness"]
        starting_trimesh, starting_heightfield, block_info, height_offset_px = self.get_starting_track(wall_thickness)
        self.heightfield_raw[
            track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0],
            track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1],
        ] = starting_heightfield
        if "heightfield_noise" in locals():
            self.heightfield_raw[
                track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0],
                track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1],
            ] += heightfield_noise[:self.track_block_resolution[0]]
            starting_trimesh_noised = convert_heightfield_to_trimesh(
                self.fill_heightfield_to_scale(self.heightfield_raw[
                    track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0],
                    track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1],
                ]),
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
            self.add_trimesh_to_sim(starting_trimesh_noised,
                np.array([
                    track_origin_px[0] * self.cfg.horizontal_scale,
                    track_origin_px[1] * self.cfg.horizontal_scale,
                    block_starting_height_px * self.cfg.vertical_scale,
                ]))
        else:
            self.add_trimesh_to_sim(starting_trimesh,
                np.array([
                    track_origin_px[0] * self.cfg.horizontal_scale,
                    track_origin_px[1] * self.cfg.horizontal_scale,
                    block_starting_height_px * self.cfg.vertical_scale,
                ]))
        self.track_info_map[row_idx, col_idx, 0, 0] = 0
        self.track_info_map[row_idx, col_idx, 0, 1:] = block_info
        self.track_width_map[row_idx, col_idx] = self.env_width - wall_thickness * 2
        self.block_starting_height_map[row_idx, col_idx, 0] = block_starting_height_px * self.cfg.vertical_scale
        block_starting_height_px += height_offset_px
        
        for obstacle_idx, obstacle_selection in enumerate(obstacle_order):
            obstacle_name = self.track_kwargs["options"][obstacle_selection]
            obstacle_id = self.track_options_id_dict[obstacle_name]
            # call method to generate trimesh and heightfield for each track block.
            # For example get_jump_track, get_tilt_track
            # using `virtual_track` to create non-collision mesh for collocation method in training.
            track_trimesh, track_heightfield, block_info, height_offset_px = getattr(self, "get_" + obstacle_name + "_track")(
                wall_thickness,
                starting_trimesh,
                starting_heightfield,
                difficulty= difficulty,
                heightfield_noise= heightfield_noise[
                    self.track_block_resolution[0] * (obstacle_idx + 1): self.track_block_resolution[0] * (obstacle_idx + 2)
                ] if "heightfield_noise" in locals() else None,
                virtual= virtual_track,
            )

            heightfield_x0 = track_origin_px[0] + self.track_block_resolution[0] * (obstacle_idx + 1)
            heightfield_y0 = track_origin_px[1]
            heightfield_x1 = track_origin_px[0] + self.track_block_resolution[0] * (obstacle_idx + 2)
            heightfield_y1 = track_origin_px[1] + self.track_block_resolution[1]

            self.heightfield_raw[
                heightfield_x0: heightfield_x1,
                heightfield_y0: heightfield_y1,
            ] = track_heightfield
            self.add_trimesh_to_sim(
                track_trimesh,
                np.array([
                    heightfield_x0 * self.cfg.horizontal_scale,
                    heightfield_y0 * self.cfg.horizontal_scale,
                    block_starting_height_px * self.cfg.vertical_scale,
                ])
            )
            self.track_info_map[row_idx, col_idx, obstacle_idx + 1, 0] = obstacle_id
            self.track_info_map[row_idx, col_idx, obstacle_idx + 1, 1:] = block_info
            self.block_starting_height_map[row_idx, col_idx, obstacle_idx + 1] = block_starting_height_px * self.cfg.vertical_scale
            block_starting_height_px += height_offset_px

        return block_starting_height_px

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ Add current terrain as trimesh to sim and update the corresponding heightfield """
        self.gym = gym
        self.sim = sim
        self.device = device
        self.initialize_track()
        self.build_heightfield_raw()
        self.initialize_track_info_buffer()

        self.track_origins_px = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= int)
        for col_idx in range(self.cfg.num_cols):
            starting_height_px = 0
            for row_idx in range(self.cfg.num_rows):
                self.track_origins_px[row_idx, col_idx] = [
                    int(row_idx * self.track_resolution[0]) + self.border,
                    int(col_idx * self.track_resolution[1]) + self.border,
                    starting_height_px,
                ]
                # NOTE: The starting heigh is passed to the `add_track_to_sim`.
                # The return value of `add_track_to_sim` is the z value, not the offset.
                starting_height_px = self.add_track_to_sim(
                    self.track_origins_px[row_idx, col_idx],
                    row_idx= row_idx,
                    col_idx= col_idx,
                )
        
        self.add_plane_to_sim(starting_height_px)
        
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                self.env_origins[i, j, 0] = self.track_origins_px[i, j, 0] * self.cfg.horizontal_scale
                self.env_origins[i, j, 1] = self.track_origins_px[i, j, 1] * self.cfg.horizontal_scale
                self.env_origins[i, j, 2] = self.track_origins_px[i, j, 2] * self.cfg.vertical_scale
                self.env_origins[i, j, 1] += self.track_kwargs["track_width"] / 2
        self.env_origins_pyt = torch.from_numpy(self.env_origins).to(self.device)

    def add_plane_to_sim(self, final_height_px= 0.):
        """
        Args:
            final_height_px: the height of the region 1 in the following figure.
        """
        if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:
            """
                 +----------------------------------------------------------------+
                 |                                                                |
                 |                                                                |
                 |region2                                                         |
                 +---------+-------------------------------------------+----------+
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |              The track grid               |          |
                 |         |                                           |          |
                y|         |                                           |          |
                ^|         |                                           |          |
                ||region3  |                                           |   region1|
                |+---------+-------------------------------------------+----------+
                ||                                                                |
                ||                                                                |
                ||region0                                                         |
                |+----------------------------------------------------------------+
                +--------------->x
            """
            trimesh_origins = [
                [0, 0, 0],
                [self.cfg.border_size + self.cfg.num_rows * self.env_length, self.cfg.border_size, final_height_px * self.cfg.vertical_scale],
                [0, self.cfg.border_size + self.cfg.num_cols * self.env_width, 0],
                [0, self.cfg.border_size, 0],
            ]
            heightfield_regions = [
                [slice(0, self.heightfield_raw.shape[0]), slice(0, self.border)],
                [
                    slice(self.heightfield_raw.shape[0] - self.border, self.heightfield_raw.shape[0]),
                    slice(self.border, self.border + self.track_resolution[1] * self.cfg.num_cols),
                ],
                [
                    slice(0, self.heightfield_raw.shape[0]),
                    slice(self.heightfield_raw.shape[1] - self.border, self.heightfield_raw.shape[1]),
                ],
                [slice(0, self.border), slice(self.border, self.heightfield_raw.shape[1] - self.border)],
            ]
            for origin, heightfield_region in zip(trimesh_origins, heightfield_regions):
                plane_trimesh = convert_heightfield_to_trimesh(
                    self.fill_heightfield_to_scale(
                        self.heightfield_raw[
                            heightfield_region[0],
                            heightfield_region[1],
                        ]
                    ),
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
                self.add_trimesh_to_sim(plane_trimesh, origin)
        else:
            plane_size_x = self.heightfield_raw.shape[0] * self.cfg.horizontal_scale
            plane_size_y = self.heightfield_raw.shape[1] * self.cfg.horizontal_scale
            plane_box_size = np.array([plane_size_x, plane_size_y, 0.02])
            plane_trimesh = trimesh.box_trimesh(plane_box_size, plane_box_size / 2)
            self.add_trimesh_to_sim(plane_trimesh, np.zeros(3))

    ##### Helper functions to compute observations that are only available in simulation #####
    def get_difficulty(self, env_row_idx, env_col_idx):
        difficulty = env_row_idx / (self.cfg.num_rows - 1) if self.cfg.curriculum else None
        virtual_terrain = self.track_kwargs["virtual_terrain"]
        return difficulty, virtual_terrain

    def get_track_idx(self, base_positions, clipped= True):
        """ base_positions: torch tensor (n, 3) float; Return: (n, 2) int """
        track_idx = torch.floor((base_positions[:, :2] - self.cfg.border_size) / torch.tensor([self.env_length, self.env_width], device= base_positions.device)).to(int)
        if clipped:
            track_idx = torch.clip(
                track_idx,
                torch.zeros_like(track_idx[0]),
                torch.tensor([self.cfg.num_rows-1, self.cfg.num_cols-1], dtype= int, device= self.device),
            )
        return track_idx

    def get_stepping_obstacle_info(self, positions):
        """ Different from `engaging`..., this method extracts the obstacle id where the point is
        currently in. If the point is passed the obstacle but still in the assigned block, it is 
        considered no stepping in the obstacle and the ID is 0.
        """
        track_idx = self.get_track_idx(positions, clipped= False)
        track_idx_clipped = self.get_track_idx(positions)
        forward_distance = positions[:, 0] - self.cfg.border_size - (track_idx[:, 0] * self.env_length) # (N,) w.r.t a track
        block_idx = torch.floor(forward_distance / self.env_block_length).to(int) # (N,)
        block_idx_clipped = torch.clip(
            block_idx,
            0,
            (self.n_blocks_per_track - 1),
        )
        in_track_mask = (track_idx == track_idx_clipped).all(dim= -1) & (block_idx == block_idx_clipped)
        in_block_distance = forward_distance % self.env_block_length
        obstacle_info = self.track_info_map[
            track_idx_clipped[:, 0],
            track_idx_clipped[:, 1],
            block_idx_clipped,
        ] # (N, 3)
        obstacle_info[in_block_distance > obstacle_info[:, 1]] = 0.
        obstacle_info[torch.logical_not(in_track_mask)] = 0.
        
        return obstacle_info

    def get_engaging_block_info(self, base_positions, volume_points_offset):
        """ Get the closet forward obstacle distance in x-axis """
        num_robots = base_positions.shape[0]
        track_idx = self.get_track_idx(base_positions, clipped= False)
        track_idx_clipped = self.get_track_idx(base_positions) # (n, 2)

        # compute which block the robot is currently in (w.r.t the track)
        forward_distance = base_positions[:, 0] - self.cfg.border_size - (track_idx_clipped[:, 0] * self.env_length)
        block_idx = torch.floor(forward_distance / self.env_block_length).to(int) # (n,)
        block_idx_clipped = torch.clip(
            block_idx,
            0.,
            (self.n_blocks_per_track - 1),
        )

        # compute whether the robot is still in any of the track
        in_track_mask = (track_idx == track_idx_clipped).all(dim= -1) & (block_idx == block_idx_clipped)

        # compute whether the robot is engaging with the next block
        engaging_obstacle_depth = self.track_info_map[
            track_idx_clipped[:, 0],
            track_idx_clipped[:, 1],
            block_idx_clipped,
            1,
        ] # (n,)
        in_block_distance = forward_distance % self.env_block_length
        engaging_next_distance = (self.env_block_length - self.track_kwargs["engaging_next_threshold"]) \
            if self.track_kwargs["engaging_next_threshold"] > 0 \
            else self.env_block_length / 2
        min_x_points_offset = torch.min(volume_points_offset[:, :, 0], dim= -1)[0] # (n,)
        engaging_next_block = ((in_block_distance + min_x_points_offset) > engaging_obstacle_depth) \
            & (in_block_distance > self.track_kwargs.get("engaging_finish_threshold", 0.)) \
            & (in_block_distance > engaging_next_distance)

        # update the engaging track_idx and block_idx if engaging next
        engaging_next_track = (block_idx == (self.n_blocks_per_track - 1)) & engaging_next_block
        track_idx_selection = track_idx_clipped.detach().clone()
        block_idx_selection = block_idx_clipped.detach().clone()
        track_idx_selection[engaging_next_track, 0] += 1
        assert track_idx_selection[:, 0].max() < self.track_info_map.shape[0], track_idx_selection[:, 0].max()
        block_idx_selection[engaging_next_block] += 1
        block_idx_selection[engaging_next_track | (~in_track_mask)] = 0
        assert block_idx_selection.max() <= (self.n_blocks_per_track - 1), block_idx_selection.max()

        # compute the engagin distance and extract the block_info
        engaging_distance = track_idx_selection[:, 0] * self.env_length + block_idx_selection * self.env_block_length + self.cfg.border_size
        engaging_distance = engaging_distance - base_positions[:, 0]
        engaging_block_info = self.track_info_map[
            track_idx_selection[:, 0],
            track_idx_selection[:, 1],
            block_idx_selection,
            1:
        ]
        engaging_obstacle_onehot = torch.zeros(
            (num_robots, self.max_track_options + 1),
            dtype= torch.float32,
            device= self.device,
        ) # (n, n_obstacle + 1)
        obstacle_id_selection = self.track_info_map[
            track_idx_selection[:, 0],
            track_idx_selection[:, 1],
            block_idx_selection,
            0
        ].to(int)
        if self.track_kwargs.get("walk_in_skill_gap", False):
            between_skill_mask = ((in_block_distance + min_x_points_offset) > engaging_obstacle_depth) \
                & (in_block_distance > self.track_kwargs.get("engaging_finish_threshold", 0.)) \
                & (in_block_distance < engaging_next_distance)
            obstacle_id_selection[between_skill_mask] = 0.
            engaging_block_info[between_skill_mask] = 0.
        engaging_obstacle_onehot[
            torch.arange(num_robots, device= self.device),
            obstacle_id_selection,
        ] = 1

        # concat and return
        engaging_block_info = torch.cat([
            engaging_distance.unsqueeze(-1),
            engaging_obstacle_onehot,
            engaging_block_info,
        ], dim= -1) # (n, 1 + (n_obstacle+1) + 2)
        return engaging_block_info

    @staticmethod
    def get_engaging_block_info_shape(cfg):
        """ The block info which is expected to be engaged
        - obstacle distance (1,), if negative means the robot is in the obstacle
        - obstacle id (onehot)
        - obstacle info (2,)
        """
        return (1 + (BarrierTrack.max_track_options + 1) + 2),

    def get_sidewall_distance(self, base_positions):
        """ Get the distances toward the sidewall where the track the robot is in """
        track_idx = self.get_track_idx(base_positions)
        track_width = self.track_width_map[track_idx[:, 0], track_idx[:, 1]]
        y_positions_in_track = base_positions[:, 1] - self.env_origins_pyt[track_idx[:, 0], track_idx[:, 1], 1]
        distance_pos_y = (track_width / 2) - y_positions_in_track
        distance_neg_y = y_positions_in_track - (-track_width / 2)
        return torch.stack([
            distance_pos_y,
            distance_neg_y,
        ], dim= -1) # (n, 2)

    def get_jump_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        in_up_mask = torch.logical_and(
            positions_in_block[:, 0] <= block_infos[:, 1],
            block_infos[:, 2] > 0.,
        )
        in_down_mask = torch.logical_and(
            positions_in_block[:, 0] <= block_infos[:, 1],
            block_infos[:, 2] < 0.,
        )
        jump_over_mask = torch.logical_and(
            positions_in_block[:, 2] > block_infos[:, 2],
            positions_in_block[:, 2] > 0, # to avoid the penetration of virtual obstacle in jump down.
        ) # (n_points,)

        penetrated_mask = torch.logical_and(
            torch.logical_or(in_up_mask, in_down_mask),
            (torch.logical_not(jump_over_mask)),
        )
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetrate_up_mask = torch.logical_and(penetrated_mask, in_up_mask)
        penetration_depths_buffer[penetrate_up_mask] = block_infos[penetrate_up_mask, 2] - positions_in_block[penetrate_up_mask, 2]
        penetrate_down_mask = torch.logical_and(penetrated_mask, in_down_mask)
        penetration_depths_buffer[penetrate_down_mask] = 0. - positions_in_block[penetrate_down_mask, 2]
        return penetration_depths_buffer

    def get_tilt_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        in_obstacle_mask = positions_in_block[:, 0] <= block_infos[:, 1]
        penetrated_mask = torch.logical_and(
            in_obstacle_mask,
            torch.abs(positions_in_block[:, 1]) > (block_infos[:, 2] / 2)
        )
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetration_depths_buffer[penetrated_mask] = torch.abs(positions_in_block[penetrated_mask, 1]) - (block_infos[penetrated_mask, 2] / 2)
        return penetration_depths_buffer

    def get_crawl_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        in_obstacle_mask = positions_in_block[:, 0] <= block_infos[:, 1]
        crash_obstacle_mask = positions_in_block[:, 2] > block_infos[:, 2]
        penetrated_mask = torch.logical_and(in_obstacle_mask, crash_obstacle_mask)
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetration_depths_buffer[penetrated_mask] = positions_in_block[penetrated_mask, 2] - block_infos[penetrated_mask, 2]
        return penetration_depths_buffer

    def get_leap_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        expected_leap_height = self.track_kwargs["leap"]["height"]
        in_obstacle_mask = positions_in_block[:, 0] <= block_infos[:, 1]
        crash_obstacle_mask = positions_in_block[:, 2] < expected_leap_height
        penetrated_mask = torch.logical_and(in_obstacle_mask, crash_obstacle_mask)
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetration_depths_buffer[penetrated_mask] = expected_leap_height - positions_in_block[penetrated_mask, 2]
        return penetration_depths_buffer

    def get_penetration_depths(self, sample_points, mask_only= False):
        """ Compute the penetrations of the sample points w.r.t the virtual obstacle.
        NOTE: this implementation is specifically tailored to these 3 obstacles.
            Review the logic if you need to add other types of obstacles.
        Args:
            sample_points: torch tensor of shape (N, 3) with world-frame coordinates (not related to env origin)
                s is the number of sample points for each env.
            mask_only: If set, the depth will only be 0 (no penetration) or 1 (have penetration)
        Return:
            penetration_depth: torch tensor (N,) >= 0.
        """
        track_idx = self.get_track_idx(sample_points, clipped= False)
        track_idx_clipped = self.get_track_idx(sample_points)
        in_track_mask = (track_idx[:, 0] >= 0) \
            & (track_idx[:, 1] >= 0) \
            & (track_idx[:, 0] < self.cfg.num_rows) \
            & (track_idx[:, 1] < self.cfg.num_cols)
        forward_distance = sample_points[:, 0] - self.cfg.border_size - (track_idx[:, 0] * self.env_length) # (N,) w.r.t a track
        block_idx = torch.floor(forward_distance / self.env_block_length).to(int) # (N,) 
        block_idx[block_idx >= self.track_info_map.shape[2]] = 0.
        positions_in_block = torch.stack([
            forward_distance % self.env_block_length,
            sample_points[:, 1] - self.env_origins_pyt[track_idx_clipped[:, 0], track_idx_clipped[:, 1]][:, 1],
            sample_points[:, 2] - self.block_starting_height_map[track_idx_clipped[:, 0], track_idx_clipped[:, 1], block_idx],
        ], dim= -1) # (N, 3) related to the origin of the block, not the track.
        block_infos = self.track_info_map[track_idx_clipped[:, 0], track_idx_clipped[:, 1], block_idx] # (N, 3)

        penetration_depths = torch.zeros_like(sample_points[:, 0]) # shape (N,)
        for obstacle_name, obstacle_id in self.track_options_id_dict.items():
            point_masks = (block_infos[:, 0] == obstacle_id) & (in_track_mask)
            if not point_masks.any(): continue
            penetration_depths[point_masks] = getattr(self, "get_" + obstacle_name + "_penetration_depths")(
                block_infos[point_masks],
                positions_in_block[point_masks],
                mask_only= mask_only,
            )
        penetration_depths[torch.logical_not(in_track_mask)] = 0.

        return penetration_depths

    def get_penetration_mask(self, sample_points):
        """ This is an approximate method to count how many points have penetrated the virtual obstacle.
        """
        return self.get_penetration_depths(sample_points, mask_only= True)

    def get_passed_obstacle_depths(self, rows, cols, distances):
        """ Get total obstacle depths (length) the distances have gone through,
        to normalize the evalution threshold.
        The evaluation threshold should be linear to the obstacle depth and type.
        Args:
            rows, cols: int vector of same shape (n,)
            distances: float vector of the same shape as rows
        Return:
            passed_depths: float vector of shape (n,)
        """
        n = rows.shape[0]
        device = rows.device
        distances = torch.clone(distances).detach()
        passed_depths = torch.zeros_like(distances)
        while True:
            track_infos = self.track_info_map[rows, cols] # (n, n_blocks, 3)
            # get the block indices the distances is currently in.
            block_indices = torch.clip(torch.floor(distances / self.env_block_length), max= self.track_info_map.shape[2]-1).to(int) # (n,)
            block_indices = torch.clip(block_indices, min= 0) # (n,)
            passed_blocks_mask = torch.arange(self.track_info_map.shape[2], device= device).repeat(n, 1) < block_indices.unsqueeze(-1) # (n, n_blocks)

            passed_depths += torch.sum(track_infos[:, :, 1] * passed_blocks_mask, dim= -1) # (n,)

            distance_in_between = torch.clip(distances - block_indices * self.env_block_length, min= 0) # (n,)
            in_between_block_depth = torch.clip(distance_in_between, max= track_infos[torch.arange(n, device= device), block_indices, 1])
            passed_depths += in_between_block_depth

            rows += 1
            rows = torch.clip(rows, min= 0)
            rows = torch.clip(rows, max= self.track_info_map.shape[0] - 1)
            distances -= self.env_length
            if (rows >= (self.track_info_map.shape[0] - 1)).all() or (distances <= 0).all():
                break

        return passed_depths

    ######## methods to draw visualization #######################
    def draw_virtual_jump_track(self,
            block_info,
            block_origin,
        ):
        jump_depth = block_info[1]
        jump_height = block_info[2]
        geom = gymutil.WireframeBoxGeometry(
            jump_depth if jump_height > 0 else self.track_kwargs["jump"].get("down_forward_length", 0.1),
            self.env_width,
            jump_height,
            pose= None,
            color= (0, 0, 1),
        )
        pose = gymapi.Transform(gymapi.Vec3(
            jump_depth/2 + block_origin[0],
            block_origin[1],
            jump_height/2 + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )

    def draw_virtual_tilt_track(self,
            block_info,
            block_origin,
        ):
        tilt_depth = block_info[1]
        tilt_width = block_info[2]
        wall_height = self.track_kwargs["tilt"]["wall_height"]
        geom = gymutil.WireframeBoxGeometry(
            tilt_depth,
            (self.env_width - tilt_width) / 2,
            wall_height,
            pose= None,
            color= (0, 0, 1),
        )
        
        pose = gymapi.Transform(gymapi.Vec3(
            tilt_depth/2 + block_origin[0],
            block_origin[1] + (self.env_width + tilt_width) / 4,
            wall_height/2 + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )
        pose = gymapi.Transform(gymapi.Vec3(
            tilt_depth/2 + block_origin[0],
            block_origin[1] - (self.env_width + tilt_width) / 4,
            wall_height/2 + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )

    def draw_virtual_crawl_track(self,
            block_info,
            block_origin,
        ):
        crawl_depth = block_info[1]
        crawl_height = block_info[2]
        wall_height = self.track_kwargs["crawl"]["wall_height"][1] if isinstance(self.track_kwargs["crawl"]["wall_height"], (list, tuple)) else self.track_kwargs["crawl"]["wall_height"]
        geom = gymutil.WireframeBoxGeometry(
            crawl_depth,
            self.env_width,
            wall_height,
            pose= None,
            color= (0, 0, 1),
        )
        pose = gymapi.Transform(gymapi.Vec3(
            crawl_depth/2 + block_origin[0],
            block_origin[1],
            wall_height/2 + crawl_height + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )

    def draw_virtual_leap_track(self,
            block_info,
            block_origin,
        ):
        # virtual/non-virtual terrain looks the same when leaping the gap.
        # but the expected height can be visualized
        leap_length = block_info[1]
        expected_height = self.track_kwargs["leap"]["height"]
        geom = gymutil.WireframeBoxGeometry(
            leap_length,
            self.env_width,
            expected_height,
            pose= None,
            color= (0, 0.5, 0.5),
        )
        pose = gymapi.Transform(gymapi.Vec3(
            leap_length/2 + block_origin[0],
            block_origin[1],
            expected_height/2 + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )
        

    def draw_virtual_track(self,
            row_idx,
            col_idx,
        ):
        difficulties = self.get_difficulty(row_idx, col_idx)
        virtual_terrain = difficulties[1]
        track_origin = self.env_origins[row_idx, col_idx]

        for block_idx in range(1, self.track_info_map.shape[2]):
            if virtual_terrain and self.track_kwargs["draw_virtual_terrain"]:
                obstacle_id = self.track_info_map[row_idx, col_idx, block_idx, 0]
                for k, v in self.track_options_id_dict.items():
                    if v == obstacle_id:
                        obstacle_name = k
                        break
                block_info = self.track_info_map[row_idx, col_idx, block_idx] # (3,)
                getattr(self, "draw_virtual_" + obstacle_name + "_track")(
                    block_info,
                    np.array([
                        track_origin[0] + self.track_kwargs["track_block_length"] * block_idx,
                        track_origin[1],
                        self.block_starting_height_map[row_idx, col_idx, block_idx].cpu().numpy(),
                    ]),
                )

    def draw_virtual_terrain(self, viewer):
        self.viewer = viewer

        for row_idx in range(self.cfg.num_rows):
            for col_idx in range(self.cfg.num_cols):
                self.draw_virtual_track(
                    row_idx= row_idx,
                    col_idx= col_idx,
                )
