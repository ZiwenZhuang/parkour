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
                depth= 0.1, # size along the forward axis
                fake_offset= 0.0, # [m] fake offset will make jump's height info greater than its physical height.
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
            hurdle= dict(
                height= 0.3,
                depth= 0.1, # size along the forward axis
            ),
            down= dict(
                height= 0.3,
                depth= 0.1, # size along the forward axis
            ),
            tilted_ramp= dict(
                tilt_angle= 0.3, # [rad]
                switch_spacing= 0., # [m] if set, after each spacing, the ramp will switch direction and use overlap_size argument.
                spacing_curriculum= False, # if True, the spacing will be also be set in curriculum mode.
                overlap_size= 0.1, # [m] if switch_spacing, two consective ramps will overlap along y-axis, check the implementation.
                depth= 0.2, # [m] the depths where the robot should not step on.
                length= 0., # [m] the length of the ramp along x-axis, after this, will be flat terrain.
            ),
            slope= dict(
                slope_angle= [0.2, 0.5], # [rad] the angle of the slope (related to difficulty)
                face_angle= [-3.14, 3.14], # [rad] the angle of the slope upward between x-axis
                length= [1.2, 2.], # [m] the length of the slope along x-axis
                use_mean_height_offset= True, # if set, the height offset for following terrain. Otherwise, zero.
                no_perlin_rate= 0.5, # the rate of no perlin noise on the slope
                length_curriculum= False, # if True, length will depend on difficulty if possible. greater length as more difficulty.
            ),
            stairsup= dict(
                height= 0.2, # [m] the height of each step
                length= 0.25, # [m] the depth of each step
                length_curriculum= False, # if True, length will depend on difficulty if possible.
                residual_distance= 0.05, # [m] the distance of volume points to the step edge, during penetration depth computation
                num_steps= None, # if set, the stairs will not necessarily finish the block
                num_steps_curriculum= False, # if True, num_steps will depend on difficulty
            ),
            stairsdown= dict(
                height= 0.2, # [m] the height of each step
                length= 0.25, # [m] the depth of each step
                length_curriculum= False, # if True, length will depend on difficulty if possible.
                num_steps= None, # if set, the stairs will not necessarily finish the block
                num_steps_curriculum= False, # if True, num_steps will depend on difficulty
            ),
            discrete_rect= dict(
                max_height= 0.2, # [m] the maximum height of the rectangular block (both-positive and negative)
                max_size= 0.8, # [m] the maximum size of the rectangular block
                min_size= 0.2, # [m] the minimum size of the rectangular block
                num_rects= 16, # number of rectangles in the terrain
            ),
            slopeup= dict(
                slope_angle= [0.2, 0.5],
                face_angle= [-0.3, 0.3],
                length= [1.2, 2.],
                use_mean_height_offset= True,
                no_perlin_rate= 0.5,
                length_curriculum= False,
            ),
            slopedown= dict(
                slope_angle= [0.2, 0.5], # filled to slope track with reversed values.
                face_angle= [-0.3, 0.3],
                length= [1.2, 2.],
                use_mean_height_offset= True,
                no_perlin_rate= 0.5,
                length_curriculum= False,
            ),
            wave= dict(
                amplitude= 0.1, # [m] the amplitude of the wave
                frequency= 1, # number of waves in the terrain
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
    block_info_dim = 2 # size along x-axis, obstacle_critical_params (constant after initialization)
    max_track_options = 200 # make track options at most 200 types, which means max track id is 199
    track_options_id_dict = {
        "tilt": 1,
        "crawl": 2,
        "jump": 3,
        "leap": 4,
        "hurdle": 5,
        "down": 6,
        "tilted_ramp": 7, # Notice the underscore. This is a single word.
        "slope": 8,
        "stairsup": 9,
        "stairsdown": 10,
        "discrete_rect": 11,
        "slopeup": 12, # slopeup and slopedown are special cases of slope
        "slopedown": 13,
        "wave": 14,
     } # track_id are aranged in this order
    def __init__(self, cfg, num_robots: int) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        
        assert self.cfg.mesh_type is None, "Not implemented for mesh_type other than None, get {}".format(self.cfg.mesh_type)
        assert getattr(self.cfg, "BarrierTrack_kwargs", None) is not None, "Must provide BarrierTrack_kwargs in cfg.terrain"

        for k, v in self.track_kwargs.items():
            if not k in self.cfg.BarrierTrack_kwargs:
                continue
            if isinstance(v, dict):
                self.track_kwargs[k].update(self.cfg.BarrierTrack_kwargs[k])
            else:
                self.track_kwargs[k] = self.cfg.BarrierTrack_kwargs[k]
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
            (self.cfg.num_rows, self.cfg.num_cols, self.n_blocks_per_track, 1 + self.block_info_dim),
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
        self.engaging_next_min_forward_distance = (self.env_block_length - self.track_kwargs["engaging_next_threshold"]) \
            if self.track_kwargs["engaging_next_threshold"] > 0 \
            else self.env_block_length / 2

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
            print("Warning: jump_down_prob is dereprecated. Please use another option `down` instead.")
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
        if virtual and height_value > 0.:
            # fake sensor reading in heightfield
            track_heightfield[
                1:depth_px+1,
                wall_thickness_px: -wall_thickness_px,
            ] += height_value
        # In non virtual mode, fake_offset only affects penetration computation.
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
        if not virtual and tilt_opening_angle == 0:
            track_heightfield[
                1: depth_px+1,
                wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px),
            ] = height_value
            track_heightfield[
                1: depth_px+1,
                int(self.track_block_resolution[1] / 2 + wall_gap_px): -wall_thickness_px,
            ] = height_value
        elif not virtual:
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
        if virtual:
            # fake the height values
            track_heightfield[
                1: depth_px+1,
                wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px),
            ] = height_value
            track_heightfield[
                1: depth_px+1,
                int(self.track_block_resolution[1] / 2 + wall_gap_px): -wall_thickness_px,
            ] = height_value
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
        if start_px > 1:
            # fake the terrain height so that the oracle policy will leap earlier
            track_heightfield[
                1: start_px,
                max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
            ] -= depth_value
        block_info = torch.tensor([
            leap_length + self.track_kwargs["leap"].get("fake_offset", 0.), # along x(forward)-axis
            leap_depth, # along z(downward)-axis
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_hurdle_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        """ Use almost the same coding principle as get_jump_track """
        if isinstance(self.track_kwargs["hurdle"]["depth"], (tuple, list)):
            hurdle_depth = np.random.uniform(*self.track_kwargs["hurdle"]["depth"])
        else:
            hurdle_depth = self.track_kwargs["hurdle"]["depth"]
        if isinstance(self.track_kwargs["hurdle"]["height"], (tuple, list)):
            if difficulty is None:
                hurdle_height = np.random.uniform(*self.track_kwargs["hurdle"]["height"])
            else:
                hurdle_height = (1-difficulty) * self.track_kwargs["hurdle"]["height"][0] + difficulty * self.track_kwargs["hurdle"]["height"][1]
        else:
            hurdle_height = self.track_kwargs["hurdle"]["height"]
        depth_px = int(hurdle_depth / self.cfg.horizontal_scale)
        height_value = hurdle_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1

        if self.track_kwargs["hurdle"].get("curved_top_rate", 0.) > 0. and np.random.uniform() < self.track_kwargs["hurdle"]["curved_top_rate"]:
            # add curved top plane as the hurdle
            height_value = np.ones(depth_px, dtype= np.float32) * height_value
            height_value = height_value * (1 - np.square(np.linspace(-1, 1, depth_px)) * 0.5)
            height_value = height_value.reshape(-1, 1)
        
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        if not virtual:
            track_heightfield[
                1:depth_px+1,
                wall_thickness_px: -wall_thickness_px,
            ] += height_value
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        if virtual:
            track_heightfield[
                1:depth_px+1,
                wall_thickness_px: -wall_thickness_px,
            ] += height_value
        # In non virtual mode, fake_offset only affects penetration computation.
        hurdle_height_ = hurdle_height + self.track_kwargs["hurdle"].get("fake_offset", 0.)
        block_info = torch.tensor([
            hurdle_depth,
            hurdle_height_,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0.
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_down_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        if isinstance(self.track_kwargs["down"]["depth"], (tuple, list)):
            if not virtual:
                down_depth = min(*self.track_kwargs["down"]["depth"])
            else:
                down_depth = np.random.uniform(*self.track_kwargs["down"]["depth"])
        else:
            down_depth = self.track_kwargs["down"]["depth"]
        if isinstance(self.track_kwargs["down"]["height"], (tuple, list)):
            if difficulty is None:
                down_height = np.random.uniform(*self.track_kwargs["down"]["height"])
            else:
                down_height = (1-difficulty) * self.track_kwargs["down"]["height"][0] + difficulty * self.track_kwargs["down"]["height"][1]
        else:
            down_height = self.track_kwargs["down"]["height"]
        down_height = -down_height if down_height > 0. else down_height
        depth_px = int(down_depth / self.cfg.horizontal_scale)
        height_value = down_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
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
        if virtual:
            # fake sensor reading in heightfield as if there is not depth change at the beginning of the block.
            track_heightfield[
                :depth_px,
                wall_thickness_px: -wall_thickness_px,
            ] += height_value
        block_info = torch.tensor([
            down_depth,
            down_height,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = height_value
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_tilted_ramp_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        """ By default, the entire track is tilted ramp. When `length` is set ( > 0.),
        the last part of the track will be flat (perlin only).
        """
        if isinstance(self.track_kwargs["tilted_ramp"]["depth"], (tuple, list)):
            depth = np.random.uniform(*self.track_kwargs["tilted_ramp"]["depth"])
        else:
            depth = self.track_kwargs["tilted_ramp"]["depth"]
        if isinstance(self.track_kwargs["tilted_ramp"]["tilt_angle"], (tuple, list)):
            if difficulty is not None:
                tilt_angle = (1-difficulty) * self.track_kwargs["tilted_ramp"]["tilt_angle"][0] + difficulty * self.track_kwargs["tilted_ramp"]["tilt_angle"][1]
            else:
                tilt_angle = np.random.uniform(*self.track_kwargs["tilted_ramp"]["tilt_angle"])
        else:
            tilt_angle = self.track_kwargs["tilted_ramp"]["tilt_angle"]
        if isinstance(self.track_kwargs["tilted_ramp"]["switch_spacing"], (tuple, list)):
            if difficulty is not None and self.track_kwargs["tilted_ramp"].get("spacing_curriculum", False):
                switch_spacing = (1-difficulty) * self.track_kwargs["tilted_ramp"]["switch_spacing"][0] + difficulty * self.track_kwargs["tilted_ramp"]["switch_spacing"][1]
            else:
                switch_spacing = np.random.uniform(*self.track_kwargs["tilted_ramp"]["switch_spacing"])
        else:
            switch_spacing = self.track_kwargs["tilted_ramp"]["switch_spacing"]
        if isinstance(self.track_kwargs["tilted_ramp"]["length"], (tuple, list)):
            length = np.random.uniform(*self.track_kwargs["tilted_ramp"]["length"])
        else:
            length = self.track_kwargs["tilted_ramp"]["length"]
        
        tilt_length_px = int(length / self.cfg.horizontal_scale) if length > 0. else self.track_block_resolution[0]
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale)
        depth_px = int(depth / self.cfg.vertical_scale)

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()

        # Define: tilt_left means the right side of the track is higher than the left side
        if switch_spacing > 0.:
            # tilted ramp will switch between left and right
            switch_spacing_px = int(switch_spacing / self.cfg.horizontal_scale)
            overlap_size_px = int(self.track_kwargs["tilted_ramp"]["overlap_size"] / self.cfg.horizontal_scale)
            start_left_px = int(self.track_block_resolution[1] / 2 + overlap_size_px / 2)
            start_right_px = int(self.track_block_resolution[1] / 2 - overlap_size_px / 2)
            tilt_left = True if np.random.uniform() < 0.5 else False
            start_x_px = 1; end_x_px = start_x_px + switch_spacing_px
            while end_x_px < self.track_block_resolution[0] and end_x_px <= tilt_length_px:
                if tilt_left:
                    tilt_left_px = start_left_px - wall_thickness_px
                    track_heightfield[start_x_px: end_x_px, wall_thickness_px: start_left_px] += \
                        np.linspace(tilt_left_px, 0, tilt_left_px) \
                        * np.tan(tilt_angle) \
                        * self.cfg.horizontal_scale / self.cfg.vertical_scale
                    track_heightfield[start_x_px: end_x_px, :wall_thickness_px] -= depth_px
                    track_heightfield[start_x_px: end_x_px, start_left_px:] -= depth_px
                else:
                    tilt_right_px = self.track_block_resolution[1] - start_right_px - wall_thickness_px
                    track_heightfield[start_x_px: end_x_px, start_right_px: -wall_thickness_px] += \
                        np.linspace(0, tilt_right_px, tilt_right_px) \
                        * np.tan(tilt_angle) \
                        * self.cfg.horizontal_scale / self.cfg.vertical_scale
                    track_heightfield[start_x_px: end_x_px, -wall_thickness_px:] -= depth_px
                    track_heightfield[start_x_px: end_x_px, :start_right_px] -= depth_px
                start_x_px = end_x_px; end_x_px = start_x_px + switch_spacing_px
                tilt_left = not tilt_left
        else:
            # two-side tilted ramp
            start_px = int(self.track_block_resolution[1] / 2)
            tilt_left_px = start_px - wall_thickness_px
            tilt_right_px = self.track_block_resolution[1] - start_px - wall_thickness_px
            track_heightfield[1:tilt_length_px, wall_thickness_px: start_px] += \
                np.linspace(tilt_left_px, 0, tilt_left_px) \
                * np.tan(tilt_angle) \
                * self.cfg.horizontal_scale / self.cfg.vertical_scale
            track_heightfield[1:tilt_length_px, start_px: self.track_block_resolution[1]-wall_thickness_px] += \
                np.linspace(0, tilt_right_px, tilt_right_px) \
                * np.tan(tilt_angle) \
                * self.cfg.horizontal_scale / self.cfg.vertical_scale
            track_heightfield[
                1:tilt_length_px,
                wall_thickness_px: -wall_thickness_px,
            ] -= depth_px
            
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        # NOTE: block_info with only 2 scalar values is not enough to describe the tilted_ramp track
        block_info = torch.tensor([
            length,
            tilt_angle,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_slope_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
            terrain_kwargs= None,
        ):
        """ NOTE: No virtual version of slope terrain """
        terrain_kwargs = self.track_kwargs["slope"] if terrain_kwargs is None else terrain_kwargs
        if isinstance(terrain_kwargs["slope_angle"], (tuple, list)):
            if difficulty is None:
                slope_angle = np.random.uniform(*terrain_kwargs["slope_angle"])
            else:
                slope_angle = (1-difficulty) * terrain_kwargs["slope_angle"][0] + difficulty * terrain_kwargs["slope_angle"][1]
        else:
            slope_angle = terrain_kwargs["slope_angle"]
        if isinstance(terrain_kwargs["face_angle"], (tuple, list)):
            if len(terrain_kwargs["face_angle"]) == 2:
                face_angle = np.random.uniform(*terrain_kwargs["face_angle"])
            else: # discrete face angle options
                face_angle = np.random.choice(terrain_kwargs["face_angle"])
        else:
            face_angle = terrain_kwargs["face_angle"]
        if isinstance(terrain_kwargs["length"], (tuple, list)):
            if difficulty is None:
                length = np.random.uniform(*terrain_kwargs["length"])
            else:
                length = (1-difficulty) * terrain_kwargs["length"][0] + difficulty * terrain_kwargs["length"][1]
        else:
            length = terrain_kwargs["length"]

        slope_length_px = int(length / self.cfg.horizontal_scale)
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale)
        
        track_heightfield = heightfield_template.copy()

        # make sure the origin of current block is at height 0.
        slope_x_size_px = slope_length_px - 1        
        slope_y_size_px = self.track_block_resolution[1] - 2 * wall_thickness_px - 1
        x_meshgrid = np.linspace(0, slope_x_size_px, slope_x_size_px)
        y_meshgrid = np.linspace(0, slope_y_size_px, slope_y_size_px) - int(slope_y_size_px / 2)
        x_meshgrid, y_meshgrid = np.meshgrid(x_meshgrid, y_meshgrid, indexing= "ij")
        xy_meshgrid = np.stack([x_meshgrid, y_meshgrid], axis= -1) # (x, y, 2)
        xy_facing_norm = np.sum(xy_meshgrid * np.array([np.cos(face_angle), np.sin(face_angle)]), axis= -1) # (x, y)
        slope_heightfield = xy_facing_norm * np.tan(slope_angle) * self.cfg.horizontal_scale / self.cfg.vertical_scale
        track_heightfield[
            1: slope_length_px,
            wall_thickness_px: -wall_thickness_px-1,
        ] += slope_heightfield
        if terrain_kwargs["use_mean_height_offset"]:
            height_offset_px = np.mean(slope_heightfield[-1, :])
        else:
            height_offset_px = 0.
        track_heightfield[
            slope_length_px:,
            wall_thickness_px: -wall_thickness_px-1,
        ] += height_offset_px
        if not heightfield_noise is None:
            if np.random.uniform() < terrain_kwargs.get("no_perlin_rate", 0.):
                track_heightfield[
                    slope_length_px:,
                    wall_thickness_px: -wall_thickness_px-1,
                ] += heightfield_noise[
                    slope_length_px:,
                    wall_thickness_px: -wall_thickness_px-1,
                ]
            else:
                track_heightfield += heightfield_noise
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        block_info = torch.tensor([
            length,
            slope_angle,
        ], dtype= torch.float32, device= self.device)
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_stairsup_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
            terrain_kwargs= None, # auxilary parameters for similar terrains
        ):
        """ NOTE: No virtual version of stairup terrain """
        assert not virtual, "No virtual version of stairsup terrain"
        terrain_kwargs = self.track_kwargs["stairsup"] if terrain_kwargs is None else terrain_kwargs
        if isinstance(terrain_kwargs["height"], (tuple, list)):
            # the height of each step
            if difficulty is None:
                height = np.random.uniform(*terrain_kwargs["height"])
            else:
                height = (1-difficulty) * terrain_kwargs["height"][0] + difficulty * terrain_kwargs["height"][1]
        else:
            height = terrain_kwargs["height"]
        if isinstance(terrain_kwargs["length"], (tuple, list)):
            # the length of the stairs
            if difficulty is None or (not terrain_kwargs.get("length_curriculum", False)):
                length = np.random.uniform(*terrain_kwargs["length"])
            else:
                length = (1-difficulty) * terrain_kwargs["length"][1] + difficulty * terrain_kwargs["length"][0]
        else:
            length = terrain_kwargs["length"]
        step_length_px = int(length / self.cfg.horizontal_scale)
        n_steps = np.floor(self.track_block_resolution[0] / step_length_px)
        if terrain_kwargs.get("num_steps", None) is not None:
            # the number of steps
            if isinstance(terrain_kwargs["num_steps"], (tuple, list)):
                terrain_kwargs["num_steps"] = [ # clip to prevent out of index error
                    min(terrain_kwargs["num_steps"][0], n_steps),
                    min(terrain_kwargs["num_steps"][1], n_steps),
                ]
                if difficulty is None:
                    n_steps = np.random.randint(*terrain_kwargs["num_steps"])
                else:
                    n_steps = (1-difficulty) * terrain_kwargs["num_steps"][0] + difficulty * terrain_kwargs["num_steps"][1]
                    n_steps = int(n_steps)
            else:
                n_steps = min(terrain_kwargs["num_steps"], n_steps)
        height_px = int(height / self.cfg.vertical_scale)
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale)

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        for i in range(int(n_steps)):
            track_heightfield[
                step_length_px * i + 1: step_length_px * (i + 1) + 1,
                max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
            ] += height_px * (i + 1)
        # make sure the origin of current block is at normal height.
        track_heightfield[
            step_length_px * int(n_steps) + 1:,
            max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
        ] += height_px * int(n_steps)
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        block_info = torch.tensor([
            length,
            height,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = n_steps * height_px
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_stairsdown_track(self,
            *args,
            **kwargs,
        ):
        terrain_kwargs = self.track_kwargs["stairsdown"].copy()
        if isinstance(terrain_kwargs["height"], (tuple, list)):
            terrain_kwargs["height"] = [-terrain_kwargs["height"][0], -terrain_kwargs["height"][1]]
        else:
            terrain_kwargs["height"] = -terrain_kwargs["height"]
        return self.get_stairsup_track(*args, terrain_kwargs= terrain_kwargs, **kwargs)
    
    def get_discrete_rect_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
            terrain_kwargs= None, # auxilary parameters for similar terrains
        ):
        """ generating randomized discrete rectuangular blocks in this terrain.
        similar logic in terrain_utils.discrete_obstacles_terrain
        """
        assert not virtual, "No virtual version of discrete_rect terrain"
        terrain_kwargs = self.track_kwargs["discrete_rect"] if terrain_kwargs is None else terrain_kwargs
        if isinstance(terrain_kwargs["max_height"], (tuple, list)):
            if difficulty is None:
                max_height = np.random.uniform(*terrain_kwargs["max_height"])
            else:
                max_height = (1-difficulty) * terrain_kwargs["max_height"][0] + difficulty * terrain_kwargs["max_height"][1]
        else:
            max_height = terrain_kwargs["max_height"]
        num_rects = terrain_kwargs.get("num_rects", 4)
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale)

        track_heightfield = heightfield_template.copy()
        for _ in range(num_rects):
            width_px = int(np.random.uniform(terrain_kwargs["min_size"], terrain_kwargs["max_size"]) / self.cfg.horizontal_scale)
            width_px = min(width_px, self.track_block_resolution[1] - 2 * wall_thickness_px)
            length_px = int(np.random.uniform(terrain_kwargs["min_size"], terrain_kwargs["max_size"]) / self.cfg.horizontal_scale)
            length_px = min(length_px, self.track_block_resolution[0] - 1)
            start_x_px = np.random.randint(
                wall_thickness_px,
                self.track_block_resolution[0] - length_px - wall_thickness_px,
            )
            start_y_px = np.random.randint(
                1,
                self.track_block_resolution[1] - width_px - 1,
            )
            height_px = int(np.random.uniform(-max_height, max_height) / self.cfg.vertical_scale)
            track_heightfield[
                start_x_px: start_x_px + length_px,
                start_y_px: start_y_px + width_px,
            ] = height_px
        if not heightfield_noise is None:
            track_heightfield += heightfield_noise
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        # NOTE: block info cannot fully describe the terrain
        block_info = torch.tensor([
            0, # should representing obstacle length
            max_height,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_slopeup_track(self,
            *args,
            **kwargs,
        ):
        terrain_kwargs = self.track_kwargs["slopeup"].copy()
        return self.get_slope_track(*args, terrain_kwargs= terrain_kwargs, **kwargs)
    
    def get_slopedown_track(self,
            *args,
            **kwargs,
        ):
        terrain_kwargs = self.track_kwargs["slopedown"].copy()
        if isinstance(terrain_kwargs["slope_angle"], (tuple, list)):
            terrain_kwargs["slope_angle"] = [-terrain_kwargs["slope_angle"][0], -terrain_kwargs["slope_angle"][1]]
        else:
            terrain_kwargs["slope_angle"] = -terrain_kwargs["slope_angle"]
        return self.get_slope_track(*args, terrain_kwargs= terrain_kwargs, **kwargs)
    
    def get_wave_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty=None,
            heightfield_noise=None,
            virtual=False,
            terrain_kwargs=None,
        ):
        """ NOTE: No virtual version of wave terrain """
        assert not virtual, "No virtual version of wave terrain"
        terrain_kwargs = self.track_kwargs["wave"] if terrain_kwargs is None else terrain_kwargs
        if isinstance(terrain_kwargs["amplitude"], (tuple, list)):
            if difficulty is None:
                amplitude = np.random.uniform(*terrain_kwargs["amplitude"])
            else:
                amplitude = (1-difficulty) * terrain_kwargs["amplitude"][0] + difficulty * terrain_kwargs["amplitude"][1]
        else:
            amplitude = terrain_kwargs["amplitude"]
        if isinstance(terrain_kwargs["frequency"], (tuple, list)):
            if difficulty is None:
                frequency = np.random.uniform(*terrain_kwargs["frequency"])
            else:
                frequency = (1-difficulty) * terrain_kwargs["frequency"][0] + difficulty * terrain_kwargs["frequency"][1]
        else:
            frequency = terrain_kwargs["frequency"]
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        wave_length_px = self.track_block_resolution[0] - 2
        wave_width_px = self.track_block_resolution[1] - 2 * wall_thickness_px

        track_heightfield = heightfield_template.copy()
        div = wave_length_px / (frequency * np.pi * 2)
        x = np.linspace(0, wave_length_px, wave_length_px)
        x += np.random.uniform(-int(wave_length_px / 2), int(wave_length_px / 2))
        y = np.linspace(0, wave_width_px, wave_width_px)
        y += np.random.uniform(-int(wave_width_px / 2), int(wave_width_px / 2))
        xx, yy = np.meshgrid(x, y, indexing= "ij")
        wave = amplitude * (np.sin(xx / div) + np.sin(yy / div))
        track_heightfield[
            1: wave_length_px+1,
            wall_thickness_px: -wall_thickness_px,
        ] += wave / self.cfg.vertical_scale

        if not heightfield_noise is None:
            track_heightfield += heightfield_noise
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        block_info = torch.tensor([
            0, # should representing obstacle length
            amplitude,
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
        print("heightfield_raw data shape:", map_x_size, map_y_size, "border size:", self.border)
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
        if self.track_kwargs.get("n_obstacles_per_track", None) == 1 and len(self.track_kwargs["options"]) > 0:
            obstacle_order = np.array([col_idx % len(self.track_kwargs["options"])])
            # NOTE: record the terrain type name for each column for later use and preventing
            # repeating code segment in other methods.
            if not hasattr(self, "track_terrain_type_names"):
                self.track_terrain_type_names = [None for _ in range(self.cfg.num_cols)]
            self.track_terrain_type_names[col_idx] = self.track_kwargs["options"][obstacle_order[0]]
        elif self.track_kwargs["randomize_obstacle_order"] and len(self.track_kwargs["options"]) > 0:
            obstacle_order = np.random.choice(
                len(self.track_kwargs["options"]),
                size= self.track_kwargs.get("n_obstacles_per_track", len(self.track_kwargs["options"])),
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
        self.heightfield_raw[
            track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0],
            track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1],
        ] += block_starting_height_px
        block_starting_height_px += height_offset_px
        
        for obstacle_idx, obstacle_selection in enumerate(obstacle_order):
            obstacle_name = self.track_kwargs["options"][obstacle_selection]
            obstacle_id = self.track_options_id_dict[obstacle_name]
            # call method to generate trimesh and heightfield for each track block.
            # For example get_jump_track, get_tilt_track
            # using `virtual_track` to create non-collision mesh for collocation method in training.
            # NOTE: The heightfield is not used for building mesh in simulation, just representing the terrain
            # data relative to the block_starting_height_px in height values.
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
            ] = track_heightfield + block_starting_height_px
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
    
    def add_column_border_to_sim(self, starting_height_px, row_idx= None, col_idx= None):
        """ Add additional terrrain to the end of each column, so that the robot will not
        see a void space and see it as a detreme danger.
        """
        assert not col_idx is None, "col_idx should be provided."

        x_start_px = int((row_idx + 1) * self.track_resolution[0]) + self.border
        y_start_px = int(col_idx * self.track_resolution[1]) + self.border
        y_end_px = int((col_idx + 1) * self.track_resolution[1]) + self.border

        origin = np.array([
            x_start_px * self.cfg.horizontal_scale,
            y_start_px * self.cfg.horizontal_scale,
            starting_height_px * self.cfg.vertical_scale,
        ])

        plane_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(
                self.heightfield_raw[
                    x_start_px:,
                    y_start_px: y_end_px,
                ]
            ),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        self.heightfield_raw[
            x_start_px:,
            y_start_px: y_end_px,
        ] += starting_height_px
        self.add_trimesh_to_sim(plane_trimesh, origin)

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ Add current terrain as trimesh to sim and update the corresponding heightfield """
        self.gym = gym
        self.sim = sim
        self.device = device
        self.initialize_track()
        self.build_heightfield_raw()
        self.initialize_track_info_buffer()

        # track_origins_px stores the min x, min y, min z of each track block in pixel
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
            self.add_column_border_to_sim(
                starting_height_px,
                row_idx= row_idx,
                col_idx= col_idx,
            )
        
        self.add_plane_to_sim(starting_height_px)
        
        # env_origins stores the min x, mid y, start z of each track block in meter
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                self.env_origins[i, j, 0] = self.track_origins_px[i, j, 0] * self.cfg.horizontal_scale
                self.env_origins[i, j, 1] = self.track_origins_px[i, j, 1] * self.cfg.horizontal_scale
                self.env_origins[i, j, 2] = self.track_origins_px[i, j, 2] * self.cfg.vertical_scale
                self.env_origins[i, j, 1] += self.track_kwargs["track_width"] / 2
        self.env_origins_pyt = torch.from_numpy(self.env_origins).to(self.device)
        self.heightfield_raw_pyt = torch.tensor(
            self.heightfield_raw,
            dtype= torch.float32,
            device= self.device,
        )

    def add_plane_to_sim(self, final_height_px= 0.):
        """
        Args:
            final_height_px: the height of the region 3 in the following figure.
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
                 |         |              The track grid               | The      |
                 |         |                                           | column   |
                y|         |                                           | border   |
                ^|         |                                           |          |
                ||         |                                           |          |
                ||region1  |                                           |          |
                |+---------+-------------------------------------------+----------+
                ||                                                                |
                ||                                                                |
                ||region0                                                         |
                |+----------------------------------------------------------------+
                +--------------->x
            """
            # Column border will be added by the `add_column_border_to_sim` method
            trimesh_origins = [
                [0, 0, 0],
                [0, self.cfg.border_size, 0],
                [0, self.cfg.border_size + self.cfg.num_cols * self.env_width, 0],
                # [self.cfg.border_size + self.cfg.num_rows * self.env_length, self.cfg.border_size, final_height_px * self.cfg.vertical_scale],
            ]
            heightfield_regions = [
                [slice(0, self.heightfield_raw.shape[0]), slice(0, self.border)],
                [slice(0, self.border), slice(self.border, self.heightfield_raw.shape[1] - self.border)],
                [
                    slice(0, self.heightfield_raw.shape[0]),
                    slice(self.heightfield_raw.shape[1] - self.border, self.heightfield_raw.shape[1]),
                ],
                # [
                #     slice(self.heightfield_raw.shape[0] - self.border, self.heightfield_raw.shape[0]),
                #     slice(self.border, self.border + self.track_resolution[1] * self.cfg.num_cols),
                # ],
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
                self.heightfield_raw[
                    heightfield_region[0],
                    heightfield_region[1],
                ] += origin[2] / self.cfg.vertical_scale
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

    def get_engaging_block_idx(self, base_positions, min_x_points_offset= 0):
        """ Different from the block idx where the robot is currently in.
        This method computes the block idx where the robot is engaging with (maybe the next block)
        """
        in_track_mask = self.in_terrain_range(base_positions)
        track_idx_clipped = self.get_track_idx(base_positions)
        forward_distance, block_idx_clipped = self.get_in_track_positions(base_positions)
        in_block_distance = forward_distance % self.env_block_length
        # compute whether the robot is engaging with the next block
        curr_obstacle_depth = self.track_info_map[
            track_idx_clipped[:, 0],
            track_idx_clipped[:, 1],
            block_idx_clipped,
            1,
        ] # (n,)
        engaging_next_block = ((in_block_distance + min_x_points_offset) > curr_obstacle_depth) \
            & (in_block_distance > self.track_kwargs.get("engaging_finish_threshold", 0.)) \
            & (in_block_distance > self.engaging_next_min_forward_distance)

        # update the engaging track_idx and block_idx if engaging next
        engaging_next_track = (block_idx_clipped == (self.n_blocks_per_track - 1)) & engaging_next_block
        track_idx_selection = track_idx_clipped.detach().clone()
        block_idx_selection = block_idx_clipped.detach().clone()
        last_track_last_block_mask = (block_idx_clipped == (self.n_blocks_per_track - 1)) & (track_idx_clipped[:, 0] == (self.cfg.num_rows - 1))
        track_idx_selection[engaging_next_track & (~last_track_last_block_mask), 0] += 1
        assert track_idx_selection[:, 0].max() < self.track_info_map.shape[0], track_idx_selection[:, 0].max()
        block_idx_selection[engaging_next_block & (~last_track_last_block_mask)] += 1
        block_idx_selection[engaging_next_track & (~last_track_last_block_mask) | (~in_track_mask)] = 0
        assert block_idx_selection.max() <= (self.n_blocks_per_track - 1), block_idx_selection.max()
        
        return (
            engaging_next_block, 
            track_idx_selection, 
            block_idx_selection,
        )
    
    def get_engaging_block_types(self, base_positions, volume_points_offset= None):
        """ return the obstacle_id of the engaging block, refer to terrain.track_options_id_dict for
        the obstacle name.
        """
        if volume_points_offset is None:
            min_x_points_offset = 0.
        else:
            min_x_points_offset = torch.min(volume_points_offset[:, :, 0], dim= -1)[0]
        
        _, track_idx_selection, block_idx_selection = self.get_engaging_block_idx(base_positions, min_x_points_offset)
        engaging_block_types = self.track_info_map[
            track_idx_selection[:, 0],
            track_idx_selection[:, 1],
            block_idx_selection,
            0,
        ].to(int) # (n,)
        return engaging_block_types
    
    def get_engaging_block_distance(self, base_positions, volume_points_offset= None):
        """ Get the closet forward obstacle distance in x-axis """
        if volume_points_offset is None:
            min_x_points_offset = 0.
        else:
            min_x_points_offset = torch.min(volume_points_offset[:, :, 0], dim= -1)[0]

        _, track_idx_selection, block_idx_selection = self.get_engaging_block_idx(base_positions, min_x_points_offset)
        engaging_distance = track_idx_selection[:, 0] * self.env_length + block_idx_selection * self.env_block_length + self.cfg.border_size
        engaging_distance -= base_positions[:, 0]
        return engaging_distance

    def get_engaging_block_info(self, base_positions, volume_points_offset= None):
        """ Get the closet forward obstacle info """
        if volume_points_offset is None:
            min_x_points_offset = 0.
        else:
            min_x_points_offset = torch.min(volume_points_offset[:, :, 0], dim= -1)[0] # (n,)

        _, track_idx_selection, block_idx_selection = self.get_engaging_block_idx(base_positions, min_x_points_offset)
        engaging_block_info = self.track_info_map[
            track_idx_selection[:, 0],
            track_idx_selection[:, 1],
            block_idx_selection,
            1:
        ]
        return engaging_block_info # (n, obstacle_info_dim)

    def get_in_track_positions(self, base_positions):
        track_idx = self.get_track_idx(base_positions, clipped= False)
        forward_distance = base_positions[:, 0] - self.cfg.border_size - (track_idx[:, 0] * self.env_length)
        block_idx = torch.floor(forward_distance / self.env_block_length).to(int) # (n,)
        block_idx_clipped = torch.clip(
            block_idx,
            0.,
            (self.n_blocks_per_track - 1),
        )
        
        return forward_distance, block_idx_clipped

    @staticmethod # deprecated, check self.block_info_dim
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
        if (block_infos[:, 2] < 0.).any():
            print("Warning: jump down is deprecated, use down instead.")

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
    
    def get_hurdle_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return self.get_jump_penetration_depths(
            block_infos,
            positions_in_block,
            mask_only= mask_only,
        )
    
    def get_down_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        in_down_mask = positions_in_block[:, 0] <= block_infos[:, 1]
        down_over_mask = positions_in_block[:, 2] > block_infos[:, 2] # (n_points,)

        penetrated_mask = torch.logical_and(
            in_down_mask,
            (torch.logical_not(down_over_mask)),
        )
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetration_depths_buffer[penetrated_mask] = 0. - positions_in_block[penetrated_mask, 2]
        return penetration_depths_buffer
    
    def get_tilted_ramp_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return torch.zeros_like(positions_in_block[:, 0])

    def get_slope_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return torch.zeros_like(positions_in_block[:, 0])
    
    def get_stairsup_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        stairs_lengths = block_infos[:, 1]
        stairs_heights = block_infos[:, 2]
        nearest_stair_edge_x = torch.round(positions_in_block[:, 0] / stairs_lengths) * stairs_lengths
        nearest_stair_edge_x[nearest_stair_edge_x >= self.env_block_length] -= \
            stairs_lengths[nearest_stair_edge_x >= self.env_block_length]
        nearest_stair_edge_z = torch.round(nearest_stair_edge_x / stairs_lengths) * stairs_heights
        distance_to_edge = torch.norm(
            torch.cat([positions_in_block[:, 0], positions_in_block[:, 2]], dim= -1) - 
            torch.cat([nearest_stair_edge_x, nearest_stair_edge_z], dim= -1),
            dim= -1,
        )
        if mask_only:
            return distance_to_edge < self.track_kwargs["stairsup"].get("residual_distance", 0.05)
        else:
            return torch.clip(
                self.track_kwargs["stairsup"].get("residual_distance", 0.05) - distance_to_edge,
                min= 0.,
            )
        
    def get_stairsdown_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return torch.zeros_like(positions_in_block[:, 0])
    
    def get_discrete_rect_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return torch.zeros_like(positions_in_block[:, 0])
    
    def get_slopeup_penetration_depths(self,
            *args,
            **kwargs,
        ):
        return self.get_slope_penetration_depths(*args, **kwargs)

    def get_slopedown_penetration_depths(self,
            *args,
            **kwargs,
        ):
        return self.get_slope_penetration_depths(*args, **kwargs)
    
    def get_wave_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        return torch.zeros_like(positions_in_block[:, 0])

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
            distances -= self.env_length
            if (rows >= (self.track_info_map.shape[0])).all() or (distances <= 0).all():
                break
            rows = torch.clip(rows, min= 0)
            rows = torch.clip(rows, max= self.track_info_map.shape[0] - 1)

        return passed_depths
    
    def get_goal_position(self, robot_positions):
        """ Get the goal position of each engaging block, assuming the robot is on the `robot_positions`
        robot_positions: shape (num_envs, 3)
        """
        in_terrain_mask = self.in_terrain_range(robot_positions)
        _, engaging_track_idx, engaging_block_idx = self.get_engaging_block_idx(robot_positions)
        last_track_last_block_mask = (engaging_block_idx == (self.n_blocks_per_track - 1)) & (engaging_track_idx[:, 0] == (self.cfg.num_rows - 1))
        goal_position_x = self.env_origins_pyt[engaging_track_idx[:, 0], engaging_track_idx[:, 1], 0] \
            + engaging_block_idx * self.env_block_length
        goal_position_x[~last_track_last_block_mask] += (self.engaging_next_min_forward_distance + self.env_block_length) / 2
        goal_position_x[last_track_last_block_mask] += self.env_block_length
            # Thus, the engaging goal will refreshed to next when the engaging goal is reached.
        goal_position_y = self.env_origins_pyt[engaging_track_idx[:, 0], engaging_track_idx[:, 1], 1]
        goal_position_z = robot_positions[:, 2]
        goal_position = torch.cat([
            goal_position_x.unsqueeze(-1),
            goal_position_y.unsqueeze(-1),
            goal_position_z.unsqueeze(-1),
        ], dim= -1) # (num_envs, 3)
        # if the robot is not in the terrain, the goal position is the robot position (Thus stopping the robot moving)
        goal_position[torch.logical_not(in_terrain_mask)] = robot_positions[torch.logical_not(in_terrain_mask)]

        return goal_position

    def in_terrain_range(self, pos):
        """ Check whether the position is in the terrain range """
        track_idx = self.get_track_idx(pos, clipped= False)
        # The robot is going +x direction, so no checking for row_idx <= 0
        in_track_mask = track_idx[:, 0] >= 0
        in_track_mask &= track_idx[:, 0] < self.cfg.num_rows
        in_track_mask &= track_idx[:, 1] >= 0
        in_track_mask &= track_idx[:, 1] < self.cfg.num_cols
        return in_track_mask
    
    @torch.no_grad()
    def get_terrain_heights(self, points):
        """ Get the terrain heights below the given points """
        points_shape = points.shape
        points = points.view(-1, 3)
        points_x_px = (points[:, 0] / self.cfg.horizontal_scale).to(int)
        points_y_px = (points[:, 1] / self.cfg.horizontal_scale).to(int)
        out_of_range_mask = torch.logical_or(
            torch.logical_or(points_x_px < 0, points_x_px >= self.heightfield_raw_pyt.shape[0]),
            torch.logical_or(points_y_px < 0, points_y_px >= self.heightfield_raw_pyt.shape[1]),
        )
        points_x_px = torch.clip(points_x_px, 0, self.heightfield_raw_pyt.shape[0] - 1)
        points_y_px = torch.clip(points_y_px, 0, self.heightfield_raw_pyt.shape[1] - 1)
        heights = self.heightfield_raw_pyt[points_x_px, points_y_px] * self.cfg.vertical_scale
        heights[out_of_range_mask] = - torch.inf
        heights = heights.view(points_shape[:-1])
        return heights
    
    @property
    def available_terrain_type_names(self):
        """ Get a list lf terrain type names currently used in the terrain. """
        return self.track_kwargs["options"]
    
    def get_terrain_type_names(self, terrain_types):
        """ Return the terrain type name given the int array (terrain_types).
        Return: list of str (None when not single type on each column)
        """
        if hasattr(self, "track_terrain_type_names"):
            return [self.track_terrain_type_names[i] for i in terrain_types]            

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
        wall_height = np.random.uniform(*self.track_kwargs["tilt"]["wall_height"]) if isinstance(self.track_kwargs["tilt"]["wall_height"], (tuple, list)) else self.track_kwargs["tilt"]["wall_height"]
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
        
    def draw_virtual_hurdle_track(self,
            block_info,
            block_origin,
        ):
        return self.draw_virtual_jump_track(
            block_info,
            block_origin,
        )
    
    def draw_virtual_down_track(self,
            block_info,
            block_origin,
        ):
        down_depth = block_info[1]
        down_height = block_info[2]
        geom = gymutil.WireframeBoxGeometry(
            down_depth,
            self.env_width,
            down_height,
            pose= None,
            color= (0, 0, 1),
        )
        pose = gymapi.Transform(gymapi.Vec3(
            down_depth/2 + block_origin[0],
            block_origin[1],
            down_height/2 + block_origin[2],
        ), r= None)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            None,
            pose,
        )

    def draw_virtual_tilted_ramp_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for tilted ramp
        pass

    def draw_virtual_slope_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for slope
        pass

    def draw_virtual_stairsup_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for stairsup
        pass

    def draw_virtual_stairsdown_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for stairsdown
        pass

    def draw_virtual_discrete_rect_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for discrete_rect
        pass

    def draw_virtual_slopeup_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for slopeup
        pass

    def draw_virtual_slopedown_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for slopedown
        pass

    def draw_virtual_wave_track(self,
            block_info,
            block_origin,
        ):
        # No virtual terrain for wave
        pass

    def draw_virtual_track(self,
            row_idx,
            col_idx,
        ):
        difficulties = self.get_difficulty(row_idx, col_idx)
        virtual_terrain = difficulties[1]
        track_origin = self.env_origins[row_idx, col_idx]

        for block_idx in range(1, self.track_info_map.shape[2]):
            if self.track_kwargs["draw_virtual_terrain"]:
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
