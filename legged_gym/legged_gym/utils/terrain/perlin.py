import numpy as np
import torch
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils, gymapi

import matplotlib.pyplot as plt

class TerrainPerlin:
    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.xSize = cfg.terrain_length * cfg.num_rows # int(cfg.horizontal_scale * cfg.tot_cols)
        self.ySize = cfg.terrain_width * cfg.num_cols # int(cfg.horizontal_scale * cfg.tot_rows)
        self.tot_rows = int(self.xSize / cfg.horizontal_scale)
        self.tot_cols = int(self.ySize / cfg.horizontal_scale)
        assert(self.xSize == cfg.horizontal_scale * self.tot_rows and self.ySize == cfg.horizontal_scale * self.tot_cols)
        self.heightsamples_float = self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_rows, self.tot_cols, **cfg.TerrainPerlin_kwargs)
        # self.heightsamples_float[self.tot_cols//2 - 100:, :] += 100000
        # self.heightsamples_float[self.tot_cols//2 - 40: self.tot_cols//2 + 40, :] = np.mean(self.heightsamples_float)
        self.heightsamples = (self.heightsamples_float * (1 / cfg.vertical_scale)).astype(np.int16)
        self.heightfield_raw_pyt = torch.tensor(self.heightsamples, device= "cpu")
        

        print("Terrain heightsamples shape: ", self.heightsamples.shape)
        print("Terrain heightsamples stat: ", *(np.array([np.min(self.heightsamples), np.max(self.heightsamples), np.mean(self.heightsamples), np.std(self.heightsamples), np.median(self.heightsamples)]) * cfg.vertical_scale))
        # self.heightsamples = np.zeros((800, 800)).astype(np.int16)
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples,
                                                                                        cfg.horizontal_scale,
                                                                                        cfg.vertical_scale,
                                                                                        cfg.slope_treshold)
    
    @staticmethod
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
    
    @staticmethod
    def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
        frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
        xScale = int(frequency * xSize)
        yScale = int(frequency * ySize)
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * TerrainPerlin.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise

    def add_trimesh_to_sim(self, trimesh, trimesh_origin):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = 0.
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ deploy the terrain mesh to the simulator
        """
        self.gym = gym
        self.sim = sim
        self.device = device
        self.add_trimesh_to_sim(
            (self.vertices, self.triangles),
            np.zeros(3,),
        )
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        for row_idx in range(self.cfg.num_rows):
            for col_idx in range(self.cfg.num_cols):
                origin_x = (row_idx + 0.5) * self.env_length
                origin_y = (col_idx + 0.5) * self.env_width
                self.env_origins[row_idx, col_idx] = [
                    origin_x,
                    origin_y,
                    self.heightsamples[
                        int(origin_x / self.cfg.horizontal_scale),
                        int(origin_y / self.cfg.horizontal_scale),
                    ] * self.cfg.vertical_scale,
                ]
        self.heightfield_raw_pyt = torch.from_numpy(self.heightsamples).to(device= self.device).float()

    def in_terrain_range(self, pos):
        """ Check if the given position still have terrain underneath. (same x/y, but z is different)
        pos: (batch_size, 3) torch.Tensor
        """
        return torch.logical_and(
            pos[..., :2] >= 0,
            pos[..., :2] < torch.tensor([self.xSize, self.ySize], device= self.device),
        ).all(dim= -1)
    
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
