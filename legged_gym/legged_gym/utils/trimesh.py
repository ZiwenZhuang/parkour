""" This file defines a mesh as a tuple of (vertices, triangles)
All operations are based on numpy ndarray
- vertices: np ndarray of shape (n, 3) np.float32
- triangles: np ndarray of shape (n_, 3) np.uint32
"""
import numpy as np

def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]

    return vertices, triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(trimeshes[1:])
        )

    # only two trimesh to combine
    trimesh_0, trimesh_1 = trimeshes
    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    """ inplace operation """
    trimesh[0] += move
