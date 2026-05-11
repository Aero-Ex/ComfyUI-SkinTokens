from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numpy import ndarray
from scipy.spatial import cKDTree # type: ignore
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from typing import Dict, List, Optional, Literal

import numpy as np

from ..rig_package.info.asset import Asset

@dataclass(frozen=True)
class VertexGroup(ABC):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> 'VertexGroup':
        pass
    
    @abstractmethod
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        pass
    
@dataclass(frozen=True)
class VertexGroupSkin(VertexGroup):
    """capture skin"""
    
    normalize: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'VertexGroupSkin':
        return VertexGroupSkin(normalize=kwargs.get('normalize', True))
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        if asset.skin is None:
            raise ValueError("do not have skin")
        if self.normalize:
            asset.normalize_skin()
        return {'skin': asset.skin.copy()}

@dataclass(frozen=True)
class VertexGroupVoxelSkin(VertexGroup):
    """capture voxel skin"""
    
    grid: int
    alpha: float
    link_dis: float
    grid_query: int
    vertex_query: int
    grid_weight: float
    mode: Literal['square', 'exp']
    
    @classmethod
    def parse(cls, **kwargs) -> 'VertexGroupVoxelSkin':
        return VertexGroupVoxelSkin(
            grid=kwargs.get('grid', 64),
            alpha=kwargs.get('alpha', 0.5),
            link_dis=kwargs.get('link_dis', 0.00001),
            grid_query=kwargs.get('grid_query', 27),
            vertex_query=kwargs.get('vertex_query', 27),
            grid_weight=kwargs.get('grid_weight', 3.0),
            mode=kwargs.get('mode', 'square'),
        )
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        if asset.vertices is None:
            raise ValueError("do not have vertices")
        if asset.faces is None:
            raise ValueError("do not have faces")
        if asset.joints is None:
            raise ValueError("do not have joints")
        # normalize into [-1, 1] first
        min_vals = np.min(asset.vertices, axis=0)
        max_vals = np.max(asset.vertices, axis=0)
        
        center = (min_vals + max_vals) / 2
        
        scale = np.max(max_vals - min_vals) / 2
        
        normalized_vertices = (asset.vertices - center) / scale
        normalized_joints = (asset.joints - center) / scale
        
        grid_coords = asset.voxel().coords
        skin = voxel_skin(
            grid=self.grid,
            grid_coords=grid_coords,
            joints=normalized_joints,
            vertices=normalized_vertices,
            faces=asset.faces,
            alpha=self.alpha,
            link_dis=self.link_dis,
            grid_query=self.grid_query,
            vertex_query=self.vertex_query,
            grid_weight=self.grid_weight,
            mode=self.mode,
        )
        skin = np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)
        return {'voxel_skin': skin,}

def voxel_skin(
    grid: int,
    grid_coords: ndarray, # (M, 3)
    joints: ndarray, # (J, 3)
    vertices: ndarray, # (N, 3)
    faces: ndarray, # (F, 3)
    alpha: float=0.5,
    link_dis: float=0.00001,
    grid_query: int=27,
    vertex_query: int=27,
    grid_weight: float=3.0,
    voxel_size: Optional[float]=None,
    mode: str='square',
    parents: Optional[ndarray]=None,
):  
    # modified from https://dl.acm.org/doi/pdf/10.1145/2485895.2485919
    assert mode in ['square', 'exp']
    J = joints.shape[0]
    M = grid_coords.shape[0]
    N = vertices.shape[0]
    
    if voxel_size is None:
        _range = 2/grid*1.74
    else:
        _range = voxel_size*1.74
    
    grid_tree = cKDTree(grid_coords)
    vertex_tree = cKDTree(vertices)
    if parents is not None:
        son = defaultdict(list)
        for i, p in enumerate(parents):
            if i == -1:
                continue
            son[p].append(i)
        divide_joints = []
        joints_map = []
        for u in range(len(parents)):
            if len(son[u]) != 1:
                divide_joints.append(joints[u])
                joints_map.append(u)
            else:
                pu = joints[u]
                pv = joints[son[u][0]]
                seg = 10
                for i in range(seg+1):
                    p = (pu*i + pv*(seg-i)) / seg
                    divide_joints.append(p)
                    joints_map.append(u)
        divide_joints = np.stack(divide_joints)
        joints_map = np.array(joints_map)
    else:
        divide_joints = joints
        joints_map = np.arange(joints.shape[0])
    joint_tree = cKDTree(divide_joints)
    
    # make combined vertices
    # 0   ~ N-1: mesh vertices
    # N   ~ N+M-1: grid vertices
    combined_vertices = np.concatenate([vertices, grid_coords], axis=0)
    
    # link adjacent grids
    dist, idx = grid_tree.query(grid_coords, grid_query) # 3*3*3
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < _range)
    source_grid2grid = np.repeat(np.arange(M), grid_query-1)[mask.ravel()] + N
    to_grid2grid = idx[mask] + N
    weight_grid2grid = dist[mask] * grid_weight
    
    # link very close vertices
    dist, idx = vertex_tree.query(vertices, 4)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < link_dis)
    source_close = np.repeat(np.arange(N), 3)[mask.ravel()]
    to_close = idx[mask]
    weight_close = dist[mask]
    
    # link grids to mesh vertices
    dist, idx = vertex_tree.query(grid_coords, vertex_query)
    mask = (0 < dist) & (dist < _range) # sqrt(3)
    source_grid2vertex = np.repeat(np.arange(M), vertex_query)[mask.ravel()] + N
    to_grid2vertex = idx[mask]
    weight_grid2vertex = dist[mask]
    
    # build combined vertices tree
    combined_tree = cKDTree(combined_vertices)
    # link bones to the neartest vertices
    _, joint_indices = combined_tree.query(divide_joints)
    
    # build graph
    source_vertex2vertex = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    to_vertex2vertex = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    weight_vertex2vertex = np.sqrt(((vertices[source_vertex2vertex] - vertices[to_vertex2vertex])**2).sum(axis=-1))
    graph = csr_matrix(
        (np.concatenate([weight_close, weight_vertex2vertex, weight_grid2grid, weight_grid2vertex]),
        (
            np.concatenate([source_close, source_vertex2vertex, source_grid2grid, source_grid2vertex], axis=0),
            np.concatenate([to_close, to_vertex2vertex, to_grid2grid, to_grid2vertex], axis=0)),
        ),
        shape=(N+M, N+M),
    )
    
    # get shortest path (J, N+M)
    dist_matrix = shortest_path(graph, method='D', directed=False, indices=joint_indices)
    
    # (sum_J, N)
    dis_vertex2bone = dist_matrix[:, :N]
    unreachable = np.isinf(dis_vertex2bone).all(axis=0)
    k = min(J, 3)
    dist, idx = joint_tree.query(vertices[unreachable], k)
    
    # make sure at least one value in dis is not inf
    unreachable_indices = np.where(unreachable)[0]
    row_indices = idx
    col_indices = np.repeat(unreachable_indices, k).reshape(-1, k)
    dis_vertex2bone[row_indices, col_indices] = dist
    
    finite_vals = dis_vertex2bone[np.isfinite(dis_vertex2bone)]
    max_dis = np.max(finite_vals)
    dis_vertex2bone = np.nan_to_num(dis_vertex2bone, nan=max_dis, posinf=max_dis, neginf=max_dis)
    dis_vertex2bone = np.maximum(dis_vertex2bone, 1e-6)
    
    # turn dis2bone to dis2vertex
    dis_vertex2joint = np.full((joints.shape[0], vertices.shape[0]), max_dis)
    for i in range(len(dis_vertex2bone)):
        dis_vertex2joint[joints_map[i]] = np.minimum(dis_vertex2bone[i], dis_vertex2joint[joints_map[i]])
    
    # (J, N)
    if mode == 'exp':
        skin = np.exp(-dis_vertex2joint / max_dis * 20.0)
    elif mode == 'square':
        skin = (1./((1-alpha)*dis_vertex2joint + alpha*dis_vertex2joint**2))**2
    else:
        assert False, f'invalid mode: {mode}'
    skin = skin / skin.sum(axis=0)
    # (N, J)
    skin = skin.transpose()
    return skin

def get_vertex_groups(*args) -> List[VertexGroup]:
    vertex_groups = []
    MAP = {
        'skin': VertexGroupSkin,
        'voxel_skin': VertexGroupVoxelSkin,
    }
    MAP: Dict[str, type[VertexGroup]]
    for (i, c) in enumerate(args):
        __target__ = c.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in config of vertex_groups of position {i}"
        assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
        c = deepcopy(c)
        del c['__target__']
        vertex_groups.append(MAP[__target__].parse(**c))
    return vertex_groups