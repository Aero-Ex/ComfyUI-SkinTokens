from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional, Dict
from numpy import ndarray
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R

import numpy as np
import random

from .spec import ConfigSpec

from ..rig_package.utils import axis_angle_to_matrix
from ..rig_package.info.asset import Asset

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def transform(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentTrim(Augment):
    """randomly delete joints and vertices"""
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentTrim':
        cls.check_keys(kwargs)
        return AugmentTrim()
    
    def transform(self, asset: Asset, **kwargs):
        asset.trim_skeleton()

@dataclass(frozen=True)
class AugmentDelete(Augment):
    """randomly delete joints and vertices"""
    
    # probability
    p: float
    
    # how much to keep
    rate: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentDelete':
        cls.check_keys(kwargs)
        return AugmentDelete(
            p=kwargs.get('p', 0.),
            rate=kwargs.get('rate', 0.5),
        )
    
    def transform(self, asset: Asset, **kwargs):
        if asset.skin is None:
            raise ValueError("do not have skin")
        if asset.parents is None:
            raise ValueError("do not have parents")
        asset.normalize_skin()
        def select_k(arr: List, k: int):
            if len(arr) <= k:
                return arr
            else:
                rest_indices = list(range(1, len(arr)))
                selected_indices = sorted(random.sample(rest_indices, k))
                return [arr[i] for i in selected_indices]
        if np.random.rand() >= self.p:
            return
        ids = select_k([i for i in range(asset.J)], max(int(asset.J * (1 - np.random.rand() * self.rate)), 1))
        if len(ids) == 0:
            return
        # keep bones with no skin
        keep = {}
        for id in ids:
            keep[id] = True
        for id in range(asset.J):
            if np.all(asset.skin[:, id] < 0.1):
                keep[id] = True
        keep[asset.root] = True
        
        vertices_to_delete = np.zeros(asset.N, dtype=bool)
        for id in range(asset.J):
            if id not in keep:
                dominant = asset.skin.argmax(axis=1) == id
                x = (asset.skin[:, id] > 0.1) & dominant
                if np.all(~x) or x.sum() * asset.J < asset.N: # avoid collapsing
                    keep[id] = 1
                    continue
                vertices_to_delete[x] = True
        if np.all(vertices_to_delete):
            return
        if asset.faces is not None:
            indices = np.where(~vertices_to_delete)[0]
            face_mask = np.all(np.isin(asset.faces, indices), axis=1)
            if np.all(~face_mask):
                return
        
        joints_to_delete: List[int|str] = [i for i in range(asset.J) if i not in keep]
        asset.delete_joints(joints_to_delete)
        asset.delete_vertices(np.arange(asset.N)[vertices_to_delete])

@dataclass(frozen=True)
class AugmentDropPart(Augment):
    """randomly drop subtrees and their vertices"""
    
    # probability
    p: float
    
    # drop rate
    rate: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentDropPart':
        cls.check_keys(kwargs)
        return AugmentDropPart(
            p=kwargs.get('p', 0.),
            rate=kwargs.get('rate', 0.5),
        )
    
    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() >= self.p:
            return
        if asset.parents is None:
            raise ValueError("do not have parents")
        if asset.skin is None:
            raise ValueError("do not have skin")
        keep = []
        for id in range(asset.J):
            if np.random.rand() < self.rate:
                keep.append(id)
        if len(keep) == 0:
            return
        for id in reversed(asset.dfs_order):
            p = asset.parents[id]
            if p == -1:
                continue
            if id in keep and p not in keep:
                keep.append(p)
        
        mask = np.zeros(asset.N, dtype=bool)
        for id in keep:
            mask[asset.skin[:, id] > 1e-5] = True
        vertices_to_delete = ~mask
        if np.all(vertices_to_delete):
            return
        if asset.faces is not None:
            indices = np.where(~vertices_to_delete)[0]
            face_mask = np.all(np.isin(asset.faces, indices), axis=1)
            if np.all(~face_mask):
                return
        
        joints_to_delete: List[int|str] = [i for i in range(asset.J) if i not in keep]
        asset.delete_joints(joints_to_delete)
        asset.delete_vertices(np.arange(asset.N)[vertices_to_delete])
    
    def inverse(self, asset: Asset):
        pass

@dataclass(frozen=True)
class AugmentCollapse(Augment):
    """randomly merge joints"""
    
    # collapse the skeleton with probability p
    p: float
    
    # probability to merge the bone
    rate: float
    
    # max bones
    max_bones: int
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentCollapse':
        cls.check_keys(kwargs)
        return AugmentCollapse(
            p=kwargs.get('p', 0.),
            rate=kwargs.get('rate', 0.),
            max_bones=kwargs.get('max_bones', 2147483647),
        )
    
    def transform(self, asset: Asset, **kwargs):
        def select_k(arr: List, k: int):
            if len(arr) <= k:
                return arr
            else:
                rest_indices = list(range(1, len(arr)))
                selected_indices = sorted(random.sample(rest_indices, k))
                return [arr[i] for i in selected_indices]
        
        root = asset.root
        if np.random.rand() < self.p:
            ids = []
            for id in range(asset.J):
                if np.random.rand() >= self.rate:
                    ids.append(id)
            if root not in ids:
                ids.append(root)
            keep: List[int|str] = select_k([i for i in range(asset.J) if i in ids], self.max_bones)
            if root not in keep:
                keep[0] = root
            asset.set_order(new_orders=keep)
        elif asset.J > self.max_bones:
            ids = select_k([i for i in range(asset.J)], k=self.max_bones)
            if root not in ids:
                ids[0] = root
            keep: List[int|str] = [i for i in range(asset.J) if i in ids]
            asset.set_order(new_orders=keep)

@dataclass(frozen=True)
class AugmentJointDiscrete(Augment):
    # perturb the skeleton with probability p
    p: float
    
    # num of discretized coord
    discrete: int
    
    # continuous range
    continuous_range: Tuple[float, float]
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentJointDiscrete':
        cls.check_keys(kwargs)
        return AugmentJointDiscrete(
            p=kwargs.get('p', 0.),
            discrete=kwargs.get('discrete', 256),
            continuous_range=kwargs.get('continuous_range', [-1., 1.]),
        )
    
    def _discretize(
        self,
        t: ndarray,
        continuous_range: Tuple[float, float],
        num_discrete: int,
    ) -> ndarray:
        lo, hi = continuous_range
        assert hi >= lo
        t = (t - lo) / (hi - lo)
        t *= num_discrete
        return np.clip(t.round(), 0, num_discrete - 1).astype(np.int64)

    def _undiscretize(
        self,
        t: ndarray,
        continuous_range: Tuple[float, float],
        num_discrete: int,
    ) -> ndarray:
        lo, hi = continuous_range
        assert hi >= lo
        t = t.astype(np.float32) + 0.5
        t /= num_discrete
        return t * (hi - lo) + lo

    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() < self.p:
            joints = asset.joints
            if joints is not None and asset.matrix_local is not None:
                joints = self._undiscretize(self._discretize(
                        joints,
                        self.continuous_range,
                        self.discrete,
                    ),
                    self.continuous_range,
                    self.discrete,
                )
                asset.matrix_local[:, :3, 3] = joints

@dataclass(frozen=True)
class AugmentJointPerturb(Augment):
    # perturb the skeleton with probability p
    p: float
    
    # jitter sigma on joints
    sigma: float
    
    # jitter clip on joints
    clip: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentJointPerturb':
        cls.check_keys(kwargs)
        return AugmentJointPerturb(
            p=kwargs.get('p', 0.),
            sigma=kwargs.get('sigma', 0.),
            clip=kwargs.get('clip', 0.),
        )
        
    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() < self.p and asset.matrix_local is not None:
            asset.matrix_local[:, :3] += np.clip(
                np.random.normal(0, self.sigma, (asset.J, 3)),
                -self.clip,
                self.clip,
            )

@dataclass(frozen=True)
class AugmentLBS(Augment):
    # apply a random pose with probability p
    random_pose_p: float
    
    # random pose angle range
    random_pose_angle: float
    
    # random scale 
    random_scale_range: Tuple[float, float]
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLBS':
        cls.check_keys(kwargs)
        return AugmentLBS(
            random_pose_p=kwargs.get('random_pose_p', 0.),
            random_pose_angle=kwargs.get('random_pose_angle', 0.),
            random_scale_range=kwargs.get('random_scale_range', (1., 1.)),
        )
    
    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
    
    def transform(self, asset: Asset, **kwargs):
        def get_matrix_basis(angle: float):
            matrix = axis_angle_to_matrix((np.random.rand(asset.J, 3) - 0.5) * angle / 180 * np.pi * 2).astype(np.float32)
            return matrix
        
        if np.random.rand() < self.random_pose_p and asset.joints is not None:
            matrix_basis = get_matrix_basis(self.random_pose_angle)
            max_offset = (asset.joints.max(axis=0) - asset.joints.min(axis=0)).max()
            matrix_basis[:, :3, :3] *= np.tile(np.random.uniform(low=self.random_scale_range[0], high=self.random_scale_range[1], size=(asset.J, 1, 1)), (1, 3, 3))
            asset.vertices_with_pose(matrix_basis=matrix_basis, inplace=True)

@dataclass(frozen=True)
class AugmentLinear(Augment):
    # apply random rotation with probability p
    random_rotate_p: float
    
    # random rotation angle(degree)
    random_rotate_angle: float
    
    # swap x with probability p
    random_flip_x_p: float
    
    # swap y with probability p
    random_flip_y_p: float
    
    # swap z with probability p
    random_flip_z_p: float
    
    # probability to pick an angle in static_rotate_x
    static_rotate_x_p: float
    
    # rotate around x axis among given angles(degrees)
    static_rotate_x: List[float]
    
    # probability to pick an angle in static_rotate_y
    static_rotate_y_p: float
    
    # rotate around y axis among given angles(degrees)
    static_rotate_y: List[float]
    
    # probability to pick an angle in static_rotate_z
    static_rotate_z_p: float
    
    # rotate around z axis among given angles(degrees)
    static_rotate_z: List[float]
    
    # apply random scaling with probability p
    random_scale_p: float
    
    # random scaling xyz axis
    random_scale: Tuple[float, float]
    
    # randomly change xyz orientation
    random_transpose: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        if kwargs.get('random_flip_x_p', 0) > 0 or kwargs.get('random_flip_y_p', 0) > 0 or kwargs.get('random_flip_z_p', 0) > 0:
            print("\033[31mWARNING: random flip is enabled and is very likely to confuse ar model !\033[0m")
        cls.check_keys(kwargs)
        return AugmentLinear(
            random_rotate_p=kwargs.get('random_rotate_p', 0.),
            random_rotate_angle=kwargs.get('random_rotate_angle', 0.),
            random_flip_x_p=kwargs.get('random_flip_x_p', 0.),
            random_flip_y_p=kwargs.get('random_flip_y_p', 0.),
            random_flip_z_p=kwargs.get('random_flip_z_p', 0.),
            static_rotate_x_p=kwargs.get('static_rotate_x_p', 0.),
            static_rotate_x=kwargs.get('static_rotate_x', []),
            static_rotate_y_p=kwargs.get('static_rotate_y_p', 0.),
            static_rotate_y=kwargs.get('static_rotate_y', []),
            static_rotate_z_p=kwargs.get('static_rotate_z_p', 0.),
            static_rotate_z=kwargs.get('static_rotate_z', []),
            random_scale_p=kwargs.get('random_scale_p', 0.),
            random_scale=kwargs.get('random_scale', [1.0, 1.0]),
            random_transpose=kwargs.get('random_transpose', 0.),
        )

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        r = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.random_rotate_p:
            angle = self.random_rotate_angle
            axis_angle = (np.random.rand(3) - 0.5) * angle / 180 * np.pi * 2
            r = R.from_rotvec(axis_angle).as_matrix()
            r = np.pad(r, ((0, 1), (0, 1)), 'constant', constant_values=0.)
            r[3, 3] = 1.
        
        if np.random.uniform(0, 1) < self.random_flip_x_p:
            r @= np.array([
                [-1.0, 0.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0, 1.0, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.random_flip_y_p:
            r @= np.array([
                [1.0,  0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0,  0.0, 1.0, 0.0],
                [0.0,  0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.random_flip_z_p:
            r @= np.array([
                [1.0, 0.0,  0.0, 0.0],
                [0.0, 1.0,  0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0,  0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.static_rotate_x_p:
            assert len(self.static_rotate_x) > 0, "static rotation of x is enabled, but static_rotate_x is empty"
            angle = np.random.choice(self.static_rotate_x) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [ 1.0, 0.0, 0.0, 0.0],
                [ 0.0,   c,   s, 0.0],
                [ 0.0,  -s,   c, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.static_rotate_y_p:
            assert len(self.static_rotate_y) > 0, "static rotation of y is enabled, but static_rotate_y is empty"
            angle = np.random.choice(self.static_rotate_y) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [   c, 0.0,  -s, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [   s, 0.0,   c, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.static_rotate_z_p:
            assert len(self.static_rotate_z) > 0, "static rotation of z is enabled, but static_rotate_z is empty"
            angle = np.random.choice(self.static_rotate_z) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [   c,   s, 0.0, 0.0],
                [  -s,   c, 0.0, 0.0],
                [ 0.0, 0.0, 1.0, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.random_scale_p:
            scale_x = np.random.uniform(self.random_scale[0], self.random_scale[1])
            scale_y = np.random.uniform(self.random_scale[0], self.random_scale[1])
            scale_z = np.random.uniform(self.random_scale[0], self.random_scale[1])
            r @= np.array([
                [scale_x, 0.0, 0.0, 0.0],
                [0.0, scale_y, 0.0, 0.0],
                [0.0, 0.0, scale_z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.random_transpose:
            permutations = [
                (0, 1, 2),  # x, y, z
                (0, 2, 1),  # x, z, y
                (1, 0, 2),  # y, x, z
                (1, 2, 0),  # y, z, x
                (2, 0, 1),  # z, x, y
                (2, 1, 0),  # z, y, x
            ]
            direction_signs = [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
            ]
            perm = permutations[np.random.randint(0, 6)]
            sign = direction_signs[np.random.randint(0, 8)]
            m = np.zeros((4, 4))
            for i in range(3):
                m[i, perm[i]] = sign[i]
            m[3, 3] = 1.0
            r = m @ r
        
        trans_vertex = r @ trans_vertex
        
        # apply transform here
        asset.transform(trans=trans_vertex)

@dataclass(frozen=True)
class AugmentAffine(Augment):
    # final normalization cube
    normalize_into: Tuple[float, float]

    # randomly scale coordinates with probability p
    random_scale_p: float
    
    # scale range (lower, upper)
    random_scale: Tuple[float, float]
    
    # randomly shift coordinates with probability p
    random_shift_p: float
    
    # shift range (lower, upper)
    random_shift: Tuple[float, float]
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAffine':
        cls.check_keys(kwargs)
        return AugmentAffine(
            normalize_into=kwargs.get('normalize_into', [-1.0, 1.0]),
            random_scale_p=kwargs.get('random_scale_p', 0.),
            random_scale=kwargs.get('random_scale', [1., 1.]),
            random_shift_p=kwargs.get('random_shift_p', 0.),
            random_shift=kwargs.get('random_shift', [0., 0.]),
        )

    def transform(self, asset: Asset, **kwargs):
        if asset.vertices is None:
            raise ValueError("do not have vertices")
        bound_min = asset.vertices.min(axis=0)
        bound_max = asset.vertices.max(axis=0)
        if asset.joints is not None:
            joints_bound_min = asset.joints.min(axis=0)
            joints_bound_max = asset.joints.max(axis=0)            
            bound_min = np.minimum(bound_min, joints_bound_min)
            bound_max = np.maximum(bound_max, joints_bound_max)
        
        trans_vertex = np.eye(4, dtype=np.float32)
        
        trans_vertex = _trans_to_m(-(bound_max + bound_min)/2) @ trans_vertex
        
        if self.normalize_into is not None:
            # scale into the cube
            normalize_into = self.normalize_into
            scale = np.max((bound_max - bound_min) / (normalize_into[1] - normalize_into[0]))
            trans_vertex = _scale_to_m(1. / scale) @ trans_vertex
            
            bias = (normalize_into[0] + normalize_into[1]) / 2
            trans_vertex = _trans_to_m(np.array([bias, bias, bias], dtype=np.float32)) @ trans_vertex
        
        if np.random.rand() < self.random_scale_p:
            scale = _scale_to_m(np.random.uniform(self.random_scale[0], self.random_scale[1]))
            trans_vertex = scale @ trans_vertex

        if np.random.rand() < self.random_shift_p:
            l, r = self.random_shift
            shift_vals = np.array([
                np.random.uniform(l, r),
                np.random.uniform(l, r),
                np.random.uniform(l, r),
            ], dtype=np.float32)
            if self.normalize_into is not None:
                def _apply(v: ndarray, trans: ndarray) -> ndarray:
                    return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
                lo, hi = self.normalize_into
                pts_min = _apply(bound_min[None, :], trans_vertex)[0]
                pts_max = _apply(bound_max[None, :], trans_vertex)[0]
                low_allowed = lo - pts_min
                high_allowed = hi - pts_max
                shift_vals = np.array([
                    np.random.uniform(low_allowed[0], high_allowed[0]),
                    np.random.uniform(low_allowed[1], high_allowed[1]),
                    np.random.uniform(low_allowed[2], high_allowed[2]),
                ], dtype=np.float32)
            shift = _trans_to_m(shift_vals.astype(np.float32))
            trans_vertex = shift @ trans_vertex
        asset.transform(trans=trans_vertex)

@dataclass(frozen=True)
class AugmentJitter(Augment):
    # probability
    p: float
    
    # jitter sigma on vertices
    vertex_sigma: float
    
    # jitter clip on vertices
    vertex_clip: float
    
    # jitter sigma on normals
    normal_sigma: float
    
    # jitter clip on normals
    normal_clip: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentJitter':
        cls.check_keys(kwargs)
        return AugmentJitter(
            p=kwargs.get('p', 0.5),
            vertex_sigma=kwargs.get('vertex_sigma', 0.),
            vertex_clip=kwargs.get('vertex_clip', 0.),
            normal_sigma=kwargs.get('normal_sigma', 0.),
            normal_clip=kwargs.get('normal_clip', 0.),
        )
    
    def transform(self, asset: Asset, **kwargs):
        vertex_sigma = self.vertex_sigma
        vertex_clip = self.vertex_clip
        normal_sigma = self.normal_sigma
        normal_clip = self.normal_clip
        
        if np.random.rand() < self.p:
            scale = np.random.rand() + 1e-6
            vertex_sigma *= scale
            vertex_clip *= scale
            scale = np.random.rand() + 1e-6
            normal_sigma *= scale
            normal_clip *= scale
            if vertex_sigma > 0 and asset.vertices is not None:
                noise = np.clip(np.random.randn(*asset.vertices.shape) * vertex_sigma, -vertex_clip, vertex_clip).astype(np.float32)
                asset.vertices += noise
            
            if normal_sigma > 0:
                if asset.vertex_normals is not None:
                    noise = np.clip(np.random.randn(*asset.vertex_normals.shape) * normal_sigma, -normal_clip, normal_clip).astype(np.float32)
                    asset.vertex_normals += noise
                
                if asset.face_normals is not None:
                    noise = np.clip(np.random.randn(*asset.face_normals.shape) * normal_sigma, -normal_clip, normal_clip).astype(np.float32)
                    asset.face_normals += noise

@dataclass(frozen=True)
class AugmentNormalize(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalize':
        cls.check_keys(kwargs)
        return AugmentNormalize()
    
    def transform(self, asset: Asset, **kwargs):
        epsilon = 1e-10
        if asset.vertex_normals is not None:
            vertex_norms = np.linalg.norm(asset.vertex_normals, axis=1, keepdims=True)
            vertex_norms = np.maximum(vertex_norms, epsilon)
            asset.vertex_normals = asset.vertex_normals / vertex_norms
            asset.vertex_normals = np.nan_to_num(asset.vertex_normals, nan=0., posinf=0., neginf=0.) # type: ignore
        
        if asset.face_normals is not None:
            face_norms = np.linalg.norm(asset.face_normals, axis=1, keepdims=True)
            face_norms = np.maximum(face_norms, epsilon)
            asset.face_normals = asset.face_normals / face_norms        
            asset.face_normals = np.nan_to_num(asset.face_normals, nan=0., posinf=0., neginf=0.) # type: ignore

def _trans_to_m(v: ndarray):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v
    return m

def _scale_to_m(r: ndarray|float):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = r
    m[1, 1] = r
    m[2, 2] = r
    m[3, 3] = 1.
    return m

def get_augments(*args) -> List[Augment]:
    MAP = {
        'trim': AugmentTrim,
        'delete': AugmentDelete,
        'drop_part': AugmentDropPart,
        'collapse': AugmentCollapse,
        'lbs': AugmentLBS,
        'linear': AugmentLinear,
        'affine': AugmentAffine,
        'jitter': AugmentJitter,
        'joint_perturb': AugmentJointPerturb,
        'joint_discrete': AugmentJointDiscrete,
        'normalize': AugmentNormalize,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments