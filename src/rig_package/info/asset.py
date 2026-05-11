from dataclasses import dataclass, field
from numpy import ndarray
from scipy.spatial import cKDTree # type: ignore
from typing import Dict, List, Optional, Tuple

import numpy as np
import os
import trimesh

from ..utils import assert_list, assert_ndarray, linear_blend_skinning, sample_vertex_groups
from .voxel import Voxel

@dataclass
class Asset():
    
    # vertices of merged mesh in edit space, shape (N, 3)
    vertices: Optional[ndarray]=None
    
    # faces of merged mesh, shape (F, 3)
    faces: Optional[ndarray]=None
    
    # vertex normals of merged mesh in edit space, shape (N, 3), calculated by trimesh
    vertex_normals: Optional[ndarray]=None
    
    # face normals of merged mesh in edit space, shape (F, 3), calculated by trimesh
    face_normals: Optional[ndarray]=None
    
    # offset of vertices in each part, shape (P,),
    # vertices[vertex_bias[i-1]:vertex_bias[i]] are in the same part (vertex_bias[-1]=0)
    vertex_bias: Optional[ndarray]=None
    
    # offset of faces in each part, shape (P,),
    # faces[face_bias[i-1]:face_bias[i]] are in the same part (face_bias[-1]=0)
    face_bias: Optional[ndarray]=None
    
    # name of each mesh part, shape (P,)
    mesh_names: Optional[List[str]]=None
    
    # name of each joint, shape (J,)
    joint_names: Optional[List[str]]=None
    
    # parent index of each joint, shape (J,), -1 for root
    parents: Optional[ndarray]=None
    
    # length of each bone indicating euclidean distance between head and tail(which is proposed in blender), shape (J,)
    lengths: Optional[ndarray]=None
    
    # matrix to convert from edit space(or motion space) to world space, shape (4, 4)
    matrix_world: Optional[ndarray]=None
    
    # local matrix of each joint, shape (J, 4, 4)
    matrix_local: Optional[ndarray]=None
    
    # matrix to convert from edit space to motion space, shape (frames, J, 4, 4)
    matrix_basis: Optional[ndarray]=None
    
    # name of the armature
    armature_name: Optional[str]=None
    
    # skinning weights, shape (N, J)
    skin: Optional[ndarray]=None
    
    ###########################################################################
    cls: Optional[str]=None
    path: Optional[str]=None
    vertex_groups: Dict[str, ndarray]=field(default_factory=dict)
    sampled_vertices: Optional[ndarray]=None
    sampled_normals: Optional[ndarray]=None
    sampled_vertex_groups: Optional[Dict[str, ndarray]]=None
    skin_samples: Optional[int]=None
    
    meta: Optional[Dict]=None
    
    @property
    def dirname(self) -> str:
        """return directory name of the asset"""
        if self.path is None:
            return ""
        return os.path.dirname(self.path)
    
    @property
    def N(self) -> int:
        """return number of vertices"""
        if self.vertices is None:
            return 0
        return self.vertices.shape[0]
    
    @property
    def F(self) -> int:
        """return number of faces"""
        if self.faces is None:
            return 0
        return self.faces.shape[0]
    
    @property
    def J(self) -> int:
        """return number of joints"""
        if self.parents is None:
            return 0
        return self.parents.shape[0]
    
    @property
    def P(self) -> int:
        """return number of mesh parts"""
        self._build_bias()
        if self.vertex_bias is None:
            return 0
        return self.vertex_bias.shape[0]
    
    @property
    def root(self) -> int:
        """return the index of root joint"""
        if self.parents is None:
            return -1
        for i, p in enumerate(self.parents):
            if p == -1:
                return i
        raise ValueError("no root found")
     
    @property
    def joints(self) -> ndarray|None:
        """return joints in edit space, shape (J, 3)"""
        if self.matrix_local is None:
            return None
        return self.matrix_local[:, :3, 3]
    
    @property
    def skeleton(self) -> ndarray|None:
        """return skeleton where joint is followed by parent, shape (J-1, 6), ignore root"""
        if self.joints is None or self.parents is None:
            return None
        indices = np.linspace(0, self.J-1, num=self.J, dtype=int)[self.parents!=-1]
        return np.concatenate([self.joints[indices], self.joints[self.parents[indices]]], axis=1)
    
    @property
    def dfs_order(self) -> List[int]:
        """return the dfs order of joints"""
        if self.parents is None:
            return []
        sons = [[] for _ in range(self.J)]
        stack = []
        for i, p in enumerate(self.parents):
            if p == -1:
                stack.append(i)
                continue
            sons[p].append(i)
        order = []
        while len(stack) > 0:
            u = stack.pop()
            order.append(u)
            for s in reversed(sons[u]):
                stack.append(s)
        return order
    
    @property
    def tails(self) -> ndarray|None:
        """
            Return tails in edit space, shape (J, 3). The bone is extrueded along local Y axis, in accordance with Blender.
        """
        joints = self.joints
        matrix_local = self.matrix_local
        if joints is None or self.lengths is None or matrix_local is None:
            return None
        
        x = np.array([0.0, 1.0, 0.0])
        x = self.lengths * x[:, np.newaxis]
        y = np.zeros((self.J, 3))
        for i in range(self.J):
            y[i] = matrix_local[i, :3, :3] @ x[:, i]
        return joints + y
    
    def _build_bias(self):
        if self.vertex_bias is None and self.vertices is not None:
            self.vertex_bias = np.array([self.vertices.shape[0]])
        if self.face_bias is None and self.faces is not None:
            self.face_bias = np.array([self.faces.shape[0]])
    
    def get_vertex_slice(self, index: int) -> slice:
        """return slice of vertices of a specific part"""
        self._build_bias()
        if self.vertex_bias is None:
            return slice(0, 0)
        if index == 0:
            return slice(0, self.vertex_bias[0])
        return slice(self.vertex_bias[index-1], self.vertex_bias[index])
    
    def get_face_slice(self, index: int) -> slice:
        """return slice of faces of a specific part"""
        self._build_bias()
        if self.face_bias is None:
            return slice(0, 0)
        if index == 0:
            return slice(0, self.face_bias[index])
        return slice(self.face_bias[index-1], self.face_bias[index])
    
    def names_to_ids(self, arr: List[int|str]) -> List[int]:
        for s in arr:
            if isinstance(s, str) and (self.joint_names is None or s not in self.joint_names):
                raise ValueError(f"do not find {s} in joint_names")
            elif not isinstance(s, int) and not isinstance(s, str):
                raise ValueError(f"element must be int or str")
        if self.joint_names is not None:
            _name_to_id = {s: i for (i, s) in enumerate(self.joint_names)}
        else:
            _name_to_id = {}
        return [_name_to_id[s] if isinstance(s, str) else s for s in arr]
    
    def set_order(
        self,
        new_orders: List[int|str],
        merge_skin: bool=True,
        do_not_normalize: bool=False
    ):
        """
        Rearrange the order of the joints.
        Args:
            new_orders: A list of int or bone names to indicate orders.
            For example, if the first element is 2, then the rearranged
            joint will be the second first joint in the current skeleton.
            
            merge_skin: If True, if some joints are merged, skin will be
            added to its nearest ancestor. Otherwise completely removes
            skin and finally normalized.
            
            do_not_normalize: Do not normalize skin.
        """
        if len(np.unique(new_orders)) != len(new_orders):
            raise ValueError("multiple values found in new_orders")
        _new_orders = self.names_to_ids(arr=new_orders)
        ancestors = []
        grandsons = []
        beyond_root = []
        root_id = 0
        if self.parents is not None:
            new_positions = [0 for i in range(self.J)]
            new_parents = [-1 for i in range(self.J)]
            for i, x in enumerate(_new_orders):
                new_positions[x] = i
            set_new_orders = set(_new_orders)
            roots = 0
            for i in self.dfs_order:
                if i not in set_new_orders:
                    if self.parents[i] == -1:
                        new_positions[i] = -1
                        beyond_root.append(i)
                    else:
                        new_positions[i] = new_positions[self.parents[i]]
                        if new_positions[i] == -1:
                            beyond_root.append(i)
                        else:
                            ancestors.append(new_positions[i])
                            grandsons.append(i)
                else:
                    if self.parents[i] == -1:
                        new_parents[i] = -1
                    else:
                        new_parents[i] = new_positions[self.parents[i]]
                    if new_parents[i] == -1:
                        roots += 1
                        root_id = new_positions[i]
                        if roots >= 2:
                            raise ValueError(f"multiple roots found: {self.path} {self.parents} {new_orders}")
            self.parents = np.array(new_parents)[_new_orders]
        if self.joint_names is not None:
            _joint_names = [self.joint_names[u] for u in _new_orders]
            self.joint_names = _joint_names
        if self.lengths is not None:
            self.lengths = self.lengths[_new_orders]
        if self.matrix_local is not None:
            self.matrix_local = self.matrix_local[_new_orders]
        if self.matrix_basis is not None:
            self.matrix_basis = self.matrix_basis[:, _new_orders]
        if self.skin is not None:
            if merge_skin:
                skin = self.skin.copy()
                self.skin = skin[:, _new_orders]
                for x, y in zip(ancestors, grandsons):
                    self.skin[:, x] += skin[:, y]
                self.skin[:, root_id] += skin[:, beyond_root].sum(axis=1)
            else:
                self.skin = self.skin[:, _new_orders]
                if not do_not_normalize:
                    self.normalize_skin()
    
    def delete_joints(self, joints_to_remove: List[int|str]):
        """
        Delete joints and their corresponding values.
        """
        _joints_to_remove = set(self.names_to_ids(arr=joints_to_remove))
        new_orders: List[int|str] = [i for i in range(self.J) if i not in _joints_to_remove]
        self.set_order(new_orders=new_orders)
    
    def delete_vertices(self, vertices_to_remove: List[int]|ndarray):
        """
        Delete vertices and their corresponding values.
        """
        if self.vertices is None:
            return
        if isinstance(vertices_to_remove, list):
            vertices_to_remove = np.array(vertices_to_remove)
        mask = np.ones(self.N, dtype=bool)
        mask[vertices_to_remove] = False
        indices = np.where(mask)[0]
        
        # handle vertex bias
        if self.vertex_bias is not None:
            cumsum_mask = np.cumsum(mask)
            self.vertex_bias = cumsum_mask[self.vertex_bias-1]
        
        N = self.N
        self.vertices = self.vertices[indices]
        if self.vertex_normals is not None:
            self.vertex_normals = self.vertex_normals[indices]
        if self.skin is not None:
            self.skin = self.skin[indices, :]
        if self.faces is not None: # keep faces
            face_mask = np.all(np.isin(self.faces, indices), axis=1)
            self.faces = self.faces[face_mask]
            old_to_new = np.zeros(N, dtype=np.int32)
            old_to_new[indices] = np.arange(len(indices))
            self.faces = old_to_new[self.faces]
            if self.face_normals is not None:
                self.face_normals = self.face_normals[indices]
            # handle face bias
            if self.face_bias is not None:
                cumsum_face_mask = np.cumsum(face_mask)
                self.face_bias = cumsum_face_mask[self.face_bias-1]
        
        self._build_bias()
    
    def normalize_skin(self) -> 'Asset':
        """
        Normalize skin so that add up to 1.
        """
        if self.skin is None:
            return self
        self.skin = self.skin / np.maximum(np.sum(self.skin, axis=1, keepdims=True), 1e-8)
        return self
    
    def build_normals(self):
        """
        Build vertex_normals and face_normals using trimesh.
        """
        if self.vertices is None:
            raise ValueError("do not have vertices")
        if self.faces is None:
            raise ValueError("do not have faces")
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False, maintain_order=True)
        self.vertex_normals = mesh.vertex_normals.copy()
        self.face_normals = mesh.face_normals.copy()
    
    def normalize_vertices(
        self,
        range: Optional[Tuple[float, float]]=None,
        range_x: Optional[Tuple[float, float]]=None,
        range_y: Optional[Tuple[float, float]]=None,
        range_z: Optional[Tuple[float, float]]=None,
    ):
        """
        Normalize vertices into cube in edit space. If range_x/y/z is provided,
        use range_x/y/z, otherwise use range by default.
        """
        if self.vertices is None:
            return
        if range is None:
            if range_x is None:
                raise ValueError("range_x is None, but range is missing")
            if range_y is None:
                raise ValueError("range_y is None, but range is missing")
            if range_z is None:
                raise ValueError("range_z is None, but range is missing")
            _range_x = range_x
            _range_y = range_y
            _range_z = range_z
        else:
            _range_x = range if range_x is None else range_x
            _range_y = range if range_y is None else range_y
            _range_z = range if range_z is None else range_z
        v_min = self.vertices.min(axis=0)
        v_max = self.vertices.max(axis=0)
        scale_range = (v_max - v_min).max()
        # normalize into [0, 1]^3
        v = (self.vertices - v_min) / scale_range
        mid_point = (v.min(axis=0) + v.max(axis=0)) / 2
        bias = np.array([0.5, 0.5, 0.5]) - mid_point
        v += bias
        dx = (_range_x[1] - _range_x[0])
        dy = (_range_y[1] - _range_y[0])
        dz = (_range_z[1] - _range_z[0])
        if self.faces is not None and np.abs(dx-dy) > 1e-3 or np.abs(dy-dz) > 1e-3 or np.abs(dy-dz) > 1e-3:
            raise ValueError("do not support non-uniform normalization")
        v[:, 0] = v[:, 0] * dx + _range_x[0]
        v[:, 1] = v[:, 1] * dy + _range_y[0]
        v[:, 2] = v[:, 2] * dz + _range_z[0]
        self.vertices = v
        if self.matrix_local is not None:
            jv = (self.matrix_local[:, :3, 3] - v_min) / scale_range + bias
            self.matrix_local[:, 0, 3] = jv[:, 0] * dx + _range_x[0]
            self.matrix_local[:, 1, 3] = jv[:, 1] * dy + _range_y[0]
            self.matrix_local[:, 2, 3] = jv[:, 2] * dz + _range_z[0]
    
    def get_matrix(
        self,
        matrix_basis: ndarray,
    ) -> ndarray:
        """
        Get pose matrix in motion space using forward kinetics.
        """
        J = self.J
        parents = self.parents
        if parents is None:
            raise ValueError("do not have parents")
        if self.matrix_local is None:
            raise ValueError("do not have matrix_local")
        assert_ndarray(matrix_basis, "matrix_basis", (J, 4, 4))
        matrix = np.zeros((J, 4, 4))
        for i in self.dfs_order:
            pid = parents[i]
            if pid==-1:
                matrix[i] = self.matrix_local[i] @ matrix_basis[i]
            else:
                matrix_parent = matrix[pid]
                matrix_local_parent = self.matrix_local[pid]
                
                matrix[i] = (
                    matrix_parent @
                    (np.linalg.inv(matrix_local_parent) @ self.matrix_local[i]) @
                    matrix_basis[i]
                )
        return matrix
    
    def vertices_with_pose(
        self,
        matrix_basis: ndarray,
        inplace: bool=True,
    ) -> ndarray:
        """
        Apply pose to vertices and return the deformed vertices.
        
        Args:
            inplace: if True, change vertices and all motion related fileds of the asset.
        """
        if self.vertices is None:
            raise ValueError("do not have vertices")
        if self.matrix_local is None:
            raise ValueError("do not have matrix_local")
        if self.joints is None:
            raise ValueError("do not have joints")
        if self.skin is None:
            raise ValueError("do not have skin")
        matrix = self.get_matrix(matrix_basis=matrix_basis)
        vertices = linear_blend_skinning(
            vertices=self.vertices,
            matrix_local=self.matrix_local,
            matrix=matrix,
            skin=self.skin,
            pad=1,
            value=1.0,
        )
        if inplace:
            self.vertices = vertices
            if self.faces is not None:
                self.build_normals()
            self.matrix_local = matrix
        return vertices
    
    def transform(self, trans: ndarray):
        """trans: 4x4 affine matrix"""
        def _apply(v: ndarray, trans: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
        
        if self.vertices is not None:
            self.vertices = _apply(self.vertices, trans)
        if self.matrix_local is not None:
            self.matrix_local = trans @ self.matrix_local
        self.build_normals()
    
    def trim_skeleton(self):
        """remove all leaf bones and coordinate bones"""
        if self.skin is None:
            return
        if self.parents is None:
            return
        has_skin = self.skin.sum(axis=0) > 1e-6
        if not np.any(has_skin):
            return
        sons = [[] for _ in range(self.J)]
        good_sons = [[] for _ in range(self.J)]
        sub_tree_has_skin = [False for _ in range(self.J)]
        dfs_order = self.dfs_order
        for u in dfs_order:
            p = self.parents[u]
            if p != -1:
                sons[p].append(u)
        for u in reversed(dfs_order):
            p = self.parents[u]
            if has_skin[u]:
                sub_tree_has_skin[u] = True
            else:
                for v in sons[u]:
                    if sub_tree_has_skin[v]:
                        sub_tree_has_skin[u] = True
                        break
        keep = [False for _ in range(self.J)]
        for u in dfs_order:
            for v in sons[u]:
                if sub_tree_has_skin[v]:
                    good_sons[u].append(v)
            if has_skin[u]:
                keep[u] = True
            else:
                p = self.parents[u]
                if len(good_sons[u]) >= 2:
                    keep[u] = True
                elif len(good_sons[u]) == 1 and p != -1:
                    if len(good_sons[p]) >= 2:
                        keep[u] = True
                    elif len(good_sons[p]) == 1 and good_sons[p][0] != u:
                        keep[u] = True
        joints_to_remove: List[int|str] = [i for i in range(self.J) if not keep[i]]
        self.delete_joints(joints_to_remove=joints_to_remove)
    
    def check_field(self):
        
        def _check_array(arr, name, shape, dtype=None):
            if arr is not None:
                assert_ndarray(arr, name=name, shape=shape, dtype=dtype)
        
        def _check_list(arr, name, dtype=None):
            if arr is not None:
                assert_list(arr, name=name, dtype=dtype)
        
        _check_array(self.vertices, name="vertices", shape=(self.N, 3))
        _check_array(self.faces, name="faces", shape=(self.F, 3))
        _check_array(self.vertex_normals, name="vertex_normals", shape=(self.N, 3))
        _check_array(self.face_normals, name="face_normals", shape=(self.F, 3))
        _check_array(self.vertex_bias, name="vertex_bias", shape=(self.P,), dtype=np.integer)
        _check_array(self.face_bias, name="face_bias", shape=(self.P,), dtype=np.integer)
        _check_list(self.mesh_names, name="mesh_names", dtype=str)
        _check_list(self.joint_names, name="joint_names", dtype=str)
        _check_array(self.parents, name="parents", shape=(-1,), dtype=np.integer)
        _check_array(self.lengths, name="lengths", shape=(-1,))
        _check_array(self.matrix_world, name="matrix_world", shape=(4, 4))
        _check_array(self.matrix_local, name="matrix_local", shape=(self.J, 4, 4))
        _check_array(self.matrix_basis, name="matrix_basis", shape=(self.F, self.J, 4, 4))
        if self.armature_name is not None:
            if not isinstance(self.armature_name, str):
                raise ValueError(f"armature_name should be str")
        _check_array(self.skin, name="skin", shape=(self.N, self.J))
        
        if self.vertices is not None and self.vertex_normals is not None:
            if self.vertices.shape[0] != self.vertex_normals.shape[0]:
                raise ValueError(f"shapes of vertices and vertex_normals do not match: {self.vertices.shape} and {self.vertex_normals.shape}")
        
        if self.faces is not None and self.face_normals is not None:
            if self.faces.shape[0] != self.face_normals.shape[0]:
                raise ValueError(f"shapes of faces and face_normals do not match: {self.faces.shape} and {self.face_normals.shape}")
        
        if self.vertex_bias is not None:
            if self.vertices is None:
                raise ValueError("have vertex_bias, but do not have vertices")
            if self.vertex_bias[-1] != self.N:
                raise ValueError(f"vertex_bias must end with number of vertices {self.N}")
        
        if self.face_bias is not None:
            if self.faces is None:
                raise ValueError("have face_bias, but do not have faces")
            if self.face_bias[-1] != self.F:
                raise ValueError(f"vertex_bias must end with number of vertices {self.N}")
        
        if self.matrix_local is not None and self.matrix_basis is not None:
            if self.matrix_local.shape[0] != self.matrix_basis.shape[1]:
                raise ValueError(f"number of joints do not match in matix_local and matrix_basis: {self.matrix_local.shape[0]} and {self.matrix_basis.shape[1]}")
        
        if self.joint_names is not None and self.matrix_local is not None:
            if len(self.joint_names) != self.matrix_local.shape[0]:
                raise ValueError(f"number of joints do not match in joint_names and matrix_local: {len(self.joint_names)} and {self.matrix_local.shape[0]}")
        
        if self.skin is not None and self.matrix_local is not None:
            if self.skin.shape[1] != self.matrix_local.shape[0]:
                raise ValueError(f"number of joints do not match in skin and matrix_local: {self.skin.shape[0]} and {self.matrix_local.shape[0]}")
        
        if self.parents is not None:
            if (self.parents==-1).sum() != 1:
                raise ValueError(f"no root or multiple roots found, count: {(self.parents==-1).sum()}")
    
    def voxel(self, resolution: int=128, voxel_size: Optional[float]=None) -> Voxel:
        """
        Return a voxel created from mesh.
        Args:
            resolution: Maximum number of cubes along one axis.
            
            voxel_size: Forcibly asign length of the cube with this value.
        """
        import open3d as o3d
        if self.vertices is None:
            raise ValueError("do not have vertices")
        if self.faces is None:
            raise ValueError("do not have faces")
        if voxel_size is None:
            max_d = (self.vertices.max(axis=1) - self.vertices.min(axis=1)).max()
            v = max_d / resolution
        else:
            v = voxel_size
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(self.vertices.copy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.faces)
        voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_o3d, voxel_size=v)
        coords = np.array([pt.grid_index for pt in voxel.get_voxels()])
        return Voxel(
            origin=voxel.origin,
            voxel_size=v,
            coords=coords,
        )
    
    def sample_pc(
        self,
        num_samples: int,
        num_vertex_samples: Optional[int]=None,
        face_mask: Optional[ndarray]=None,
        shuffle: bool=True,
    ) -> 'Asset':
        """
        Return a asset where vertices, normals and skin are sampled.
        """
        if self.vertices is None:
            raise ValueError("do not have vertices")
        if self.faces is None:
            raise ValueError("do not have faces")
        if self.vertex_normals is None or self.face_normals is None:
            self.build_normals()
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="face_mask", shape=(self.F,))
        sampled_vertices, sampled_normals, sampled_vertex_groups = sample_vertex_groups(
            vertices=self.vertices,
            faces=self.faces,
            num_samples=num_samples,
            num_vertex_samples=num_vertex_samples,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals,
            vertex_groups=self.skin,
            face_mask=face_mask,
            shuffle=shuffle,
            same=True,
        )
        asset = self.copy()
        asset.vertices = sampled_vertices[:, 0]
        asset.vertex_normals = sampled_normals[:, 0] # type: ignore
        asset.skin = sampled_vertex_groups
        asset.vertex_bias = None
        asset.faces = None
        asset.face_bias = None
        asset.face_normals = None
        asset._build_bias()
        return asset
    
    def copy(self) -> 'Asset':
        def _copy(x):
            if isinstance(x, ndarray):
                return x.copy()
            elif isinstance(x, list):
                return x.copy()
            elif isinstance(x, str):
                return x
            else:
                return None
        return Asset(
            vertices=_copy(self.vertices),
            faces=_copy(self.faces),
            vertex_normals=_copy(self.vertex_normals), # type: ignore
            face_normals=_copy(self.face_normals),
            vertex_bias=_copy(self.vertex_bias),
            face_bias=_copy(self.face_bias),
            mesh_names=_copy(self.mesh_names),
            joint_names=_copy(self.joint_names),
            parents=_copy(self.parents),
            lengths=_copy(self.lengths),
            matrix_world=_copy(self.matrix_world),
            matrix_local=_copy(self.matrix_local),
            matrix_basis=_copy(self.matrix_basis),
            armature_name=_copy(self.armature_name), # type: ignore
            skin=_copy(self.skin),
            cls=_copy(self.cls), # type: ignore
            path=_copy(self.path), # type: ignore
        )
    
    def change_dtype(self, float_dtype=np.float32, int_dtype=np.int32) -> 'Asset':
        """change dtype"""
        def convert(arr):
            if arr is None:
                return None
            if np.issubdtype(arr.dtype, np.floating):
                return arr.astype(float_dtype)
            elif np.issubdtype(arr.dtype, np.integer):
                return arr.astype(int_dtype)
            else:
                return arr
        
        self.vertices = convert(self.vertices)
        self.faces = convert(self.faces)
        self.vertex_normals = convert(self.vertex_normals)
        self.face_normals = convert(self.face_normals)
        self.vertex_bias = convert(self.vertex_bias)
        self.face_bias = convert(self.face_bias)
        self.parents = convert(self.parents)
        self.lengths = convert(self.lengths)
        self.matrix_world = convert(self.matrix_world)
        self.matrix_local = convert(self.matrix_local)
        self.matrix_basis = convert(self.matrix_basis)
        self.skin = convert(self.skin)
        return self
    
    @classmethod
    def from_data(
        c,
        vertices: Optional[ndarray]=None,
        faces: Optional[ndarray]=None,
        vertex_normals: Optional[ndarray]=None,
        face_normals: Optional[ndarray]=None,
        vertex_bias: Optional[ndarray]=None,
        face_bias: Optional[ndarray]=None,
        mesh_names: Optional[List[str]]=None,
        joint_names: Optional[List[str]]=None,
        parents: Optional[ndarray]=None,
        lengths: Optional[ndarray]=None,
        matrix_world: Optional[ndarray]=None,
        matrix_local: Optional[ndarray]=None,
        matrix_basis: Optional[ndarray]=None,
        armature_name: Optional[str]=None,
        skin: Optional[ndarray]=None,
        joints: Optional[ndarray]=None,
        sampled_vertices: Optional[ndarray]=None,
        sampled_skin: Optional[ndarray]=None,
        cls: Optional[str]=None,
        path: Optional[str]=None,
    ) -> 'Asset':
        """
        Return an asset with as many fields as possible.
        """
        if matrix_local is None and joints is not None:
            J = joints.shape[0]
            matrix_local = np.zeros((J, 4, 4), dtype=np.float32)
            matrix_local[...] = np.eye(4)
            matrix_local[:, :3, 3] = joints
        if joint_names is None and matrix_local is not None:
            joints_names = [f"bone_{i}" for i in range(matrix_local.shape[0])]
        
        if sampled_vertices is not None and vertices is not None and sampled_skin is not None:
            tree = cKDTree(sampled_vertices)
            distances, indices = tree.query(vertices)
            _s = sampled_skin[indices]
            skin = _s
        asset = Asset(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            vertex_bias=vertex_bias,
            face_bias=face_bias,
            mesh_names=mesh_names,
            joint_names=joint_names,
            parents=parents,
            lengths=lengths,
            matrix_world=matrix_world,
            matrix_local=matrix_local,
            matrix_basis=matrix_basis,
            armature_name=armature_name,
            skin=skin,
            cls=cls,
            path=path,
        )
        asset.check_field()
        return asset
