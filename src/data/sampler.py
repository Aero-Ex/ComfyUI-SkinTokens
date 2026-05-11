from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy import ndarray
from scipy.spatial import cKDTree # type: ignore
from typing import Dict, Optional

import numpy as np
import random

from ..rig_package.info.asset import Asset
from ..rig_package.utils import sample_vertex_groups
from .spec import ConfigSpec

@dataclass
class SamplerResult():
    sampled_vertices: Optional[ndarray]=None
    sampled_normals: Optional[ndarray]=None
    sampled_vertex_groups: Optional[Dict[str, ndarray]]=None

    # number of sampled skin
    skin_samples: Optional[int]=None

class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        asset: Asset,
    ) -> SamplerResult:
        '''
        Return sampled vertices, sampled normals and vertex groups.
        '''
        pass
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwargs) -> 'Sampler':
        pass

@dataclass
class SamplerMix(Sampler, ConfigSpec):
    num_samples: int
    num_vertex_samples: int
    num_skin_samples: Optional[int]=None
    replace: bool=True
    all_skeleton: Optional[bool]=None
    max_distance: float=0.1
    rate_distance: float=0.1
    
    @classmethod
    def parse(cls, **kwargs) -> 'SamplerMix':
        cls.check_keys(kwargs)
        return SamplerMix(
            num_samples=kwargs.get('num_samples', 0),
            num_vertex_samples=kwargs.get('num_vertex_samples', 0),
            num_skin_samples=kwargs.get('num_skin_samples', None),
            replace=kwargs.get('replace', True),
            all_skeleton=kwargs.get('all_skeleton', None),
            max_distance=kwargs.get('max_distance', 0.1),
            rate_distance=kwargs.get('rate_distance', 0.1),
        )
    
    def sample_on_skin(
        self,
        skin: ndarray,
        vertices: ndarray,
        faces: ndarray,
    ):
        face_has_skin = np.any(skin[faces] > 0, axis=-1)
        if face_has_skin.sum() == 0:
            face_has_skin = np.ones_like(face_has_skin)
        elif self.max_distance < 1e-5:
            return face_has_skin
        else:
            # sample near points
            p = np.unique(faces[face_has_skin].reshape(-1))
            tree = cKDTree(vertices[p])
            dis, _ = tree.query(vertices, k=1)
            dis_skin = np.sqrt(((np.max(vertices[p], axis=0) - np.min(vertices[p], axis=0))**2).sum())
            mask_face_near = np.any(dis[faces] < min(self.max_distance, dis_skin * self.rate_distance), axis=-1)
            face_has_skin |= mask_face_near
        return face_has_skin
    
    def sample(
        self,
        asset: Asset,
    ) -> SamplerResult:
        if asset.vertices is None:
            raise ValueError("do not have vertices")
        if asset.faces is None:
            raise ValueError("do not have faces")
        vertex_groups = []
        mapping = {}
        tot = 0
        for k, v in asset.vertex_groups.items():
            if v.ndim == 1:
                v = v[:, None]
            elif v.ndim != 2:
                raise ValueError(f"ndim of key {k} is {v.ndim}")
            s = tot
            e = tot + v.shape[1]
            mapping[k] = slice(s,e)
            vertex_groups.append(v)
        if len(vertex_groups) > 0:
            vertex_groups = np.concatenate(vertex_groups, axis=1)
        else:
            vertex_groups = None
        final_sampled_vertices, final_sampled_normals, sampled_vertex_groups = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            vertex_normals=asset.vertex_normals,
            face_normals=asset.face_normals,
            vertex_groups=vertex_groups,
            face_mask=None,
            shuffle=True,
            same=True,
        )
        if vertex_groups is not None:
            final_sampled_vertices = final_sampled_vertices[:, 0]
            if final_sampled_normals is not None:
                final_sampled_normals = final_sampled_normals[:, 0]
        final_sampled_vertex_groups = {}
        if sampled_vertex_groups is not None:
            for k, s in mapping.items():
                final_sampled_vertex_groups[k] = sampled_vertex_groups[:, s] # (N, k)
        if vertex_groups is not None and self.num_skin_samples is not None:
            dense_vertices = []
            dense_normals = []
            dense_skin = []
            if 'skin' not in mapping:
                raise ValueError("do not have skin")
            if self.all_skeleton:
                dense_indices = [i for i in range(asset.J)]
            else:
                dense_indices = [random.randint(0, asset.J-1)]
            for indice in dense_indices:
                _s = asset.vertex_groups['skin'][:, indice]
                face_has_skin = self.sample_on_skin(
                    skin=_s,
                    vertices=asset.vertices,
                    faces=asset.faces,
                )
                sampled_vertices, sampled_normals, sampled_skin = sample_vertex_groups(
                    vertices=asset.vertices,
                    faces=asset.faces,
                    vertex_normals=asset.vertex_normals,
                    face_normals=asset.face_normals,
                    vertex_groups=_s,
                    num_samples=self.num_skin_samples,
                    num_vertex_samples=self.num_vertex_samples,
                    face_mask=face_has_skin,
                    shuffle=True,
                    same=True,
                )
                assert sampled_skin is not None
                assert sampled_skin.ndim == 2
                dense_vertices.append(sampled_vertices[:, 0])
                if sampled_normals is not None:
                    dense_normals.append(sampled_normals[:, 0])
                dense_skin.append(sampled_skin[:, 0])
            dense_vertices = np.stack(dense_vertices, axis=0)   # (J, m, 3)
            if len(dense_normals) > 0:
                dense_normals = np.stack(dense_normals, axis=0) # (J, m, 3)
            else:
                dense_normals = None
            dense_skin = np.stack(dense_skin, axis=0)           # (J, m, 1)
            final_sampled_vertex_groups['skin'] = final_sampled_vertex_groups['skin'][:, dense_indices]
            if asset.meta is None:
                asset.meta = {}
            asset.meta['dense_vertices'] = dense_vertices
            asset.meta['dense_normals'] = dense_normals
            asset.meta['dense_skin'] = dense_skin
            asset.meta['dense_indices'] = dense_indices
        return SamplerResult(
            sampled_vertices=final_sampled_vertices,
            sampled_normals=final_sampled_normals if final_sampled_normals is not None else None,
            sampled_vertex_groups=final_sampled_vertex_groups,
            skin_samples=self.num_skin_samples,
        )

def get_sampler(**kwargs) -> Sampler:
    __target__ = kwargs.get('__target__')
    assert __target__ is not None
    del kwargs['__target__']
    if __target__ == 'mix':
        sampler = SamplerMix.parse(**kwargs)
    else:
        raise ValueError(f"sampler method {__target__} not supported")
    return sampler