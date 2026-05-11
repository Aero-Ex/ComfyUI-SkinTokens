from dataclasses import dataclass
from numpy import ndarray
from typing import Optional

import numpy as np

@dataclass
class Voxel():
    # coordinates of mesh
    coords: ndarray
    
    # origin of the voxel
    origin: ndarray
    
    # grid size
    voxel_size: float
    
    # a boolen array
    _voxel: Optional[ndarray]=None
    
    @property
    def voxel(self) -> ndarray:
        if self._voxel is None:
            max_coords = np.max(self.coords, axis=0)
            shape = tuple(max_coords + 1)
            voxel = np.zeros(shape, dtype=bool)
            voxel[tuple(self.coords.T)] = True
            self._voxel = voxel
        return self._voxel
    
    @property
    def pc(self) -> ndarray:
        return self.origin + (self.coords + 0.5) * self.voxel_size
    
    def projection_fill(self, rigid: bool=True):
        """
        Fill up holes in the voxel.
        """
        grids = np.indices(self.voxel.shape)
        x_coord = grids[0, ...]
        y_coord = grids[1, ...]
        z_coord = grids[2, ...]
        
        INF = 2147483647
        x_tmp = x_coord.copy()
        x_tmp[~self.voxel] = INF
        x_min = x_tmp.min(axis=0)
        
        x_tmp[~self.voxel] = -1
        x_max = x_tmp.max(axis=0)
        
        y_tmp = y_coord.copy()
        y_tmp[~self.voxel] = INF
        y_min = y_tmp.min(axis=1)
        
        y_tmp[~self.voxel] = -1
        y_max = y_tmp.max(axis=1)
        
        z_tmp = z_coord.copy()
        z_tmp[~self.voxel] = INF
        z_min = z_tmp.min(axis=2)
        z_tmp[~self.voxel] = -1
        z_max = z_tmp.max(axis=2)
        
        in_x = (x_coord >= x_min[None, :, :]) & (x_coord <= x_max[None, :, :])
        in_y = (y_coord >= y_min[:, None, :]) & (y_coord <= y_max[:, None, :])
        in_z = (z_coord >= z_min[:, :, None]) & (z_coord <= z_max[:, :, None])
        
        count = in_x.astype(int) + in_y.astype(int) + in_z.astype(int)
        fill_mask = count >= (3 if rigid else 2)
        self._voxel = self.voxel | fill_mask
        x, y, z = np.where(self.voxel)
        self.coords = np.stack([x, y, z], axis=1)
    
    def inside(self, points: ndarray) -> ndarray:
        if points.ndim == 1:
            points = points[None, :]
        points = np.asarray(points)
        idx = np.floor((points - self.origin) / self.voxel_size).astype(int)
        invalid = (
            (idx < 0).any(axis=1) |
            (idx >= np.array(self.voxel.shape)).any(axis=1)
        )
        result = np.zeros(len(points), dtype=bool)
        valid_idx = np.where(~invalid)[0]
        valid_voxel_idx = idx[valid_idx]
        result[valid_idx] = self.voxel[valid_voxel_idx[:, 0], valid_voxel_idx[:, 1], valid_voxel_idx[:, 2]]
        return result