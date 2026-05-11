
from numpy import ndarray
from typing import Optional, Tuple

import numpy as np
import scipy

def assert_ndarray(arr, name: str="arr", shape: Optional[Tuple[int, ...]]=None, dtype=None):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray or None, got {type(arr)}")
    if shape is not None:
        # shape may contain None as wildcard
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}")
        for i, (exp, actual) in enumerate(zip(shape, arr.shape)):
            if exp > 0 and exp != actual:
                raise ValueError(f"{name} shape mismatch at axis {i}: expected {exp}, got {actual}")
    if dtype is not None:
        if not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")

def assert_list(arr, name: str="arr", dtype=None):
    if not isinstance(arr, list):
        raise ValueError(f"found type {type(arr)}, expect a list")
    if dtype is not None:
        for x in arr:
            if not isinstance(x, dtype):
                raise ValueError(f"found type {type(x)} in {name}, expect all to be {dtype}")

def linear_blend_skinning(
    vertices: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=1,
    value: float=1.0,
) -> ndarray:
    """
    Args:
        vertices: (N, 4-pad)
        matrix_local: (J, 4, 4)
        matrix: (J, 4, 4)
        skin: (N, J)
        pad: 0 or 1
        value: value to pad
    Returns:
        (N, 3) vertices using LBS algorithm: Skinning with dual quaternions, Kavan, 2007
    """
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    assert_ndarray(vertices, name='vertices', shape=(N, 3))
    assert_ndarray(matrix_local, name="matrix_local", shape=(J, 4, 4))
    assert_ndarray(matrix, name="matrix", shape=(J, 4, 4))
    assert_ndarray(skin, name="skin", shape=(N, J))
    assert vertices.shape[-1] + pad == 4
    # (4, N)
    padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    # (J, 4, 4)
    trans = matrix @ np.linalg.inv(matrix_local)
    weighted_per_bone_matrix = []
    # (J, N)
    mask = (skin > 0).T
    for i in range(J):
        offset = np.zeros((4, N), dtype=np.float32)
        offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
        weighted_per_bone_matrix.append(offset)
    weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
    g = np.sum(weighted_per_bone_matrix, axis=0)
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T

def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """
    Turn axis angle representation to matrix representation.
    """
    res = np.pad(scipy.spatial.transform.Rotation.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray]=None,
) -> Tuple[ndarray, ndarray, ndarray]:
    '''
    Randomly pick samples proportional to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    
    Args:
        mask: (num_faces,), only sample points on the faces where value is True.
    Return:
        vertex_samples: sampled vertices
        
        original_face_index: on which face is sampled
        
        random_lengths: sampled vectors on face
    '''
    original_face_indices = np.arange(len(faces))
    # sample according to mask
    if mask is not None:
        assert_ndarray(arr=mask, name="mask", shape=(faces.shape[0],))
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]

    # get face area
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    # TODO: change to correct uniform sampling...
    face_weight = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # face_weight = np.cross(offset_0, offset_1, axis=-1)
    # face_weight = (face_weight * face_weight).sum(axis=1)
    
    # weight_cum = np.cumsum(face_weight, axis=0)
    # face_pick = np.random.rand(num_samples) * weight_cum[-1]
    # face_index = np.searchsorted(weight_cum, face_pick)
    
    # map face_index back to original indices
    original_face_index = original_face_indices[face_index]
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, random_lengths

def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]]
    v_vectors -= v_origins[:, np.newaxis, :]
    
    sample_vector = (v_vectors * random_lengths).sum(axis=1)
    v_samples = sample_vector + v_origins
    return v_samples

def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int]=None,
    vertex_normals: Optional[ndarray]=None,
    face_normals: Optional[ndarray]=None,
    vertex_groups: Optional[ndarray]=None,
    face_mask: Optional[ndarray]=None,
    shuffle: bool=True,
    same: bool=False,
) -> Tuple[ndarray, ndarray|None, ndarray|None]:
    """
    Choose num_samples samples on the mesh and get their positions and normals.
    If vertex_group is provided, get its weights using barycentric sampling.
    
    Return:
        sampled_vertices, sampled_normals, sampled_vertex_groups
    
    Args:
        vertices: (N, 3)
        
        faces: (F, 3)
        
        num_samples: how many samples
        
        num_vertex_samples:
            At most num_vertex_samples unique vertices to be included,
            these points will be concatenated in the last (if shuffle is False).
        
        vertex_normals: (N, 3), sampled_normals will be None if not provided
        
        face_normals: (N, 3), sampled_normals will be None if not provided
        
        vertex_groups: (N, m), sampled_vertex_groups will be None if not provided
        
        face_mask:
            (F,) or (F, m), if shape is (F,), use the same mask across all
            vertex groups. Only sample on faces where value is True.
        
        shuffle: shuffle samples in the end
        
        same:
            Sample on the same locations, only useful when using mutiple
            vertex groups and mask is None or shape of (F,).
    """
    
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples cannot be larger than num_samples, found: {num_vertex_samples} > {num_samples}")
    
    def get_mask_perm(mask: Optional[ndarray]):
        if mask is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask)
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:num_vertex_samples]]
    
    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0],))
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0], -1))
            vertex_groups = vertex_groups
    
    if vertex_groups is not None:
        if face_mask is not None:
            if face_mask.ndim == 1:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            else:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0], vertex_groups.shape[1]))
        list_sampled_vertices = []
        list_sampled_normals = []
        list_sampled_vertex_groups = []
        perm = None
        _mask = None
        same = same and (face_mask is None or (face_mask is not None and face_mask.ndim != 2))
        for i in range(vertex_groups.shape[1]):
            if face_mask is not None:
                if face_mask.ndim == 1:
                    perm = get_mask_perm(faces[face_mask])
                    _mask = face_mask
                else:
                    perm = get_mask_perm(faces[face_mask[:, i]])
                    _mask = face_mask[:, i]
            else:
                perm = get_mask_perm(None)
                _mask = None
            _num_samples = num_samples - len(perm)
            
            face_vertices, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
            )
            
            list_sampled_vertices.append(np.concatenate([vertices[perm], face_vertices], axis=0))
            if vertex_normals is not None and face_normals is not None:
                list_sampled_normals.append(np.concatenate([vertex_normals[perm], face_normals[face_index]], axis=0))
            
            if same:
                g = sample_barycentric(
                    vertex_group=vertex_groups,
                    faces=faces,
                    face_index=face_index,
                    random_lengths=random_lengths,
                )
                list_sampled_vertex_groups.append(np.concatenate([vertex_groups[perm], g], axis=0))
                break
            g = sample_barycentric(
                vertex_group=vertex_groups[:, i:i+1],
                faces=faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )[:, 0]
            list_sampled_vertex_groups.append(np.concatenate([vertex_groups[:, i][perm], g], axis=0))
        sampled_vertices = np.stack(list_sampled_vertices, axis=1)
        if len(list_sampled_normals) > 0:
            sampled_normals = np.stack(list_sampled_normals, axis=1)
        else:
            sampled_normals = None
        if same:
            sampled_vertex_groups = list_sampled_vertex_groups[0]
        else:
            sampled_vertex_groups = np.stack(list_sampled_vertex_groups, axis=1)
    else: # otherwise only sample vertices and normals
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
            perm = get_mask_perm(faces[face_mask])
        else:
            perm = get_mask_perm(None)
        num_samples -= len(perm)
        n_vertex = vertices[perm]
        face_vertices, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=vertices,
            faces=faces,
            mask=face_mask,
        )
        sampled_vertices = np.concatenate([n_vertex, face_vertices], axis=0)
        if vertex_normals is not None and face_normals is not None:
            sampled_normals = np.concatenate([vertex_normals[perm], face_normals[face_index]], axis=0)
        else:
            sampled_normals = None
        sampled_vertex_groups = None
    
    return sampled_vertices, sampled_normals, sampled_vertex_groups