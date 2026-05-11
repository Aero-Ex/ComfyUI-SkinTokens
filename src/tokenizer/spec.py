from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
from numpy import ndarray

from typing import Union, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TokenizeInput():
    # (J, 3)
    joints: ndarray
    
    # (J)
    parents: List[Union[None, int]]
    
    # string of class in tokenizer
    cls: Optional[str]=None
    
    joint_names: Optional[List[str]]=None
    
    @property
    def J(self) -> int:
        return self.joints.shape[0]
    
    @property
    def branch(self) -> ndarray:
        if not hasattr(self, '_branch'):
            branch = []
            last = None
            for i in range(self.J):
                if i == 0:
                    branch.append(False)
                else:
                    pid = self.parents[i]
                    branch.append(pid!=last)
                last = i
            self._branch = np.array(branch, dtype=bool)
        return self._branch
    
    @property
    def bones(self):
        _p = self.parents.copy()
        _p[0] = 0
        return np.concatenate([self.joints[_p], self.joints], axis=1)
    
    @property
    def num_bones(self):
        return self.bones.shape[0]

@dataclass
class DetokenizeOutput():
    # original tokens
    tokens: ndarray

    # (J, 6), (parent position, position)
    bones: ndarray
    
    # (J), parent of each bone
    parents: List[int]
    
    # string of class in tokenizer
    cls: Optional[str]=None
    
    # names of joints
    joint_names: Optional[List[str]]=None
    
    continuous_range: Optional[Tuple[float, float]]=None
    
    @property
    def joints(self):
        return self.bones[:, 3:]
    
    @property
    def p_joints(self):
        return self.bones[:, :3]
    
    @property
    def num_bones(self):
        return self.bones.shape[0]    
    
    @property
    def J(self):
        return self.bones.shape[0]
    
    def _get_parents(self) -> List[int]:
        parents = []
        for (i, bone) in enumerate(self.bones):
            p_joint = bone[:3]
            dis = 999999
            pid = -1
            for j in reversed(range(i)):
                n_dis = ((self.bones[j][3:] - p_joint)**2).sum()
                if n_dis < dis:
                    pid = j
                    dis = n_dis
            parents.append(pid)
        return parents

class Tokenizer(ABC):
    """
    Abstract class for tokenizer
    """
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Tokenizer':
        pass
    
    @abstractmethod
    def tokenize(self, input: TokenizeInput) -> ndarray:
        pass
    
    @abstractmethod
    def detokenize(self, ids: ndarray, **kwargs) -> DetokenizeOutput:
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """The vocabulary size"""
        raise NotImplementedError()
    
    @property
    def pad(self):
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))
    
    @property
    def bos(self):
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))
    
    def cls_name_to_token(self, cls: str) -> int:
        raise NotImplementedError()
    
    def next_posible_token(self, ids: ndarray) -> List[int]:
        raise NotImplementedError()
    
    def bones_in_sequence(self, ids: ndarray) -> int:
        raise NotImplementedError()
    
    def make_cls_head(self, **kwargs) -> List[int]:
        raise NotImplementedError()

def make_skeleton(
    joints: ndarray,
    p_joints: ndarray,
    tails_dict: Dict[int, ndarray],
    convert_leaf_bones_to_tails: bool,
    extrude_tail_for_leaf: bool,
    extrude_tail_for_branch: bool,
    extrude_scale: float=0.5,
    strict: bool=False,
) -> Tuple[ndarray, ndarray, List[int], List[int]]:
    '''
    Args:
        joints: heads of bones
        
        p_joints: parent position of joints
        
        tails_dict: tail position of the i-th joint
        
        convert_leaf_bones_to_tails: remove leaf bones and make them tails of their parents
        
        extrude_tail_for_leaf: add a tail for leaf bone
        
        extrude_tail_for_branch: add a tail for joint with multiple children
        
        extrude_scale: length scale of tail offset
        
        strict: if true, raise error when there are joints in the same location
        
    Returns:
        bones, tails, available_bones_id, parents
    '''
    assert (convert_leaf_bones_to_tails & extrude_tail_for_leaf)==False, 'cannot extrude tail for leaf when convert_leaf_bones_to_tails is True'
    assert joints.shape[0] == p_joints.shape[0]
    # build parents
    bones = [] # (parent_position, position)
    parents = []
    for (i, joint) in enumerate(joints):
        if len(bones) == 0:
            bones.append(np.concatenate([joint, joint])) # root
            parents.append(-1)
            continue
        p_joint = p_joints[i]
        dis = 999999
        pid = None
        for j in reversed(range(i)):
            n_dis = ((bones[j][3:] - p_joint)**2).sum()
            if n_dis < dis:
                pid = j
                dis = n_dis
        bones.append(np.concatenate([joints[pid], joint]))
        parents.append(pid)
    bones = np.stack(bones)
    
    children = defaultdict(list)
    for (i, pid) in enumerate(parents):
        if pid == -1:
            continue
        children[pid].append(i)
    
    available_bones_id = []
    if convert_leaf_bones_to_tails:
        for (i, pid) in enumerate(parents):
            if len(children[i]) != 0:
                available_bones_id.append(i)
                continue
            tails_dict[pid] = bones[i, 3:]
    else:
        available_bones_id = [i for i in range(bones.shape[0])]
    
    # tail for leaf
    for (i, pid) in enumerate(parents):
        if len(children[i]) != 0:
            continue
        if extrude_tail_for_leaf:
            d = bones[i, 3:] - bones[pid, 3:]
            length = np.linalg.norm(d)
            if strict:
                assert length > 1e-9, 'two joints in the same point found'
            elif length <= 1e-9:
                d = np.array([0., 0., 1.])
            tails_dict[i] = bones[i, 3:] + d * extrude_scale
        else:
            tails_dict[i] = bones[i, 3:]
    
    # tail for branch
    for (i, pid) in enumerate(parents):
        if len(children[i]) <= 1:
            continue
        if extrude_tail_for_branch:
            if pid == -1: # root
                av_len = 0
                for child in children[i]:
                    av_len += np.linalg.norm(bones[i, 3:] - bones[child, 3:])
                av_len /= len(children[i])
                d = bones[i, 3:] + np.array([0., 0., extrude_scale * av_len])
            else:
                d = bones[i, 3:] - bones[pid, 3:]
                length = np.linalg.norm(d)
                if strict:
                    assert length > 1e-9, 'two joints in the same point found'
                elif length <= 1e-9:
                    d = np.array([0., 0., 1.])
            tails_dict[i] = bones[i, 3:] + d * extrude_scale
        else:
            tails_dict[i] = bones[i, 3:]
    
    # assign new tail
    for (i, pid) in enumerate(parents):
        if len(children[i]) != 1:
            continue
        child = children[i][0]
        tails_dict[i] = bones[child, 3:]
    
    tails = []
    for i in range(bones.shape[0]):
        tails.append(tails_dict[i])
    tails = np.stack(tails)
    return bones, tails, available_bones_id, parents