
from collections import defaultdict
from dataclasses import dataclass
from numpy import ndarray
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Optional

from .spec import ConfigSpec

@dataclass
class Order(ConfigSpec):
    
    # {part_name: [bone_name_1, bone_name_2, ...]}
    parts: Dict[str, Dict[str, List[str]]]
    
    # parts of bones to be arranged in [part_name_1, part_name_2, ...]
    parts_order: Dict[str, List[str]]
    
    # {skeleton_name: path}
    skeleton_path: Optional[Dict[str, str]]=None
    
    sort_by_xyz: bool=False
    
    @classmethod
    def parse(cls, **kwargs) -> 'Order':
        cls.check_keys(kwargs)
        skeleton_path = kwargs.get('skeleton_path', None)
        if skeleton_path is not None:
            parts = {}
            parts_order = {}
            for (cls, path) in skeleton_path.items():
                assert cls not in parts, 'cls conflicts'
                import os
                if not os.path.isabs(path):
                    skintokens_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    path = os.path.join(skintokens_dir, path)
                d = OmegaConf.load(path)
                parts[cls] = d.parts
                parts_order[cls] = d.parts_order
        else:
            parts = kwargs.get('parts')
            parts_order = kwargs.get('parts_order')
            assert parts is not None
            assert parts_order is not None
        return Order(
            skeleton_path=skeleton_path,
            parts=parts,
            parts_order=parts_order,
            sort_by_xyz=kwargs.get('sort_by_xyz', False),
        )
    
    def part_exists(self, cls: str, part: str, names: List[str]) -> bool:
        '''
        Check if part exists.
        '''
        if part not in self.parts[cls]:
            return False
        for name in self.parts[cls][part]:
            if name not in names:
                return False
        return True
    
    def make_names(self, cls: str|None, parts: List[str|None], num_bones: int) -> List[str]:
        '''
        Get names for specified cls.
        '''
        names = []
        for part in parts:
            if part is None: # spring
                continue
            if cls in self.parts and part in self.parts[cls]:
                names.extend(self.parts[cls][part])
        assert len(names) <= num_bones, "number of bones in required skeleton is more than existing bones"
        for i in range(len(names), num_bones):
            names.append(f"bone_{i}")
        return names
    
    def arrange_names(self, cls: str|None, names: List[str], parents: List[int], joints: Optional[ndarray]=None) -> Tuple[List[str], Dict[int, str|None]]:
        '''
        Arrange names according to required parts order.
        '''
        def sort_by_xyz(joints):
            return sorted(joints, key=lambda joint: (joint[1][2], joint[1][0], joint[1][1]))
        
        if self.sort_by_xyz:
            assert joints is not None
            new_names = []
            root = -1
            son = defaultdict(list)
            not_root = {}
            for (i, p) in enumerate(parents):
                if p != -1:
                    son[p].append(i)
                    not_root[i] = True
            for i in range(len(parents)):
                if not_root.get(i, False) == False:
                    root = i
                    break
            Q = [root]
            while Q:
                u = Q.pop(0)
                new_names.append(names[u])
                wait = []
                for v in son[u]:
                    wait.append((v, joints[v]))
                wait_sorted = sort_by_xyz(wait)
                new_wait = [v for v, _ in wait_sorted]
                Q = new_wait + Q
            return new_names, {}
        if cls not in self.parts_order:
            return names, {0: None} # add a spring token
        vis = defaultdict(bool)
        name_to_id = {name: i for (i, name) in enumerate(names)}
        new_names = []
        parts_bias = {}
        for part in self.parts_order[cls]:
            if self.part_exists(cls=cls, part=part, names=names):
                for name in self.parts[cls][part]:
                    vis[name] = True
                flag = False
                for name in self.parts[cls][part]:
                    pid = parents[name_to_id[name]]
                    if pid==-1:
                        continue
                    if not vis[names[pid]]:
                        flag = True
                        break
                if flag: # incorrect parts order and should immediately add a spring token
                    break
                parts_bias[len(new_names)] = part
                new_names.extend(self.parts[cls][part])
        parts_bias[len(new_names)] = None # add a spring token
        for name in names:
            if name not in new_names:
                new_names.append(name)
        return new_names, parts_bias