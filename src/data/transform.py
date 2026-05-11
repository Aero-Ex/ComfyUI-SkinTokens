from dataclasses import dataclass
from typing import List, Optional

from ..rig_package.info.asset import Asset
from .augment import Augment, get_augments
from .order import Order
from .sampler import Sampler, get_sampler
from .spec import ConfigSpec
from .vertex_group import VertexGroup, get_vertex_groups

@dataclass
class Transform(ConfigSpec):
    
    order: Optional[Order]=None
    
    vertex_groups: Optional[List[VertexGroup]]=None
    
    augments: Optional[List[Augment]]=None
    
    sampler: Optional[Sampler]=None
    
    @classmethod
    def parse(cls, **kwargs) -> 'Transform':
        cls.check_keys(kwargs)
        order_config = kwargs.get('order')
        vertex_groups_config = kwargs.get('vertex_groups')
        augments_config = kwargs.get('augments')
        sampler_config = kwargs.get('sampler')
        
        d = {}
        if order_config is not None:
            d['order'] = Order.parse(**order_config)
        if vertex_groups_config is not None:
            d['vertex_groups'] = get_vertex_groups(*vertex_groups_config)
        if augments_config is not None:
            d['augments'] = get_augments(*augments_config)
        if sampler_config is not None:
            d['sampler'] = get_sampler(**sampler_config)
        return Transform(**d)
    
    def apply(self, asset: Asset, **kwargs):
        
        # 1. arrange bones
        if self.order is not None:
            if asset.joint_names is not None and asset.parents is not None:
                new_names, _ = self.order.arrange_names(cls=asset.cls, names=asset.joint_names, parents=asset.parents.tolist())
                asset.set_order(new_orders=new_names) # type: ignore
        
        # 2. collapse must perform first
        if self.augments is not None:
            kwargs = {}
            for augment in self.augments:
                augment.transform(asset=asset, **kwargs)
        
        # 3. get vertex groups
        if self.vertex_groups is not None:
            d = {}
            for v in self.vertex_groups:
                d.update(v.get_vertex_group(asset=asset))
            asset.vertex_groups = d
        else:
            asset.vertex_groups = {}
        
        # 4. sample
        if self.sampler is not None:
            res = self.sampler.sample(asset=asset)
            asset.sampled_vertices          = res.sampled_vertices
            asset.sampled_normals           = res.sampled_normals
            asset.sampled_vertex_groups     = res.sampled_vertex_groups
            asset.skin_samples              = res.skin_samples