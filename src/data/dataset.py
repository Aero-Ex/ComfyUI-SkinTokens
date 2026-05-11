from copy import deepcopy
from dataclasses import dataclass
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from numpy import ndarray
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Callable, Optional

import os
import lightning.pytorch as pl
import numpy as np
import torch

from .datapath import Datapath, LazyAsset
from .spec import ConfigSpec
from .transform import Transform

from ..model.spec import ModelInput
from ..rig_package.info.asset import Asset
from ..tokenizer.spec import Tokenizer, TokenizeInput

@dataclass
class DatasetConfig(ConfigSpec):
    shuffle: bool
    batch_size: int
    num_workers: int
    datapath: Datapath
    pin_memory: bool=True
    persistent_workers: bool=True
    
    @classmethod
    def parse(cls, **kwargs) -> 'DatasetConfig':
        cls.check_keys(kwargs)
        return DatasetConfig(
            shuffle=kwargs.get('shuffle', False),
            batch_size=kwargs.get('batch_size', 1),
            num_workers=kwargs.get('num_workers', 1),
            pin_memory=kwargs.get('pin_memory', True),
            persistent_workers=kwargs.get('persistent_workers', True),
            datapath=Datapath.parse(**kwargs.get('datapath')),
        )
    
    def split_by_cls(self) -> Dict[str|None, 'DatasetConfig']:
        res: Dict[str|None, DatasetConfig] = {}
        datapath_dict = self.datapath.split_by_cls()
        for cls, v in datapath_dict.items():
            res[cls] = DatasetConfig(
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                datapath=v,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return res

class RigDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        process_fn: Optional[Callable[[List[ModelInput]], List[Dict]]]=None,
        train_dataset_config: Optional[DatasetConfig]=None,
        validate_dataset_config: Optional[Dict[str|None, DatasetConfig]]=None,
        predict_dataset_config: Optional[Dict[str|None, DatasetConfig]]=None,
        train_transform: Optional[Transform]=None,
        validate_transform: Optional[Transform]=None,
        predict_transform: Optional[Transform]=None,
        tokenizer: Optional[Tokenizer]=None,
        debug: bool=False,
    ):
        super().__init__()
        self.process_fn                 = process_fn
        self.train_dataset_config       = train_dataset_config
        self.validate_dataset_config    = validate_dataset_config
        self.predict_dataset_config     = predict_dataset_config
        self.train_transform            = train_transform
        self.validate_transform         = validate_transform
        self.predict_transform          = predict_transform
        self.tokenizer                  = tokenizer
        self.debug                      = debug
        
        if debug:
            print("\033[31mWARNING: debug mode, dataloader will be extremely slow !!!\033[0m")
        
        # build train datapath
        if self.train_dataset_config is not None:
            self.train_datapath = self.train_dataset_config.datapath
        else:
            self.train_datapath = None
        
        # build validate datapath
        if self.validate_dataset_config is not None:
            self.validate_datapath = {
                cls: self.validate_dataset_config[cls].datapath
                for cls in self.validate_dataset_config
            }
        else:
            self.validate_datapath = None
        
        # build predict datapath
        if self.predict_dataset_config is not None:
            self.predict_datapath = {
                cls: self.predict_dataset_config[cls].datapath
                for cls in self.predict_dataset_config
            }
        else:
            self.predict_datapath = None
        
        self.tokenizer = tokenizer

    def prepare_data(self):
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset_config is None:
            raise ValueError("do not have train_dataset_config")
        if self.train_transform is None:
            raise ValueError("do not have train_transform")
        if self.train_datapath is not None:
            self._train_ds = RigDataset(
                process_fn=self.process_fn,
                data=self.train_datapath.get_data(),
                name="train",
                tokenizer=self.tokenizer,
                transform=self.train_transform,
                debug=self.debug,
            )
        else:
            return None
        return self._create_dataloader(
            dataset=self._train_ds,
            config=self.train_dataset_config,
            is_train=True,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.validate_dataset_config is None:
            raise ValueError("do not have validate_dataset_config")
        if self.validate_transform is None:
            raise ValueError("do not have validate_transform")
        if self.validate_datapath is not None:
            self._validation_ds = {}
            for cls in self.validate_datapath:
                self._validation_ds[cls] = RigDataset(
                    process_fn=self.process_fn,
                    data=self.validate_datapath[cls].get_data(),
                    name=f"validate-{cls}",
                    tokenizer=self.tokenizer,
                    transform=self.validate_transform,
                    debug=self.debug,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._validation_ds,
            config=self.validate_dataset_config,
            is_train=False,
            drop_last=False,
        )
    
    def predict_dataloader(self):
        if self.predict_dataset_config is None:
            raise ValueError("do not have predict_dataset_config")
        if self.predict_transform is None:
            raise ValueError("do not have predict_transform")
        if self.predict_datapath is not None:
            self._predict_ds = {}
            for cls in self.predict_datapath:
                self._predict_ds[cls] = RigDataset(
                    process_fn=self.process_fn,
                    data=self.predict_datapath[cls].get_data(),
                    name=f"predict-{cls}",
                    tokenizer=self.tokenizer,
                    transform=self.predict_transform,
                    debug=self.debug,
                )
        else:
            return None
        return self._create_dataloader(
            dataset=self._predict_ds,
            config=self.predict_dataset_config,
            is_train=False,
            drop_last=False,
        )

    def _create_dataloader(
        self,
        dataset: Dataset|Dict[str, Dataset],
        config: DatasetConfig|Dict[str|None, DatasetConfig],
        is_train: bool,
        **kwargs,
    ) -> DataLoader|Dict[str, DataLoader]:
        def create_single_dataloader(dataset, config: DatasetConfig, **kwargs):
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers,
                collate_fn=dataset.collate_fn,
                **kwargs,
            )
        if isinstance(dataset, Dict):
            assert isinstance(config, dict)
            return {k: create_single_dataloader(v, config[k], **kwargs) for k, v in dataset.items()}
        else:
            assert isinstance(config, DatasetConfig)
            return create_single_dataloader(dataset, config, **kwargs)

class RigDataset(Dataset):
    def __init__(
        self,
        data: List[LazyAsset],
        transform: Transform,
        name: Optional[str]=None,
        process_fn: Optional[Callable[[List[ModelInput]], List[Dict]]]=None,
        tokenizer: Optional[Tokenizer]=None,
        debug: bool=False,
    ) -> None:
        super().__init__()
        
        self.data       = data
        self.name       = name
        self.process_fn = process_fn
        self.tokenizer  = tokenizer
        self.transform  = transform
        self.debug      = debug
        
        if not debug:
            assert self.process_fn is not None, 'missing data processing function'
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> ModelInput:
        lazy_asset = self.data[idx]
        asset = lazy_asset.load()
        self.transform.apply(asset=asset)
        if self.tokenizer is not None and asset.parents is not None:
            x = TokenizeInput(
                joints=asset.joints,
                parents=asset.parents,
                cls=asset.cls,
                joint_names=asset.joint_names,
            )
            tokens = self.tokenizer.tokenize(input=x)
        else:
            tokens = None
        return ModelInput(asset=asset, tokens=tokens)
    
    def _collate_fn_debug(self, batch):
        return batch
    
    def _collate_fn(self, batch):
        processed_batch = self.process_fn(batch) # type: ignore
        processed_batch: List[Dict]
        
        tensors_stack = {}
        tensors_cat = {}
        non_tensors = {}
        vis = {}
        def check(x):
            assert x not in vis, f"multiple keys found: {x}"
            vis[x] = True
        
        for k, v in processed_batch[0].items():
            if k == "cat":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    tensors_cat[k1] = []
                    for i in range(len(processed_batch)):
                        v1 = processed_batch[i]['cat'][k1]
                        if isinstance(v1, ndarray):
                            v1 = torch.from_numpy(v1)
                        elif isinstance(v1, Tensor):
                            v1 = v1
                        else:
                            raise ValueError(f"cannot concatenate non-tensor type of key {k1}, type: {type(v1)}")
                        tensors_cat[k1].append(v1)
            elif k == "non":
                assert isinstance(v, dict)
                for k1 in v.keys():
                    check(k1)
                    non_tensors[k1] = []
                    for i in range(len(processed_batch)):
                        v1 = processed_batch[i]['non'][k1]
                        if isinstance(v1, ndarray):
                            v1 = torch.from_numpy(v1)
                        non_tensors[k1].append(v1)
            else:
                check(k)
                tensors_stack[k] = []
                for i in range(len(processed_batch)):
                    v1 = processed_batch[i][k]
                    if isinstance(v1, ndarray):
                        v1 = torch.from_numpy(v1)
                    elif isinstance(v1, Tensor):
                        v1 = v1
                    else:
                        raise ValueError(f"cannot stack type of key {k}, type: {type(v1)}")
                    tensors_stack[k].append(v1)
        
        collated_stack = {k: torch.stack(v) for k, v in tensors_stack.items()}
        collated_cat = {k: torch.concat(v, dim=1) for k, v in tensors_cat.items()}
        
        collated_batch = {
            **collated_stack,
            **collated_cat,
            **non_tensors,
        }
        return collated_batch

    def collate_fn(self, batch):
        if self.debug:
            return self._collate_fn_debug(batch)
        return self._collate_fn(batch)