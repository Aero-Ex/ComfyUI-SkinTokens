"""Abstract class for parsers."""

from abc import ABC, abstractmethod

from ..info.asset import Asset

class AbstractParser(ABC):
    """Abstract class for parsers."""
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str, **kwargs) -> Asset:
        pass
    
    @classmethod
    def export(cls, asset: Asset, filepath: str, **kwargs):
        raise NotImplementedError("do not implement")