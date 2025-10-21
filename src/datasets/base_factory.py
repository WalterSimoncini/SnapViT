import torch.nn as nn

from abc import ABC, abstractmethod
from .enums import DatasetSplit


class DatasetFactory(ABC):
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    @abstractmethod
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        """Returns a torch Dataset for the given split"""
        raise NotImplementedError
