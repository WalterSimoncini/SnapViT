import os
import torch.nn as nn

from .enums import DatasetSplit
from torchvision.datasets import CIFAR10

from .base_factory import DatasetFactory


class CIFAR10DatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "CIFAR10 has no validation split"

        return CIFAR10(
            root=os.path.join(self.cache_dir, "cifar10"),
            train=split == DatasetSplit.TRAIN,
            download=True,
            transform=transform
        )
