import os
import torch.nn as nn

from .enums import DatasetSplit
from torchvision.datasets import CIFAR100

from .base_factory import DatasetFactory


class CIFAR100DatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "CIFAR100 has no validation split"

        return CIFAR100(
            root=os.path.join(self.cache_dir, "cifar100"),
            train=split == DatasetSplit.TRAIN,
            download=True,
            transform=transform
        )
