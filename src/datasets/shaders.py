import os
import torch.nn as nn

from torchvision.datasets import ImageFolder

from .enums import DatasetSplit
from .base_factory import DatasetFactory


class Shaders21KDatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        **kwargs
    ) -> nn.Module:
        assert split == DatasetSplit.TRAIN, "The shaders21k dataset has only a training split"

        return ImageFolder(
            root=os.path.join(self.cache_dir, "shaders21k"),
            transform=transform
        )
