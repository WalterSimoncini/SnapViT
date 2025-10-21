import os
import torch.nn as nn

from torchvision.datasets import DTD

from .enums import DatasetSplit
from .base_factory import DatasetFactory


class TexturesDatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        split_name = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALID: "val",
            DatasetSplit.TEST: "test"
        }[split]

        return DTD(
            root=os.path.join(self.cache_dir, "textures"),
            split=split_name,
            download=True,
            transform=transform
        )
