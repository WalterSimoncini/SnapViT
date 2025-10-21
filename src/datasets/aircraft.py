import os
import torch.nn as nn

from .enums import DatasetSplit
from torchvision.datasets import FGVCAircraft

from .base_factory import DatasetFactory


class FGVCAircraftDatasetFactory(DatasetFactory):
    """
        Factory class for the FGVC Aircraft dataset.

        The performance of this dataset should be evaluated on the "average
        per-class classification accuracy" according to the SimCLR paper
    """
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        split = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALID: "val",
            DatasetSplit.TEST: "test"
        }[split]

        return FGVCAircraft(
            root=os.path.join(self.cache_dir, "aircraft"),
            split=split,
            download=True,
            transform=transform
        )
