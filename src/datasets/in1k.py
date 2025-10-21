import os
import torch.nn as nn

from torchvision.datasets import ImageNet

from .enums import DatasetSplit
from .base_factory import DatasetFactory


class IN1KDatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        **kwargs
    ) -> nn.Module:
        assert split != DatasetSplit.VALID, "ImageNet-1K has no validation split (use test to get the validation one)"

        # The validation set of ImageNet is generally used as the test set
        split = {
            DatasetSplit.TEST: "val",
            DatasetSplit.TRAIN: "train"
        }[split]

        return ImageNet(
            root=os.path.join(self.cache_dir, "imagenet-1k"),
            split=split,
            transform=transform
        )
