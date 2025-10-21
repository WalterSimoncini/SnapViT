import os
import torch.nn as nn

from PIL.Image import Image
from typing import Tuple, List
from datasets import load_from_disk
from torch.utils.data import Dataset

from .enums import DatasetSplit
from .base_factory import DatasetFactory


class OxfordIITPetsDatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        **kwargs
    ) -> nn.Module:
        return OxfordIITPets(
            split=split,
            transform=transform,
            cache_dir=self.cache_dir
        )


class OxfordIITPets(Dataset):
    """
        The Oxford-IIT Pets dataset. Follow the instructions in
        the README.md file to setup this dataset
    """
    def __init__(
        self,
        split: DatasetSplit,
        transform: nn.Module,
        cache_dir: str = None
    ) -> None:
        """
            Initializes an Oxford-IIT Pets dataset. The dataset is assumed to
            be in the cache_dir/pets-hf folder, which should contain an
            HuggingFace dataset generated using the tools/pets.py script.

            Performance of this dataset should be evaluated as the "average
            per-class classification accuracy" according to the original paper
        """
        assert split != DatasetSplit.VALID, "Oxford-IIT Pets has no validation split"

        self.transform = transform
        self.dataset = load_from_disk(
            os.path.join(cache_dir, "pets-hf")
        )[split.value]

    @property
    def targets(self) -> List[int]:
        return self.dataset["label"]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[Image, int]:
        example = self.dataset[int(idx)]
        image = example["image"]

        if image.mode != "RGB":
            # Convert grayscale images to RGB
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, example["label"]
