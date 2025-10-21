import os
import logging
import torch.nn as nn

from typing import Tuple
from PIL.Image import Image
from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset

from .base_factory import DatasetFactory
from .enums import DatasetSplit, DatasetType


class HuggingFaceDatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        dataset_type: DatasetType = None,
        keep_in_memory: bool = False,
        **kwargs
    ) -> nn.Module:
        dataset_name = {
            DatasetType.DIFFUSION_DB_LARGE_FIRST_10K: "poloclub/diffusiondb",
            DatasetType.MERGED: "merged"
        }[dataset_type]

        return HuggingFaceDataset(
            split=split,
            transform=transform,
            cache_dir=self.cache_dir,
            dataset_name=dataset_name,
            variant=self.__dataset_variant(type_=dataset_type),
            from_disk=self.__is_dataset_on_disk(type_=dataset_type),
            keep_in_memory=keep_in_memory
        )

    def __dataset_variant(self, type_: DatasetType) -> str:
        return {
            DatasetType.DIFFUSION_DB_LARGE_FIRST_10K: "large_first_10k",
        }.get(type_)

    def __is_dataset_on_disk(self, type_: DatasetType) -> bool:
        return type_ not in [
            DatasetType.DIFFUSION_DB_LARGE_FIRST_10K
        ]


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        split: DatasetSplit,
        transform: nn.Module,
        cache_dir: str = None,
        dataset_name: str = None,
        from_disk: bool = True,
        variant: str = None,
        keep_in_memory: bool = False
    ) -> None:
        """
            Initializes a HuggingFace dataset from the hub or disk. If the dataset
            is on disk, it is assumed to be in the cache_dir/dataset_name folder.
            The dataset should have two columns:

            - image: a PIL.Image instance, either in RGB or grayscale
            - label: an integer class label. If this column does not exist,
                     the label will always be 0.

            Args:
                split: the dataset split to loaded. Should be None if the dataset
                    has no splits.
                transform: a torchvision transform to apply to images.
                cache_dir: the read/write directory to store datasets.
                dataset_name: the name of the dataset on the HuggingFace hub or
                    the name of the directory containing the dataset on disk.
                from_disk: whether the dataset should be loaded from disk
                variant: the variant of the dataset to load. Only used for
                    HuggingFace hub datasets (with variants).
                keep_in_memory: whether the dataset should be loaded in memory.
        """
        self.transform = transform
        self.keep_in_memory = keep_in_memory

        if from_disk:
            dataset_path = os.path.join(cache_dir, dataset_name)

            logging.info(f"loading on-disk huggingface dataset from {dataset_path}")

            self.dataset = load_from_disk(
                dataset_path=dataset_path,
                keep_in_memory=self.keep_in_memory
            )
        else:
            logging.info(f"loading huggingface dataset {dataset_name}")

            base_dataset_args = {
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": cache_dir
            }

            if variant:
                self.dataset = load_dataset(dataset_name, variant, **base_dataset_args)
            else:
                self.dataset = load_dataset(dataset_name, **base_dataset_args)

        # Select the dataset split if one was specified
        if split is not None:
            self.dataset = self.dataset[split.value]

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

        return image, example.get("label", 0)
