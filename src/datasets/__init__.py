import os
import torch
import logging
import torch.nn as nn

from typing import List, Dict
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from .enums import DatasetSplit, DatasetType

from .in1k import IN1KDatasetFactory
from .cifar10 import CIFAR10DatasetFactory
from .eurosat import EuroSATDatasetFactory
from .cifar100 import CIFAR100DatasetFactory
from .textures import TexturesDatasetFactory
from .pets import OxfordIITPetsDatasetFactory
from .shaders import Shaders21KDatasetFactory
from .aircraft import FGVCAircraftDatasetFactory
from .huggingface import HuggingFaceDatasetFactory


def load_dataset(
    type_: DatasetType,
    split: DatasetSplit,
    cache_dir: str,
    transform: nn.Module = None,
    max_samples: int = None,
    **kwargs
) -> nn.Module:
    """Returns a torch Dataset of the given type and split"""
    # Create the cache directory if it does not exists already
    os.makedirs(cache_dir, exist_ok=True)

    factory = {
        DatasetType.IN1K: IN1KDatasetFactory,
        DatasetType.SHADERS_21K: Shaders21KDatasetFactory,
        DatasetType.TEXTURES: TexturesDatasetFactory,
        DatasetType.FGVC_AIRCRAFT: FGVCAircraftDatasetFactory,
        DatasetType.OXFORD_IIT_PETS: OxfordIITPetsDatasetFactory,
        DatasetType.DIFFUSION_DB_LARGE_FIRST_10K: HuggingFaceDatasetFactory,
        DatasetType.EUROSAT: EuroSATDatasetFactory,
        DatasetType.CIFAR10: CIFAR10DatasetFactory,
        DatasetType.CIFAR100: CIFAR100DatasetFactory,
        DatasetType.MERGED: HuggingFaceDatasetFactory
    }[type_](cache_dir=cache_dir)

    dataset = factory.load(
        transform=transform,
        split=split,
        dataset_type=type_,
        **kwargs
    )

    if max_samples is not None and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples]

        return Subset(dataset=dataset, indices=indices)

    return dataset


def load_eval_datasets(
    dataset_types: List[DatasetType],
    cache_dir: str,
    transform: nn.Module = None,
    batch_size: int = 16,
    num_workers: int = 18,
    max_knn_train_samples: int = None,
) -> List[Dict]:
    datasets = []

    for type_ in dataset_types:
        dataset = {
            "type": type_,
            "name": type_.value.replace("-", "_"),
            "train": DataLoader(load_dataset(
                type_=type_,
                split=DatasetSplit.TRAIN,
                max_samples=max_knn_train_samples,
                cache_dir=cache_dir,
                transform=transform
            ), batch_size=batch_size, num_workers=num_workers, pin_memory=True),
            "test": DataLoader(load_dataset(
                type_=type_,
                split=DatasetSplit.TEST,
                cache_dir=cache_dir,
                transform=transform
            ), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        }

        # Try to load the validation dataset if possible
        try:
            validation_dataset = load_dataset(
                type_=type_,
                split=DatasetSplit.VALID,
                cache_dir=cache_dir,
                transform=transform
            )

            dataset["valid"] = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True
            )
        except Exception as e:
            logging.warning(f"the validation dataset for {type_} could not be loaded: {e}")

        datasets.append(dataset)

    return datasets
