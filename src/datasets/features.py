import torch

from typing import Tuple
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.features[idx], self.targets[idx]

    @property
    def num_classes(self) -> int:
        return torch.unique(self.targets).numel()

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    def merge(self, dataset: "FeaturesDataset") -> "FeaturesDataset":
        """Merge this dataset with the given one and return a new FeaturesDataset."""
        merged_features = torch.cat([self.features, dataset.features])
        merged_targets = torch.cat([self.targets, dataset.targets])

        return FeaturesDataset(
            features=merged_features,
            targets=merged_targets
        )
