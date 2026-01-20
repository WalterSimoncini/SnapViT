import torch
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader

from .base import PrunableModel


class RandomPrunableModel(PrunableModel):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        min_hidden_dim_keep_ratio: float = 0.2,
        min_head_keep_ratio: float = 0.2,
        **kwargs
    ):
        super().__init__(
            model=model,
            device=device,
            min_hidden_dim_keep_ratio=min_hidden_dim_keep_ratio,
            min_head_keep_ratio=min_head_keep_ratio
        )

    def estimate_pruning_weights(self, data_loader: DataLoader):
        pass

    def _reduce_pruning_weights(self) -> List[torch.Tensor]:
        """Generate a random [E] tensor for each block."""
        return [
            torch.randn(block.mlp.fc1.weight.shape[0], device=self.device) for block in self.model.blocks
        ]

    def _reduce_heads_pruning_weights(self) -> List[torch.Tensor]:
        """Generate a random [H] tensor for each block."""
        return [
            torch.randn(block.attn.num_heads, device=self.device) for block in self.model.blocks
        ]
