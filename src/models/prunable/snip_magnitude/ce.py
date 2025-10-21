import torch
import torch.nn as nn

from typing import List

from src.models.prunable.base import PrunableModel

from .reductions import snip_reduce_mlp_pruning_weights, snip_reduce_heads_pruning_weights


class CESNIPMagnitudePrunableModel(PrunableModel):
    """SNIP Magnitude with cross-entropy gradients."""
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        min_hidden_dim_ratio: float = 0.2,
        min_head_ratio: float = 0.2,
        alpha: float = 0.001,
        **kwargs
    ):
        super().__init__(
            model=model,
            device=device,
            min_hidden_dim_ratio=min_hidden_dim_ratio,
            min_head_ratio=min_head_ratio
        )

        self.alpha = alpha

    def _reduce_pruning_weights(self) -> List[torch.Tensor]:
        return snip_reduce_mlp_pruning_weights(model=self.model, alpha=self.alpha)

    def _reduce_heads_pruning_weights(self) -> List[torch.Tensor]:
        return snip_reduce_heads_pruning_weights(
            model=self.model,
            head_dim=self.head_dim,
            embeddings_dim=self.embeddings_dim,
            alpha=self.alpha
        )
