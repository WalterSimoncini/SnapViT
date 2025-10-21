import torch
import torch.nn as nn

from typing import List

from .base import PrunableModel
from src.utils.models import block_num_heads


class SNIPMagnitudePrunableModel(PrunableModel):
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
        """
            Produce an [E] tensor for each MLP block according to the
            SNIP-magnitude pruning criterion as described in:

            https://openreview.net/forum?id=shw9MeqlMF
        """
        reduced = []

        for block in self.model.blocks:
            reduced.append(
                ((block.mlp.fc1.weight.grad * block.mlp.fc1.weight).abs() + self.alpha * (block.mlp.fc1.weight ** 2)).mean(dim=1)
            )

        return reduced

    def _reduce_heads_pruning_weights(self) -> List[torch.Tensor]:
        """
            Produce a [H] tensor for each attention block according to the
            SNIP-magnitude pruning criterion as described in:

            https://openreview.net/forum?id=shw9MeqlMF
        """
        reduced = []

        for block in self.model.blocks:
            num_heads = block_num_heads(block=block)

            # Compute the SNIP-magnitude pruning criterion
            head_grads = (
                (block.attn.qkv.weight.grad * block.attn.qkv.weight).abs() + self.alpha * (block.attn.qkv.weight ** 2)
            )

            # Reduce the pruning weights to an [H] tensor
            head_grads = head_grads.reshape(
                3,
                num_heads,
                self.head_dim,
                self.embeddings_dim
            )[2, :, :, :] ** 2

            reduced.append(head_grads.mean(dim=(1, 2)))

        return reduced
