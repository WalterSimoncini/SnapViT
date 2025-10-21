import torch

from typing import List

from .base import PrunableModel
from src.utils.models import block_num_heads


class LAMPPrunableModel(PrunableModel):
    """
        LAMP-score based pruning as described in:

        Layer-adaptive sparsity for the  magnitude-based pruning by Lee et al.
    """
    def _reduce_pruning_weights(self) -> List[torch.Tensor]:
        """
            Use the L2 norm of the weights across rows to produce a [E]
            tensor for each block.
        """
        reduced = []

        for i, block in enumerate(self.model.blocks):
            scores = self.__lamp_score(block.mlp.fc1.weight.data)

            # Average over the input dimension to have a single score per neuron
            reduced.append(scores.mean(dim=-1))

        return reduced

    def _reduce_heads_pruning_weights(self) -> List[torch.Tensor]:
        reduced = []

        for i, block in enumerate(self.model.blocks):
            num_heads = block_num_heads(block=block)

            scores = self.__lamp_score(
                block.attn.qkv.weight.data.chunk(3, dim=0)[2]
            )

            # Average over the input dimension and then over the heads
            # to have a single score per head
            reduced.append(
                scores.mean(dim=1).reshape(num_heads, -1).mean(dim=1)
            )

        return reduced

    def __lamp_score(self, weights: torch.Tensor) -> torch.Tensor:
        """
            Compute the LAMP score for a given weight matrix.
        """
        normalizer = weights.norm() ** 2

        # Sort the weights in ascending order
        sorted, indices = weights.abs().view(-1).sort(descending=False)

        # Calculate the cumulative weight sums and normalize the weight using them
        weights_cumsum = (sorted ** 2).cumsum(dim=0).roll(shifts=1)
        weights_cumsum[0] = 0

        sorted /= (normalizer - weights_cumsum).sqrt()

        # Create a scores tensor with the original weights order
        scores = torch.zeros_like(sorted, device=self.device)
        scores[indices] = sorted

        return scores.reshape(weights.shape)
