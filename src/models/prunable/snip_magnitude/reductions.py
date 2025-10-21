import torch
import torch.nn as nn

from typing import List

from src.utils.models import block_num_heads


def snip_reduce_mlp_pruning_weights(model: nn.Module, alpha: float = 0.001) -> List[torch.Tensor]:
    """
        Produce an [E] tensor for each MLP block according to the
        SNIP-magnitude pruning criterion as described in:

        https://openreview.net/forum?id=shw9MeqlMF
    """
    reduced = []

    for block in model.blocks:
        reduced.append(
            ((block.mlp.fc1.weight.grad * block.mlp.fc1.weight).abs() + alpha * (block.mlp.fc1.weight ** 2)).mean(dim=1)
        )

    return reduced

def snip_reduce_heads_pruning_weights(model: nn.Module, head_dim: int, embeddings_dim: int, alpha: float = 0.001) -> List[torch.Tensor]:
    """
        Produce a [H] tensor for each attention block according to the
        SNIP-magnitude pruning criterion as described in:

        https://openreview.net/forum?id=shw9MeqlMF
    """
    reduced = []

    for block in model.blocks:
        num_heads = block_num_heads(block=block)

        # Compute the SNIP-magnitude pruning criterion
        head_grads = (
            (block.attn.qkv.weight.grad * block.attn.qkv.weight).abs() + alpha * (block.attn.qkv.weight ** 2)
        )

        # Reduce the pruning weights to an [H] tensor
        head_grads = head_grads.reshape(
            3,
            num_heads,
            head_dim,
            embeddings_dim
        )[2, :, :, :] ** 2

        reduced.append(head_grads.mean(dim=(1, 2)))

    return reduced
