import torch
import logging
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Optional, List, Tuple

from src.models.enums import MLPLayerType
from src.utils.models import predict, block_num_heads


class PrunableModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        min_hidden_dim_keep_ratio: float = 0.2,
        min_head_keep_ratio: float = 0.2,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.head_dim = model.embed_dim // model.blocks[0].attn.num_heads
        self.default_num_heads = model.blocks[0].attn.num_heads
        self.min_head_keep_count = int(self.default_num_heads * min_head_keep_ratio)
        self.min_head_keep_ratio = min_head_keep_ratio
        self.num_blocks = len(self.model.blocks)
        self.min_hidden_dim_keep_ratio = min_hidden_dim_keep_ratio

        if hasattr(self.model.blocks[0].mlp, MLPLayerType.FC1.value):
            self.target_input_mlp_layer = MLPLayerType.FC1
            self.target_output_mlp_layer = MLPLayerType.FC2
        elif hasattr(self.model.blocks[0].mlp, MLPLayerType.W1.value):
            self.target_input_mlp_layer = MLPLayerType.W1
            self.target_output_mlp_layer = MLPLayerType.W3
        else:
            raise ValueError(f"Unknown MLP architecture: {self.model.blocks[0].mlp}")

        self.embeddings_dim = getattr(self.model.blocks[0].mlp, self.target_output_mlp_layer.value).out_features
        self.default_mlp_hidden_dim = getattr(self.model.blocks[0].mlp, self.target_input_mlp_layer.value).out_features
        self.min_hidden_dim_keep_count = int(self.default_mlp_hidden_dim * min_hidden_dim_keep_ratio)

    @property
    def blocks(self) -> List[nn.Module]:
        return self.model.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def estimate_pruning_weights(self, data_loader: DataLoader, backward: bool = True):
        """Estimate the model pruning weights using a cross-entropy loss."""
        predict(model=self.model, data_loader=data_loader, device=self.device, backward=backward)

    def prune(
        self,
        data_loader: DataLoader,
        pruning_ratio: float,
        pruning_ratio_heads: float,
        block_weights: Optional[torch.Tensor] = None,
        estimate_pruning_weights: bool = True,
        **kwargs
    ):
        """
            Prune the model's blocks by the given pruning ratio using the
            data loader to estimate the pruning weights. Optionally, a block
            weights tensor of shape [B] can be provided to rescale the
            estimated weights for each block.
        """
        if estimate_pruning_weights:
            self.estimate_pruning_weights(data_loader=data_loader)

        # Compute the current hidden dimension and number of heads for each transformer block.
        # These are needed to compute the correct offset in the pruning weights.
        hidden_dims = [getattr(block.mlp, self.target_input_mlp_layer.value).bias.shape[0] for block in self.model.blocks]
        head_dims = [block_num_heads(block=block) for block in self.model.blocks]

        pruning_weights, head_pruning_weights = self.compute_pruning_weights(
            pruning_ratio=pruning_ratio,
            pruning_ratio_heads=pruning_ratio_heads,
            block_weights=block_weights
        )

        # Iterate over all the block hidden dimensionalities, and for each block,
        # select its pruning weights and corresponding indices
        block_offset, head_offset = 0, 0

        for i, (hidden_dim, num_heads) in enumerate(zip(hidden_dims, head_dims)):
            self.prune_block_mlp(
                block_index=i,
                pruning_weights=pruning_weights[block_offset:block_offset + hidden_dim],
                hidden_dim=hidden_dim
            )

            self.prune_block_attn(
                block_index=i,
                pruning_weights=head_pruning_weights[head_offset:head_offset + num_heads],
                num_heads=num_heads
            )

            # Update the block offsets
            block_offset, head_offset = block_offset + hidden_dim, head_offset + num_heads

    def compute_pruning_weights(
        self,
        pruning_ratio: float,
        pruning_ratio_heads: float,
        block_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Compute the pruning weights for each model head and hidden neuron.
        """
        # Compute the pruning weights for each block
        pruning_weights = self._reduce_pruning_weights()
        head_pruning_weights = self._reduce_heads_pruning_weights()

        # Scale each block's pruning weights by the provided weights. This is implemented
        # as a loop because the block pruning weights may have different sizes (e.g. after
        # the first step of global pruning)
        if block_weights is not None:
            block_weights = block_weights.to(self.device)

            # Split the block weights into mlp and head weights
            mlp_weights = block_weights[:self.num_blocks]
            head_weights = block_weights[self.num_blocks:].reshape(self.num_blocks, -1)

            for i in range(self.num_blocks):
                # Scale the pruning weights by their block weights
                pruning_weights[i] = pruning_weights[i] * mlp_weights[i]
                head_pruning_weights[i] = head_pruning_weights[i] * head_weights[i]

                # Sort the pruning weights in descending order, i.e. the inverse of what
                # we use for pruning, select the top-x% of the weights by magnitude,
                # and set their weight to a large value, so that they are not pruned.
                sorted_indices = pruning_weights[i].argsort(descending=True)
                sorted_indices = sorted_indices[:self.min_hidden_dim_keep_count]

                pruning_weights[i][sorted_indices] = torch.finfo(pruning_weights[i].dtype).max

                # Do the same for the head pruning weights
                sorted_head_indices = head_pruning_weights[i].argsort(descending=True)
                sorted_head_indices = sorted_head_indices[:self.min_head_keep_count]

                head_pruning_weights[i][sorted_head_indices] = torch.finfo(head_pruning_weights[i].dtype).max

        # Create a pruning vectors for the MLP blocks and attention heads
        pruning_weights = torch.cat(pruning_weights, dim=0)
        head_pruning_weights = torch.cat(head_pruning_weights, dim=0)

        # Sort the neuron indices in ascending order based on the pruning weights magnitudes.
        pruning_weights_indices = pruning_weights.argsort(descending=False)
        head_weights_indices = head_pruning_weights.argsort(descending=False)

        num_total_pruned_heads = int(self.default_num_heads * pruning_ratio_heads * self.num_blocks)
        num_total_pruned_neurons = int(self.default_mlp_hidden_dim * pruning_ratio * self.num_blocks)

        logging.info(f"pruning {num_total_pruned_neurons} neurons...")
        logging.info(f"pruning {num_total_pruned_heads} attention heads...")

        # Select the top-k neurons to be pruned
        indices_to_prune = pruning_weights_indices[:num_total_pruned_neurons]
        heads_indices_to_prune = head_weights_indices[:num_total_pruned_heads]

        # Zero out the pruning weights of the neurons to be pruned
        pruning_weights[indices_to_prune] = 0
        head_pruning_weights[heads_indices_to_prune] = 0

        return pruning_weights, head_pruning_weights

    def prune_block_mlp(self, block_index: int, pruning_weights: torch.tensor, hidden_dim: int):
        if self.target_input_mlp_layer == MLPLayerType.FC1:
            self.prune_block_standard_mlp(
                block_index=block_index,
                pruning_weights=pruning_weights,
                hidden_dim=hidden_dim
            )
        elif self.target_input_mlp_layer == MLPLayerType.W1:
            self.prune_block_swiglu_mlp(
                block_index=block_index,
                pruning_weights=pruning_weights,
                hidden_dim=hidden_dim
            )
        else:
            raise ValueError(f"Unknown MLP architecture: {self.model.blocks[block_index].mlp}")

    def prune_block_standard_mlp(self, block_index: int, pruning_weights: torch.tensor, hidden_dim: int):
        # Sort the pruning weights such that the zeroes are at the end of the list
        sorting_indices = pruning_weights.argsort(dim=-1, descending=True)

        # Calculate the number of neurons to be kept
        num_zeros = len(torch.where(pruning_weights == 0)[0])
        num_neurons = hidden_dim - num_zeros

        # Sort the columns and rows of the weight matrices and bias to shift the
        # dimensions to be pruned to the end of the weight matrices and bias
        self.model.blocks[block_index].mlp.fc1.weight.data = self.model.blocks[block_index].mlp.fc1.weight.data[sorting_indices, :]
        self.model.blocks[block_index].mlp.fc2.weight.data = self.model.blocks[block_index].mlp.fc2.weight.data[:, sorting_indices]

        self.model.blocks[block_index].mlp.fc1.bias.data = self.model.blocks[block_index].mlp.fc1.bias[sorting_indices]

        # Prune the weight matrices by removing the last num_zeros columns and rows
        self.model.blocks[block_index].mlp.fc1.weight.data = self.model.blocks[block_index].mlp.fc1.weight.data[:num_neurons, :]
        self.model.blocks[block_index].mlp.fc2.weight.data = self.model.blocks[block_index].mlp.fc2.weight.data[:, :num_neurons]

        self.model.blocks[block_index].mlp.fc1.bias.data = self.model.blocks[block_index].mlp.fc1.bias[:num_neurons]

    def prune_block_swiglu_mlp(self, block_index: int, pruning_weights: torch.tensor, hidden_dim: int):
        # Sort the pruning weights such that the zeroes are at the end of the list
        sorting_indices = pruning_weights.argsort(dim=-1, descending=True)

        # Calculate the number of neurons to be kept
        num_zeros = len(torch.where(pruning_weights == 0)[0])
        num_neurons = hidden_dim - num_zeros

        # Sort the columns and rows of the weight matrices and bias to shift the
        # dimensions to be pruned to the end of the weight matrices and bias
        self.model.blocks[block_index].mlp.w1.weight.data = self.model.blocks[block_index].mlp.w1.weight.data[sorting_indices, :]
        self.model.blocks[block_index].mlp.w2.weight.data = self.model.blocks[block_index].mlp.w2.weight.data[sorting_indices, :]

        self.model.blocks[block_index].mlp.w3.weight.data = self.model.blocks[block_index].mlp.w3.weight.data[:, sorting_indices]

        self.model.blocks[block_index].mlp.w1.bias.data = self.model.blocks[block_index].mlp.w1.bias[sorting_indices]
        self.model.blocks[block_index].mlp.w2.bias.data = self.model.blocks[block_index].mlp.w2.bias[sorting_indices]

        # Prune the weight matrices by removing the last num_zeros columns and rows
        self.model.blocks[block_index].mlp.w1.weight.data = self.model.blocks[block_index].mlp.w1.weight.data[:num_neurons, :]
        self.model.blocks[block_index].mlp.w2.weight.data = self.model.blocks[block_index].mlp.w2.weight.data[:num_neurons, :]

        self.model.blocks[block_index].mlp.w3.weight.data = self.model.blocks[block_index].mlp.w3.weight.data[:, :num_neurons]

        self.model.blocks[block_index].mlp.w1.bias.data = self.model.blocks[block_index].mlp.w1.bias[:num_neurons]
        self.model.blocks[block_index].mlp.w2.bias.data = self.model.blocks[block_index].mlp.w2.bias[:num_neurons]

    def prune_block_attn(self, block_index: int, pruning_weights: torch.tensor, num_heads: int):
        # Sort the pruning weights such that the zeroes are at the end of the list
        sorting_indices = pruning_weights.argsort(dim=-1, descending=True)

        # Calculate the number of heads to be kept
        num_zeros = len(torch.where(pruning_weights == 0)[0])
        num_remaining_heads = num_heads - num_zeros

        block = self.model.blocks[block_index]

        # Reshape the output projection weights to [E, H, H_D], where H_D
        # is the head dimension. We split the matrix into heads along the
        # input dimension, and then sort the heads using the indices, so
        # that the least important heads are at the end.
        out_weights = block.attn.proj.weight.reshape(
            self.embeddings_dim,
            num_heads,
            self.head_dim
        )[:, sorting_indices, :]

        # We then prune the least important heads and update the weight matrix.
        # There is no need to update the bias, as we are operating on the input dimensions.
        block.attn.proj.weight.data = out_weights[:, :num_remaining_heads, :].reshape(
            self.embeddings_dim,
            -1
        )

        # For the qkv matrix, we first slice it in q, k, v, and then split it
        # into heads along the output dimensions. We then proceed as before
        qkv_weights = block.attn.qkv.weight.view(3, num_heads, self.head_dim, self.embeddings_dim)
        qkv_weights = qkv_weights[:, sorting_indices, :, :]

        # Prune the least important heads
        qkv_weights = qkv_weights[:, :num_remaining_heads, :, :]
        qkv_weights = qkv_weights.reshape(-1, self.embeddings_dim)

        block.attn.qkv.weight.data = qkv_weights

        # Prune the QKV bias (and optional bias mask, used by DINOv3)
        self.prune_qkv_bias(
            block=block,
            sorting_indices=sorting_indices,
            num_heads=num_heads,
            num_remaining_heads=num_remaining_heads,
            bias_key="bias"
        )

        self.prune_qkv_bias(
            block=block,
            sorting_indices=sorting_indices,
            num_heads=num_heads,
            num_remaining_heads=num_remaining_heads,
            bias_key="bias_mask"
        )

        # Some models, such as EVA ViT-G, have separate qkv biases. Prune them appropriately if they exist.
        for key in ["q_bias", "k_bias", "v_bias"]:
            if not hasattr(block.attn, key):
                continue

            if getattr(block.attn, key) is None:
                continue

            qkv_bias = getattr(block.attn, key).view(num_heads, self.head_dim)
            qkv_bias = qkv_bias[sorting_indices, :]

            # Prune the least important heads from the bias
            qkv_bias = qkv_bias[:num_remaining_heads, :]
            qkv_bias = qkv_bias.reshape(-1)

            getattr(block.attn, key).data = qkv_bias

        # Update the number of heads to ensure the attention layer works as expected
        block.attn.num_heads = num_remaining_heads

    def prune_qkv_bias(self, block: nn.Module, sorting_indices: torch.Tensor, num_heads: int, num_remaining_heads: int, bias_key: str):
        if hasattr(block.attn.qkv, bias_key) and getattr(block.attn.qkv, bias_key) is not None:
            # Repeat the process for the bias, as we're editing the output dimension
            bias = getattr(block.attn.qkv, bias_key).view(3, num_heads, self.head_dim)
            bias = bias[:, sorting_indices, :]

            # Prune the least important heads from the bias
            bias = bias[:, :num_remaining_heads]
            bias = bias.reshape(-1)

            getattr(block.attn.qkv, bias_key).data = bias

    def _reduce_pruning_weights(self) -> List[torch.Tensor]:
        """
            Reduce the pruning weights of the model's MLP blocks to a list
            of [E] tensors, where E is the embeddings dimension.
        """
        return [
            (getattr(block.mlp, self.target_input_mlp_layer.value).weight.grad ** 2).mean(dim=1) for block in self.model.blocks
        ]

    def _reduce_heads_pruning_weights(self) -> List[torch.Tensor]:
        """
            Reduce the pruning weights of each attention block to a list
            of [H] tensors, where H is the number of heads.
        """
        reduced = []

        for block in self.model.blocks:
            num_heads = block_num_heads(block=block)

            # We use the values matrix to estimate the head pruning weights
            head_grads = block.attn.qkv.weight.grad.reshape(
                3,
                num_heads,
                self.head_dim,
                self.embeddings_dim
            )[2, :, :, :] ** 2

            head_grads = head_grads.mean(dim=(1, 2))

            reduced.append(head_grads)

        return reduced
