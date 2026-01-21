import torch
import logging

from typing import Optional
from torch.utils.data import DataLoader

from src.models.prunable.base import PrunableModel
from src.utils.models import block_num_heads, capture_block_inputs

from .layer_wrapper import SparseGPTLayerWrapper


class SparseGPTPrunableModel(PrunableModel):
    def prune(
        self,
        data_loader: DataLoader,
        pruning_ratio: float,
        pruning_ratio_heads: float,
        block_weights: Optional[torch.Tensor] = None,
        estimate_pruning_weights: bool = True,
        apply_correction: bool = False,
        correction_data_loader: Optional[DataLoader] = None,
        **kwargs
    ):
        # If no correction needs to be applied, prune the model as usual.
        if not apply_correction:
            super().prune(
                data_loader=data_loader,
                pruning_ratio=pruning_ratio,
                pruning_ratio_heads=pruning_ratio_heads,
                block_weights=block_weights,
                estimate_pruning_weights=estimate_pruning_weights,
            )

            return

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

        # Capture the inputs for the first block
        inputs = capture_block_inputs(
            model=self.model,
            data_loader=correction_data_loader,
            device=self.device,
            block_index=0,
            show_progress=True
        )

        for i, (hidden_dim, num_heads) in enumerate(zip(hidden_dims, head_dims)):
            logging.info(f"pruning block {i} with weight correction...")

            block = self.model.blocks[i]

            fc2_mask = self.compute_fc2_mask(
                block_index=i,
                pruning_weights=pruning_weights[block_offset:block_offset + hidden_dim]
            )

            attn_proj_mask = self.compute_attn_proj_mask(
                block_index=i,
                pruning_weights=head_pruning_weights[head_offset:head_offset + num_heads],
                num_heads=num_heads
            )

            # Only apply SparseGPT correction to column-pruned layers (fc2, attn.proj).
            # Row-pruned layers (fc1, qkv) don't benefit from correction since corrections
            # only propagate to their right, but the entire row is then discarded.
            layers = {
                "attn.proj": SparseGPTLayerWrapper(layer=block.attn.proj, mask=attn_proj_mask),
                "mlp.fc2": SparseGPTLayerWrapper(layer=block.mlp.fc2, mask=fc2_mask)
            }

            # Collect the input activations for every target layer in the block
            def build_hook(name: str):
                def forward_hook(_, input, __):
                    layers[name].update(input[0])

                return forward_hook

            outputs = []
            handles = [
                layer.layer.register_forward_hook(build_hook(name)) for name, layer in layers.items()
            ]

            with torch.no_grad():
                for sample in inputs:
                    block(sample.unsqueeze(dim=0))

            for handle in handles:
                handle.remove()

            # Zero out the target weights and apply the weight correction
            for layer in layers.values():
                layer.prune()

            # Prune the MLP block and heads as per usual
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

            # Collect the next inputs for the next block after pruning
            with torch.no_grad():
                for sample in inputs:
                    outputs.append(block(sample.unsqueeze(dim=0)))

            inputs = torch.cat(outputs, dim=0)

            # Update the block offsets
            block_offset, head_offset = block_offset + hidden_dim, head_offset + num_heads

    def compute_fc2_mask(self, block_index: int, pruning_weights: torch.Tensor) -> torch.Tensor:
        """Compute column mask for fc2 based on pruning weights."""
        fc2 = self.model.blocks[block_index].mlp.fc2
        prune_indices = pruning_weights == 0

        mask = torch.zeros_like(fc2.weight, dtype=torch.bool)
        mask[:, prune_indices] = True

        return mask

    def compute_attn_proj_mask(
        self,
        block_index: int,
        pruning_weights: torch.Tensor,
        num_heads: int
    ) -> torch.Tensor:
        """Compute column mask for attn.proj based on head pruning weights."""
        proj = self.model.blocks[block_index].attn.proj
        prune_indices = pruning_weights == 0

        mask = torch.zeros_like(proj.weight, dtype=torch.bool)

        for h in range(num_heads):
            if prune_indices[h]:
                mask[:, h * self.head_dim:(h + 1) * self.head_dim] = True

        return mask
