import torch
import logging

from typing import Optional, Tuple
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

            fc1_mask, fc2_mask, fc1_bias_mask = self.compute_block_mlp_mask(
                block_index=i,
                pruning_weights=pruning_weights[block_offset:block_offset + hidden_dim]
            )

            qkv_mask, qkv_bias_mask, qkv_bias_mask_mask, attn_proj_mask = self.compute_block_attn_mask(
                block_index=i,
                pruning_weights=head_pruning_weights[head_offset:head_offset + num_heads],
                num_heads=num_heads
            )

            layers = {
                "attn.qkv": SparseGPTLayerWrapper(layer=block.attn.qkv, mask=qkv_mask, bias_mask=qkv_bias_mask, bias_mask_mask=qkv_bias_mask_mask),
                "attn.proj": SparseGPTLayerWrapper(layer=block.attn.proj, mask=attn_proj_mask),
                "mlp.fc1": SparseGPTLayerWrapper(layer=block.mlp.fc1, mask=fc1_mask, bias_mask=fc1_bias_mask),
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

    def compute_block_mlp_mask(self, block_index: int, pruning_weights: torch.Tensor):
        mlp = self.model.blocks[block_index].mlp
        pruning_weights = pruning_weights == 0

        fc1_mask = torch.zeros_like(mlp.fc1.weight, dtype=torch.bool)
        fc2_mask = torch.zeros_like(mlp.fc2.weight, dtype=torch.bool)

        fc1_mask[pruning_weights, :] = True        
        fc2_mask[:, pruning_weights] = True

        return fc1_mask, fc2_mask, pruning_weights.clone()

    def compute_block_attn_mask(
        self,
        block_index: int,
        pruning_weights: torch.Tensor,
        num_heads: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.model.blocks[block_index].attn
        pruning_weights = pruning_weights == 0

        # Create masks for the QKV and attn projection weights
        attn_proj_mask = torch.zeros_like(attn.proj.weight, dtype=torch.bool)
        qkv_mask = torch.zeros_like(attn.qkv.weight, dtype=torch.bool)

        for h in range(num_heads):
            if pruning_weights[h]:
                attn_proj_mask[:, h * self.head_dim:(h + 1) * self.head_dim] = True

                # Prune the Q, K, V sub-matrices
                qkv_mask[h * self.head_dim:(h + 1) * self.head_dim, :] = True
                qkv_mask[(num_heads + h) * self.head_dim:(num_heads + h + 1) * self.head_dim, :] = True
                qkv_mask[(2 * num_heads + h) * self.head_dim:(2 * num_heads + h + 1) * self.head_dim, :] = True

        # Create masks for the QKV bias and bias mask
        qkv_bias_mask = self.compute_block_qkv_bias_mask(
            block_index=block_index,
            pruning_weights=pruning_weights,
            num_heads=num_heads,
            bias_key="bias"
        )

        qkv_bias_mask_mask = self.compute_block_qkv_bias_mask(
            block_index=block_index,
            pruning_weights=pruning_weights,
            num_heads=num_heads,
            bias_key="bias_mask"
        )

        return qkv_mask, qkv_bias_mask, qkv_bias_mask_mask, attn_proj_mask

    def compute_block_qkv_bias_mask(
        self,
        block_index: int,
        pruning_weights: torch.Tensor,
        num_heads: int,
        bias_key: str
    ) -> Optional[torch.Tensor]:
        attn = self.model.blocks[block_index].attn
        pruning_weights = pruning_weights == 0

        if not hasattr(attn, bias_key):
            return None

        bias_mask = torch.zeros_like(getattr(attn, bias_key).bias, dtype=torch.bool)

        for h in range(num_heads):
            if pruning_weights[h]:
                bias_mask[h * self.head_dim:(h + 1) * self.head_dim] = True
                bias_mask[(num_heads + h) * self.head_dim:(num_heads + h + 1) * self.head_dim] = True
                bias_mask[(2 * num_heads + h) * self.head_dim:(2 * num_heads + h + 1) * self.head_dim] = True

        return bias_mask
