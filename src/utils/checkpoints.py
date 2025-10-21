import torch
import torch.nn as nn

from src.utils.models import block_num_heads


def load_pruned_checkpoint(
    base_model: nn.Module,
    checkpoint_path: str, 
    device: torch.device
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        checkpoint = checkpoint["model"]

    # Remove the optional linear head
    if "head.linear.weight" in checkpoint:
        del checkpoint["head.linear.weight"]
        del checkpoint["head.linear.bias"]

    # Remove regular and distillation heads
    if "head.weight" in checkpoint:
        del checkpoint["head.weight"]
        del checkpoint["head.bias"]

    if "head_dist.weight" in checkpoint:
        del checkpoint["head_dist.weight"]
        del checkpoint["head_dist.bias"]

    for i, block in enumerate(base_model.blocks):
        # Reshape each hidden MLP dimension of the base model to match the checkpoint
        fc1_size = checkpoint[f"blocks.{i}.mlp.fc1.weight"].shape[0]
        fc2_size = checkpoint[f"blocks.{i}.mlp.fc2.weight"].shape[1]

        block.mlp.fc1.weight.data = block.mlp.fc1.weight.data[:fc1_size, :]
        block.mlp.fc2.weight.data = block.mlp.fc2.weight.data[:, :fc2_size]

        block.mlp.fc1.bias.data = block.mlp.fc1.bias[:fc1_size]

        # Reshape each self-attention block dimensions to match the checkpoint
        qkv_shape = checkpoint[f"blocks.{i}.attn.qkv.weight"].shape[0]
        proj_shape = checkpoint[f"blocks.{i}.attn.proj.weight"].shape[1]

        block.attn.qkv.weight.data = block.attn.qkv.weight.data[:qkv_shape, :]
        block.attn.qkv.bias.data = block.attn.qkv.bias[:qkv_shape]

        block.attn.proj.weight.data = block.attn.proj.weight.data[:, :proj_shape]

        # Update the number of heads
        block.attn.num_heads = block_num_heads(block=block)

    # Load the checkpoint
    base_model.load_state_dict(checkpoint)

    return base_model
