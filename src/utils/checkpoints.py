import torch
import torch.nn as nn

from src.utils.models import block_num_heads


def load_pruned_checkpoint(
    base_model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
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
        # Get the pruned dimensions from checkpoint
        fc1_out, fc1_in = checkpoint[f"blocks.{i}.mlp.fc1.weight"].shape
        fc2_out, fc2_in = checkpoint[f"blocks.{i}.mlp.fc2.weight"].shape
        qkv_out, qkv_in = checkpoint[f"blocks.{i}.attn.qkv.weight"].shape
        proj_out, proj_in = checkpoint[f"blocks.{i}.attn.proj.weight"].shape

        # Re-initialize the MLP layers with correct dimensions
        block.mlp.fc1 = nn.Linear(fc1_in, fc1_out, bias=block.mlp.fc1.bias is not None)
        block.mlp.fc2 = nn.Linear(fc2_in, fc2_out, bias=block.mlp.fc2.bias is not None)

        # Re-initialize the attention layers with correct dimensions
        block.attn.qkv = nn.Linear(qkv_in, qkv_out, bias=block.attn.qkv.bias is not None)
        block.attn.proj = nn.Linear(proj_in, proj_out, bias=block.attn.proj.bias is not None)

        # Update the number of heads in the block
        block.attn.num_heads = block_num_heads(block=block)

    # Load the checkpoint
    base_model.load_state_dict(checkpoint)
    base_model = base_model.to(device)

    return base_model
