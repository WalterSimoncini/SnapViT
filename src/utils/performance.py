import time
import torch
import logging
import torch.nn as nn

from tqdm import tqdm
from typing import Union

from src.models.enums import MLPLayerType
from src.models.prunable import PrunableModel
from src.utils.models import model_image_size


def estimate_model_speed(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    num_steps: int = 100
) -> float:
    """
        Measures the number of samples per second that the model can process.

        Args:
            model (nn.Module): The model to measure the performance of.
            device (torch.device): The device to run the measurements on.
            batch_size (int): The batch size to use for the measurements.
            num_steps (int): The number of forward steps to run for the measurements.

        Returns:
            float: The number of samples per second.
    """
    model.eval()

    input_size = model_image_size(model=model)
    dummy_tensor = torch.randn(batch_size, 3, input_size, input_size).to(device)
    start_time = time.time()

    with torch.no_grad():
        for _ in tqdm(range(num_steps), desc="estimating model speed"):
            model(dummy_tensor)

    elapsed_time = time.time() - start_time
    total_forwarded_samples = num_steps * batch_size

    return total_forwarded_samples / elapsed_time


def estimate_model_flops(model: Union[nn.Module, PrunableModel]) -> int:
    """
        Estimate the model FLOPs using the method described here:
        
        https://www.adamcasson.com/posts/transformer-flops#appendix-a

        Which is mostly based on Training Compute-Optimal Large Language
        Models by Hoffmann et al. 
    """
    if isinstance(model, PrunableModel):
        model = model.model

    patch_size = model_patch_size(model=model)

    logging.info(f"the model patch size is: {patch_size}")

    # Split the input image into patches and add the [CLS] token
    input_size = model_image_size(model=model)
    num_tokens = (input_size // patch_size) ** 2 + 1

    total_flops = 0

    for block in model.blocks:
        # Compute the MLP FLOPs
        if hasattr(block.mlp, MLPLayerType.FC1.value):
            hidden_mlp_dim = block.mlp.fc1.weight.shape[0]
            mlp_flops = 2 * (2 * model.embed_dim * hidden_mlp_dim) * num_tokens
        elif hasattr(block.mlp, MLPLayerType.W1.value):
            hidden_mlp_dim = block.mlp.w1.weight.shape[0]

            # SwiGLUFFN has an additional linear layer
            mlp_flops = 2 * (3 * model.embed_dim * hidden_mlp_dim) * num_tokens
        else:
            raise ValueError(f"Unknown MLP architecture: {block.mlp}")

        # Compute the QKV and projection FLOPs
        head_dim, num_heads = block.attn.head_dim, block.attn.num_heads

        qkv_flops = 6 * model.embed_dim * head_dim * num_heads * num_tokens
        proj_flops = 2 * model.embed_dim * head_dim * num_heads * num_tokens

        attention_flops = qkv_flops + proj_flops

        # Compute the FLOPs for the self-attention
        qk_logits_flops = 2 * num_heads * head_dim * (num_tokens ** 2)
        softmax_flops = 3 * num_heads * (num_tokens ** 2)
        reduction_flops = 2 * num_heads * head_dim * (num_tokens ** 2)

        self_attention_flops = qk_logits_flops + softmax_flops + reduction_flops

        # Add the block FLOPs to the total
        total_flops += mlp_flops + attention_flops + self_attention_flops

    # Add the FLOPs for the embedding layer. Here we assume that input images
    # have 3 channels and ignore the [CLS] token.
    total_flops += 2 * (num_tokens - 1) * (patch_size ** 2) * 3 * model.embed_dim

    return total_flops


def model_patch_size(model: Union[nn.Module, PrunableModel]) -> int:
    """
        Get the patch size for the given model.
    """
    if isinstance(model, PrunableModel):
        model = model.model

    if hasattr(model, "patch_size"):
        patch_size = model.patch_size
    elif hasattr(model, "patch_embed"):
        patch_embed = model.patch_embed

        if hasattr(patch_embed, "patch_size"):
            patch_size = patch_embed.patch_size
        else:
            patch_size = patch_embed.proj.kernel_size
    else:
        raise ValueError(f"could not find the patch size for the given model of class {model.__class__.__name__}.")

    # Unwrap the patch size if needed
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]

    return patch_size
