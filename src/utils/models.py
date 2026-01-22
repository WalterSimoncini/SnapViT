import copy
import torch
import torch.nn as nn

from tqdm import tqdm
from contextlib import contextmanager
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from timm.models.vision_transformer import Block
from typing import Tuple, List, Union, Generator

from src.models.enums import MLPLayerType


@contextmanager
def eval_mode(model: nn.Module) -> Generator[None, None, None]:
    """
        Temporarily set a model in eval mode, restoring
        its original state on exit.
    """
    state = model.training
    model.eval()

    try:
        yield
    finally:
        model.train(state)


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    backward: bool = False,
    estimation_epochs: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward (and optionally backward) pass over the given data loader."""
    predictions, targets = [], []

    # Always clear gradients to avoid issues with size mismatches after pruning
    model.zero_grad()

    # Only enable the gradient calculation if asked to backpropagate
    with torch.set_grad_enabled(backward):
        for _ in range(estimation_epochs):
            for images, labels in tqdm(data_loader, desc=f"running predict with backward={backward}"):
                images, labels = images.to(device), labels.to(device)
                preds = model(images)

                if backward:
                    loss = cross_entropy(preds, labels)
                    loss.backward()

                predictions.append(preds.argmax(dim=-1))
                targets.append(labels)

    return torch.cat(predictions), torch.cat(targets)


def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
    move_to_cpu: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    features, targets = [], []
    cpu_device = torch.device("cpu")

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_fp16):
        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                images = images.to(device, non_blocking=True)
                targets.append(labels)

                embeddings = model(images)

                if move_to_cpu:
                    embeddings = embeddings.to(cpu_device, non_blocking=True)

                features.append(embeddings)

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def count_parameters(model: nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])


def count_mlp_parameters(model: nn.Module) -> int:
    """Returns the total number of MLP parameters in the model."""
    if hasattr(model.blocks[0].mlp, MLPLayerType.FC1.value):
        params = sum([
            model.blocks[0].mlp.fc1.weight.numel(),
            model.blocks[0].mlp.fc2.weight.numel()
        ])
    elif hasattr(model.blocks[0].mlp, MLPLayerType.W1.value):
        params = sum([
            model.blocks[0].mlp.w1.weight.numel(),
            model.blocks[0].mlp.w2.weight.numel(),
            model.blocks[0].mlp.w3.weight.numel()
        ])
    else:
        raise ValueError(f"Unknown MLP architecture: {model.blocks[0].mlp}")

    return params * len(model.blocks)


def count_attn_parameters(model: nn.Module) -> int:
    """Returns the total number of MLP parameters in the model."""
    params = sum([
        model.blocks[0].attn.qkv.weight.numel(),
        model.blocks[0].attn.proj.weight.numel()
    ])

    return params * len(model.blocks)


def deepcopy_model(model: nn.Module) -> nn.Module:
    """Deepcopy a model, including its gradients."""
    copied_model = copy.deepcopy(model)

    # Copy over the gradients from the original model for each parameter
    for original, copied in zip(model.parameters(), copied_model.parameters()):
        if original.grad is not None:
            copied.grad = original.grad.clone()

    return copied_model


def freeze_model(model: nn.Module, exclusions: List[str] = []) -> None:
    """
        Freezes a model, excluding the weight and bias of the
        layer paths in the exclusions array
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for layer_path in exclusions:
        layer = get_layer(model=model, path=layer_path)
        layer.weight.requires_grad = True

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad = True


def get_layer(model: nn.Module, path: str):
    """
        Returns the layer specified by path for the
        given model. Path should be specified as a property
        path, e.g. "model.mlp.proj". Indices within Sequential
        blocks should be specified as "model.sequential.0"
    """
    path_components = path.split(".")
    layer = getattr(model, path_components[0])

    for comp in path_components[1:]:
        layer = getattr(layer, comp)

    return layer


def block_num_heads(block: Union[Block, nn.Module]) -> int:
    """Returns the number of heads in a given transformer block."""
    return block.attn.qkv.weight.shape[0] // 3 // block.attn.head_dim


def model_image_size(model: nn.Module) -> int:
    if hasattr(model, "image_size"):
        return model.image_size[0]

    return model.patch_embed.img_size[0]


def capture_block_inputs(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    block_index: int = 0,
    show_progress: bool = False
) -> torch.Tensor:
    """Capture input activations for a given transformer block."""
    inputs = []

    def hook(_, inp, __):
        # inp is a tuple of the forward function's positional arguments
        # For standard ViT: inp = (tensor,)
        # For DINOv3: inp = (list_of_tensors, rope_sincos)
        x = inp[0]

        # If x is a list of tensors (DINOv3), take the first one
        if isinstance(x, (tuple, list)):
            x = x[0]

        inputs.append(x.detach())

    handle = model.blocks[block_index].register_forward_hook(hook)
    loader = tqdm(data_loader, desc="capturing block inputs") if show_progress else data_loader

    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))

    handle.remove()

    return torch.cat(inputs, dim=0)
