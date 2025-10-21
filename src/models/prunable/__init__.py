import torch
import torch.nn as nn

from typing import Tuple

from .base import PrunableModel
from .lamp import LAMPPrunableModel
from .dino import DINOPrunableModel
from .random import RandomPrunableModel
from .sparsegpt.model import SparseGPTPrunableModel
from .dino.sparsegpt import SparseGPTDINOPrunableModel
from .snip_magnitude.ce import CESNIPMagnitudePrunableModel
from .snip_magnitude.dino import DINOSNIPMagnitudePrunableModel

from src.utils.models import model_image_size
from src.models.enums import PrunableModelType
from .dino.augmentation import DINODataAgumentation


def load_prunable_model(
    type_: PrunableModelType,
    backbone: nn.Module,
    backbone_transform: nn.Module,
    device: torch.device,
    estimation_epochs: int = 1,
    min_hidden_dim_ratio: float = 0.2,
    min_head_ratio: float = 0.2
) -> Tuple[PrunableModel, nn.Module]:
    model_class = {
        PrunableModelType.DINO: DINOPrunableModel,
        PrunableModelType.CROSS_ENTROPY: PrunableModel,
        PrunableModelType.LAMP: LAMPPrunableModel,
        PrunableModelType.RANDOM: RandomPrunableModel,
        PrunableModelType.SNIP_MAGNITUDE: CESNIPMagnitudePrunableModel,
        PrunableModelType.SNIP_MAGNITUDE_DINO: DINOSNIPMagnitudePrunableModel,
        PrunableModelType.SPARSE_GPT: SparseGPTPrunableModel,
        PrunableModelType.SPARSE_GPT_DINO: SparseGPTDINOPrunableModel
    }[type_]

    input_size = model_image_size(model=backbone)

    transform = {
        PrunableModelType.DINO: DINODataAgumentation(crops_size=input_size),
        PrunableModelType.CROSS_ENTROPY: backbone_transform,
        PrunableModelType.LAMP: backbone_transform,
        PrunableModelType.RANDOM: backbone_transform,
        PrunableModelType.SNIP_MAGNITUDE: backbone_transform,
        PrunableModelType.SNIP_MAGNITUDE_DINO: DINODataAgumentation(crops_size=input_size),
        PrunableModelType.SPARSE_GPT: backbone_transform,
        PrunableModelType.SPARSE_GPT_DINO: DINODataAgumentation(crops_size=input_size)
    }[type_]

    return model_class(
        model=backbone,
        device=device,
        estimation_epochs=estimation_epochs,
        min_hidden_dim_ratio=min_hidden_dim_ratio,
        min_head_ratio=min_head_ratio
    ), transform
