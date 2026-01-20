import os
import torch
import torch.nn as nn

from typing import Tuple, Optional

from src.utils.modules import OptionalLinear

from .enums import ModelType
from .clip import SigLIP2ViT16BFactory, SigLIP2ViT16GFactory
from .augreg import ViT16SFactory, ViT16BFactory, ViT16LFactory
from .dino import DINOViT16SFactory, DINOViT16BFactory, DINOV3Factory
from .deit import DeITViT16BFactory, DeIT3ViT16HFactory, DeIT3ViT16LFactory, DeIT3ViT16BFactory, DeIT3ViT16SFactory


def load_model(type_: ModelType, cache_dir: str, device: torch.device = torch.device("cpu"), transform_only: bool = False, **kwargs) -> Tuple[Optional[nn.Module], nn.Module]:
    """Returns an initialized model of the given kind"""
    factory = {
        ModelType.AUGREG_VIT_S_16_IN21K_FT_IN1K: ViT16SFactory,
        ModelType.AUGREG_VIT_B_16_IN21K_FT_IN1K: ViT16BFactory,
        ModelType.AUGREG_VIT_L_16_IN21K_FT_IN1K: ViT16LFactory,
        ModelType.DINO_VIT_S_16: DINOViT16SFactory,
        ModelType.DINO_VIT_B_16: DINOViT16BFactory,
        ModelType.SIGLIP2_VIT_B_16: SigLIP2ViT16BFactory,
        ModelType.SIGLIP2_VIT_G_16: SigLIP2ViT16GFactory,
        ModelType.DEIT_VIT_B_16: DeITViT16BFactory,
        ModelType.DEIT_3_VIT_H_14: DeIT3ViT16HFactory,
        ModelType.DEIT_3_VIT_L_16: DeIT3ViT16LFactory,
        ModelType.DEIT_3_VIT_B_16: DeIT3ViT16BFactory,
        ModelType.DEIT_3_VIT_S_16: DeIT3ViT16SFactory,
        ModelType.DINO_V3_VIT_B_16: DINOV3Factory,
        ModelType.DINO_V3_VIT_H_16: DINOV3Factory
    }[type_]()

    # Make sure that the cache directory exists, and if that
    # is not the case create it
    cache_dir = os.path.join(cache_dir, "models")
    os.makedirs(cache_dir, exist_ok=True)

    if transform_only:
        return None, factory.get_transform(**kwargs)

    model = factory.build(
        model_type=type_,
        cache_dir=cache_dir,
        **kwargs
    ).to(device)

    if hasattr(model, "head"):
        model.head = OptionalLinear(linear=model.head)

    if hasattr(model, "head_dist"):
        model.head_dist = OptionalLinear(linear=model.head_dist)

    return model, factory.get_transform(**kwargs)
