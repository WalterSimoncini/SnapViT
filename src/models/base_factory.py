import types
import torch.nn as nn

from timm.models.eva import Eva
from abc import ABC, abstractmethod
from timm.models.vision_transformer import VisionTransformer

from src.utils.modules.eva import eva_forward
from src.utils.modules import DynamicHeadsAttention


class ModelFactory(ABC):
    @abstractmethod
    def build(self, **kwargs) -> nn.Module:
        """
            Returns a randomly-initialized or pretrained
            model or feature extractor
        """
        raise NotImplementedError

    @abstractmethod
    def get_transform(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def replace_attention(self, model: VisionTransformer | nn.Module) -> VisionTransformer | nn.Module:
        for block in model.blocks:
            if type(model).__name__ == "DinoVisionTransformer":
                block.attn.head_dim = 64
            elif isinstance(model, VisionTransformer):
                block.attn = DynamicHeadsAttention.from_timm_attn(block.attn)
            elif isinstance(model, Eva):
                block.attn.head_dim = block.attn.proj.in_features // block.attn.num_heads
                block.attn.forward = types.MethodType(eva_forward, block.attn)
            else:
                block.attn = DynamicHeadsAttention.from_dino_attn(block.attn)

        return model
