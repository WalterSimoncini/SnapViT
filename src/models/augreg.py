import timm
import torch.nn as nn

from .base_factory import ModelFactory


class ViT16SFactory(ModelFactory):
    """Factory class for a ViT-S/16 model trained on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_small_patch16_224.augreg_in21k_ft_in1k"),
            is_training=False
        )


class ViT16BFactory(ModelFactory):
    """Factory class for a ViT-B/16 model trained on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_base_patch16_224.augreg_in21k_ft_in1k"),
            is_training=False
        )


class ViT16LFactory(ModelFactory):
    """Factory class for a ViT-L/16 model trained on ImageNet-21K and fine-tuned on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("vit_large_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_large_patch16_224.augreg_in21k_ft_in1k"),
            is_training=False
        )
