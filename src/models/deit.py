import timm
import torch.nn as nn

from .base_factory import ModelFactory


class DeITViT16BFactory(ModelFactory):
    """Factory class for a ViT-B/16 model trained on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("deit_base_distilled_patch16_224"),
            is_training=False
        )


class DeIT3ViT16HFactory(ModelFactory):
    """Factory class for a ViT-H/14 model trained on ImageNet-22K and fine-tuned on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("deit3_huge_patch14_224.fb_in22k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("deit3_huge_patch14_224.fb_in22k_ft_in1k"),
            is_training=False
        )


class DeIT3ViT16LFactory(ModelFactory):
    """Factory class for a ViT-L/16 model trained on ImageNet-22K and fine-tuned on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("deit3_large_patch16_224.fb_in22k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("deit3_large_patch16_224.fb_in22k_ft_in1k"),
            is_training=False
        )


class DeIT3ViT16BFactory(ModelFactory):
    """Factory class for a ViT-B/16 model trained on ImageNet-22K and fine-tuned on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("deit3_base_patch16_224.fb_in22k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("deit3_base_patch16_224.fb_in22k_ft_in1k"),
            is_training=False
        )


class DeIT3ViT16SFactory(ModelFactory):
    """Factory class for a ViT-S/16 model trained on ImageNet-22K and fine-tuned on ImageNet-1K."""
    def build(self, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("deit3_small_patch16_224.fb_in22k_ft_in1k", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("deit3_small_patch16_224.fb_in22k_ft_in1k"),
            is_training=False
        )
