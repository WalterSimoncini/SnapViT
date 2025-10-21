import os
import timm
import torch
import torch.nn as nn

from torchvision import transforms as tf

from .enums import ModelType
from .base_factory import ModelFactory


class DINOViT16SFactory(ModelFactory):
    def build(self, model_type: ModelType, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("vit_small_patch16_224.dino", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_small_patch16_224.dino"),
            is_training=False
        )


class DINOViT16BFactory(ModelFactory):
    def build(self, model_type: ModelType, **kwargs) -> nn.Module:
        return self.replace_attention(
            timm.create_model("vit_base_patch16_224.dino", pretrained=True)
        )

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_base_patch16_224.dino"),
            is_training=False
        )


class DINOV3Factory(ModelFactory):
    def build(self, model_type: ModelType, cache_dir: str, **kwargs) -> nn.Module:
        model2weights = {
            ModelType.DINO_V3_VIT_B_16: "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            ModelType.DINO_V3_VIT_H_16: "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
        }

        weights_path = os.path.join(os.path.join(cache_dir, "dinov3"), model2weights[model_type])

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file for {model_type} not found at {weights_path}")

        model = torch.hub.load(os.path.join("architectures", "dinov3"), model_type.value, source="local", weights=weights_path)
        model = self.replace_attention(model)

        return model

    def get_transform(self, **kwargs) -> nn.Module:
        return tf.Compose([
            tf.ToTensor(),
            tf.Resize((224, 224), antialias=True),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
