import torch.nn as nn

from open_clip import create_model_from_pretrained

from .base_factory import ModelFactory
 

class SigLIP2ViT16BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        model, transform = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP2")

        model = model.visual

        model.blocks = model.trunk.blocks
        model.embed_dim = model.trunk.embed_dim
        model.patch_embed = model.trunk.patch_embed

        # Cache the transform to avoid re-creating the whole model when requested
        self.transform = transform

        return self.replace_attention(model)

    def get_transform(self, **kwargs) -> nn.Module:
        if not hasattr(self, "transform"):
            self.transform = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP2")[1]

        return self.transform


class SigLIP2ViT16GFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        model, transform = create_model_from_pretrained("hf-hub:timm/ViT-gopt-16-SigLIP2-256")

        model = model.visual

        model.blocks = model.trunk.blocks
        model.embed_dim = model.trunk.embed_dim
        model.patch_embed = model.trunk.patch_embed

        # Cache the transform to avoid re-creating the whole model when requested
        self.transform = transform

        return self.replace_attention(model)

    def get_transform(self, **kwargs) -> nn.Module:
        if not hasattr(self, "transform"):
            self.transform = create_model_from_pretrained("hf-hub:timm/ViT-gopt-16-SigLIP2-256")[1]

        return self.transform
