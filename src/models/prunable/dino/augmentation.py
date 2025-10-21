"""Modified from https://github.com/facebookresearch/dino"""
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf

from PIL import Image
from typing import Tuple
from torchvision.transforms.v2.functional import to_image


class DINODataAgumentation(nn.Module):
    """
        The DINO data augmentation. By default, only random crops
        are used an style augmentations are disabled.
    """
    def __init__(
        self,
        global_crops_scale: Tuple[int, int] = (0.25, 1.0),
        local_crops_scale: Tuple[int, int] = (0.05, 0.25),
        local_crops_number: int = 10,
        crops_size: int = 224
    ):
        super().__init__()

        self.local_crops_number = local_crops_number

        self.global_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=global_crops_scale),
            tf.ToDtype(torch.float32, scale=True),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.local_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=local_crops_scale),
            tf.ToDtype(torch.float32, scale=True),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image: Image):
        # Convert the image to a tensor and create a batched input for the data augmentation
        images = to_image(image).unsqueeze(dim=0).repeat(self.local_crops_number + 2, 1, 1, 1)

        # Generate local and global crops
        return torch.cat([
            self.global_crop(images[:2, :, :, :]),
            self.local_crop(images[2:, :, :, :])
        ], dim=0)
