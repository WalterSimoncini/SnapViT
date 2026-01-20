from enum import Enum


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DatasetType(str, Enum):
    IN1K = "imagenet-1k"
    DIFFUSION_DB_LARGE_FIRST_10K = "diffusiondb-large-first-10k"
    SHADERS_21K = "shaders-21k"
    TEXTURES = "textures"
    FGVC_AIRCRAFT = "aircraft"
    OXFORD_IIT_PETS = "pets"
    EUROSAT = "eurosat"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    MERGED = "merged"

    def is_imagenet(self):
        return self == DatasetType.IN1K
