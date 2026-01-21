from enum import Enum


class ModelType(str, Enum):
    DEIT_VIT_B_16 = "deit-vit-b-16"

    DEIT_3_VIT_H_14 = "deit-3-vit-h-14"
    DEIT_3_VIT_L_16 = "deit-3-vit-l-16"
    DEIT_3_VIT_B_16 = "deit-3-vit-b-16"
    DEIT_3_VIT_S_16 = "deit-3-vit-s-16"

    AUGREG_VIT_S_16_IN21K_FT_IN1K = "augreg-vit-s-16-in21k-ft-in1k"
    AUGREG_VIT_B_16_IN21K_FT_IN1K = "augreg-vit-b-16-in21k-ft-in1k"
    AUGREG_VIT_L_16_IN21K_FT_IN1K = "augreg-vit-l-16-in21k-ft-in1k"

    DINO_VIT_S_16 = "dino_vits16"
    DINO_VIT_B_16 = "dino_vitb16"

    DINO_V3_VIT_B_16 = "dinov3_vitb16"
    DINO_V3_VIT_H_16 = "dinov3_vith16plus"

    SIGLIP2_VIT_B_16 = "siglip2-vit-b16"
    SIGLIP2_VIT_G_16 = "siglip2-vit-g16"


class PrunableModelType(str, Enum):
    DINO = "dino"
    CROSS_ENTROPY = "cross-entropy"
    SNIP_MAGNITUDE = "snip-magnitude"
    SNIP_MAGNITUDE_DINO = "snip-magnitude-dino"
    LAMP = "lamp"
    RANDOM = "random"
    SPARSE_GPT = "sparse-gpt"
    SPARSE_GPT_DINO = "sparse-gpt-dino"


class MLPArchitecture(str, Enum):
    STANDARD = "standard"
    SWIGLU = "swiglu"


class MLPLayerType(str, Enum):
    # Standard MLP
    FC1 = "fc1"
    FC2 = "fc2"

    # SwiGLUFFN
    W1 = "w1"
    W2 = "w2"
    W3 = "w3"


class SparseGPTCorrectionDirection(str, Enum):
    """The propagation direction for the SparseGPT weight correction."""
    # Standard: corrections propagate to the right
    LEFT_TO_RIGHT = "left-to-right"

    # Reversed: corrections propagate to the left
    RIGHT_TO_LEFT = "right-to-left"
