import torch

from typing import Dict, Optional

from src.models.prunable import PrunableModel
from src.inference.importance_scores import ElasticImportanceScores
from src.models.enums import MLPLayerType, MLPArchitecture, ModelType


def export_importance_scores(
    prunable_model: PrunableModel,
    block_weights: torch.Tensor,
    model_type: ModelType,
    metadata: Optional[Dict] = None
) -> ElasticImportanceScores:
    """
        Save pre-computed importance scores with the GA weights baked in.

        Args:
            prunable_model: a PrunableModel with gradient-based importance scores computed.
            block_weights:  the GA-optimized block weights tensor of shape
                            [num_blocks + num_blocks * num_heads]. The first num_blocks
                            elements indicate the MLP weights, while the rest are head
                            weights.
            model_type:     the model type (e.g., ModelType.DINO_VIT_B_16).
            metadata:       optional dictionary with additional metadata

        Returns:
            an ElasticImportanceScores object.
    """
    # Split the block weights into MLP and head weights
    mlp_weights = block_weights[:prunable_model.num_blocks].cpu()
    head_weights = block_weights[prunable_model.num_blocks:].reshape(
        prunable_model.num_blocks,
        prunable_model.default_num_heads
    ).cpu()

    # Retrieve and stack the raw importance scores
    mlp_scores = torch.stack(prunable_model._reduce_pruning_weights(), dim=0).cpu()
    head_scores = torch.stack(prunable_model._reduce_heads_pruning_weights(), dim=0).cpu()

    # Rescale the importance scores by the GA block weights
    mlp_scores = mlp_scores * mlp_weights[:, None]
    head_scores = head_scores * head_weights

    # Determine MLP architecture
    if prunable_model.target_input_mlp_layer == MLPLayerType.FC1:
        mlp_architecture = MLPArchitecture.STANDARD
    else:
        mlp_architecture = MLPArchitecture.SWIGLU

    return ElasticImportanceScores(
        mlp_scores=mlp_scores,
        head_scores=head_scores,
        mlp_architecture=mlp_architecture,
        model_type=model_type,
        num_blocks=prunable_model.num_blocks,
        hidden_dim=prunable_model.default_mlp_hidden_dim,
        num_heads=prunable_model.default_num_heads,
        embed_dim=prunable_model.embeddings_dim,
        head_dim=prunable_model.head_dim,
        min_hidden_dim_keep_ratio=prunable_model.min_hidden_dim_keep_ratio,
        min_head_keep_ratio=prunable_model.min_head_keep_ratio,
        metadata=metadata if metadata is not None else {}
    )
