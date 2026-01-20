import torch

from typing import Dict, Optional
from dataclasses import asdict, dataclass, field

from src.models.enums import MLPArchitecture, ModelType


@dataclass
class ElasticImportanceScores:
    """Pre-computed importance scores for elastic inference."""
    # Importance scores for each MLP neuron: [num_blocks, hidden_dim]
    mlp_scores: torch.Tensor

    # Importance scores for each attention head: [num_blocks, num_heads]
    head_scores: torch.Tensor

    # Architecture information
    model_type: ModelType
    mlp_architecture: MLPArchitecture

    num_blocks: int
    hidden_dim: int
    embed_dim: int

    num_heads: int
    head_dim: int

    # The minimum remaining ratio constraints for the MLP
    # hidden dimension and the attention heads.
    min_hidden_dim_keep_ratio: float = 0.05
    min_head_keep_ratio: float = 0.2

    # Optional metadata (GA config, dataset used, etc.)
    metadata: Optional[Dict] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save importance scores to disk."""
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "ElasticImportanceScores":
        """Load importance scores from disk."""
        data = torch.load(path, map_location=device, weights_only=False)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a dict for the importance scores, got {type(data).__name__}")

        # Reconstruct enums from strings
        data["model_type"] = ModelType(data["model_type"])
        data["mlp_architecture"] = MLPArchitecture(data["mlp_architecture"])

        # Make sure the metadata dictionary is present
        data.setdefault("metadata", {})

        return cls(**data)
