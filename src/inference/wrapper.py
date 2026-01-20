import torch
import torch.nn as nn

from typing import Optional, List

from src.models.enums import MLPArchitecture
from src.inference.importance_scores import ElasticImportanceScores


class ElasticViT(nn.Module):
    """
        Wrapper for a test-time prunable elastic ViT.

        NOTE: the model gradients are zeroed out before permuting
              and pruning to avoid misaligned gradients.

        Usage:
            elastic = ElasticViT(model, scores)
            elastic.prune(mlp_pruning_ratio=0.35, head_pruning_ratio=0.2)

            output = elastic(images)
    """
    def __init__(self, model: nn.Module, scores: ElasticImportanceScores):
        """
            Initialize the elastic ViT wrapper by automatically reordering
            neurons/heads by importance in descending order.

            Args:
                model:  the base ViT model to wrap (modified in-place).
                scores: the pre-computed importance scores to permute.
        """
        super().__init__()

        self.model = model
        self.scores = scores

        # Current pruning state
        self._current_mlp_keep_counts: Optional[List[int]] = None
        self._current_head_keep_counts: Optional[List[int]] = None

        self._permute()

    def _permute(self) -> None:
        """Reorder neurons and heads by importance in descending order."""
        # Zero out any gradients that would become misaligned after permuting.
        self.model.zero_grad()

        for i in range(self.scores.num_blocks):
            self._permute_block_mlp(
                block=self.model.blocks[i],
                indices=self.scores.mlp_scores[i].argsort(descending=True)
            )

            self._permute_block_attn(
                block=self.model.blocks[i],
                indices=self.scores.head_scores[i].argsort(descending=True)
            )

    def _permute_block_mlp(self, block: nn.Module, indices: torch.Tensor) -> None:
        """Reorder MLP weights by importance indices."""
        if self.scores.mlp_architecture == MLPArchitecture.STANDARD:
            # Standard MLP: fc1, fc2
            block.mlp.fc1.weight.data = block.mlp.fc1.weight.data[indices, :]
            block.mlp.fc1.bias.data = block.mlp.fc1.bias.data[indices]

            block.mlp.fc2.weight.data = block.mlp.fc2.weight.data[:, indices]
        else:
            # SwiGLU: w1, w2, w3
            block.mlp.w1.weight.data = block.mlp.w1.weight.data[indices, :]
            block.mlp.w1.bias.data = block.mlp.w1.bias.data[indices]

            block.mlp.w2.weight.data = block.mlp.w2.weight.data[indices, :]
            block.mlp.w2.bias.data = block.mlp.w2.bias.data[indices]

            block.mlp.w3.weight.data = block.mlp.w3.weight.data[:, indices]

    def _permute_block_attn(self, block: nn.Module, indices: torch.Tensor) -> None:
        """Reorder attention weights by head importance indices."""
        block.attn.proj.weight.data = block.attn.proj.weight.data.reshape(
            self.scores.embed_dim,
            self.scores.num_heads,
            self.scores.head_dim
        )[:, indices, :].reshape(self.scores.embed_dim, -1)

        block.attn.qkv.weight.data = block.attn.qkv.weight.data.view(
            3,
            self.scores.num_heads,
            self.scores.head_dim,
            self.scores.embed_dim
        )[:, indices, :, :].reshape(-1, self.scores.embed_dim)

        # Permute the QKV bias and optional bias mask (DINOv3) if present
        self._permute_block_qkv_bias(block=block, indices=indices, bias_key="bias")
        self._permute_block_qkv_bias(block=block, indices=indices, bias_key="bias_mask")

        # Some models, such as EVA ViT-G, have separate qkv biases. Permute them appropriately if they exist.
        for key in ["q_bias", "k_bias", "v_bias"]:
            if hasattr(block.attn, key) and getattr(block.attn, key) is not None:
                # Practically identical to _permute_block_qkv_bias, but these biases are
                # associated with the attention block rather than the qkv matrix.
                getattr(block.attn, key).data = getattr(block.attn, key).data.view(
                    self.scores.num_heads,
                    self.scores.head_dim
                )[indices, :].reshape(-1)

    def _permute_block_qkv_bias(self, block: nn.Module, indices: torch.Tensor, bias_key: str) -> None:
        if hasattr(block.attn.qkv, bias_key) and getattr(block.attn.qkv, bias_key) is not None:
            getattr(block.attn.qkv, bias_key).data = getattr(
                block.attn.qkv,
                bias_key
            ).data.view(
                3,
                self.scores.num_heads,
                self.scores.head_dim
            )[:, indices, :].reshape(-1)

    def prune(
        self,
        mlp_pruning_ratio: float = 0.0,
        head_pruning_ratio: float = 0.0
    ) -> "ElasticViT":
        """
        Prune the model to target sparsity using global ranking.

        Args:
            mlp_pruning_ratio: Fraction of MLP neurons to remove (0.0 to 1.0).
            head_pruning_ratio: Fraction of attention heads to remove (0.0 to 1.0).

        Returns:
            self
        """
        # Zero out any gradients that would become misaligned after pruning.
        self.model.zero_grad()

        # Compute per-block keep counts using global ranking
        mlp_keep_counts = self._compute_global_keep_counts(
            scores=self.scores.mlp_scores,
            pruning_ratio=mlp_pruning_ratio,
            min_keep_ratio=self.scores.min_hidden_dim_keep_ratio
        )

        head_keep_counts = self._compute_global_keep_counts(
            scores=self.scores.head_scores,
            pruning_ratio=head_pruning_ratio,
            min_keep_ratio=self.scores.min_head_keep_ratio
        )

        # Apply pruning (truncation since we're already permuted)
        for i in range(self.scores.num_blocks):
            block = self.model.blocks[i]
            mlp_keep = mlp_keep_counts[i]
            head_keep = head_keep_counts[i]

            self._prune_block_mlp(block, mlp_keep)
            self._prune_block_attn(block, head_keep)

        self._current_mlp_keep_counts = mlp_keep_counts
        self._current_head_keep_counts = head_keep_counts

        return self

    def _compute_global_keep_counts(
        self,
        scores: torch.Tensor,
        pruning_ratio: float,
        min_keep_ratio: float = 0.0
    ) -> List[int]:
        """
            Compute per-block keep counts using the given
            global importance scores. This should be computed
            twice, once for the heads and once for the MLP.

            Example: assuming a ViT with 4 blocks and 12 heads
            per block, a 25% uniform pruning across blocks would
            result in a [9, 9, 9, 9] keep counts.

            Args:
                scores: importance scores of shape [num_blocks, num_structures_per_block].
                pruning_ratio: fraction of units to remove in [0.0 to 1.0).
                min_keep_ratio: minimum fraction of structures to keep per block.

            Returns:
                List of keep counts per block.
        """
        if not 0 <= pruning_ratio < 1:
            raise ValueError(f"pruning ratio must be in [0, 1), got {pruning_ratio}")

        num_blocks, num_structures_per_block = scores.shape

        # Compute minimum structures to keep per block
        min_keep_per_block = int(num_structures_per_block * min_keep_ratio)

        num_total_structures = num_blocks * num_structures_per_block
        num_target_to_prune = int(num_total_structures * pruning_ratio)

        if num_target_to_prune == 0:
            # Nothing to do here.
            return [num_structures_per_block] * num_blocks

        # Clone and sort the scores to match the permuted model order.
        scores = scores.clone().sort(dim=1, descending=True).values

        # Protect the top min_keep_per_block structures per block by
        # setting their scores to the maximum value supported by the dtype.
        if min_keep_per_block > 0:
            for i in range(num_blocks):
                scores[i, :min_keep_per_block] = torch.finfo(scores.dtype).max

        # Flatten the scores and compute the threshold. To do so, sort the
        # scores in ascending order and find the threshold at the pruned
        # units count position.
        sorted_scores, _ = scores.flatten().sort()
        threshold = sorted_scores[num_target_to_prune - 1]

        return [
            int((scores[i] > threshold).sum().item()) for i in range(num_blocks)
        ]

    def _prune_block_mlp(self, block: nn.Module, keep_count: int) -> None:
        """Truncate the MLP weights to keep the top-k neurons."""
        if self.scores.mlp_architecture == MLPArchitecture.STANDARD:
            block.mlp.fc1.weight.data = block.mlp.fc1.weight.data[:keep_count, :]
            block.mlp.fc1.bias.data = block.mlp.fc1.bias.data[:keep_count]

            block.mlp.fc2.weight.data = block.mlp.fc2.weight.data[:, :keep_count]
        else:
            block.mlp.w1.weight.data = block.mlp.w1.weight.data[:keep_count, :]
            block.mlp.w1.bias.data = block.mlp.w1.bias.data[:keep_count]

            block.mlp.w2.weight.data = block.mlp.w2.weight.data[:keep_count, :]
            block.mlp.w2.bias.data = block.mlp.w2.bias.data[:keep_count]

            block.mlp.w3.weight.data = block.mlp.w3.weight.data[:, :keep_count]

    def _prune_block_attn(self, block: nn.Module, keep_count: int) -> None:
        """Truncate the attention weights to keep the top-k heads."""
        block.attn.proj.weight.data = block.attn.proj.weight.data.reshape(
            self.scores.embed_dim,
            self.scores.num_heads,
            self.scores.head_dim
        )[:, :keep_count, :].reshape(self.scores.embed_dim, -1)

        block.attn.qkv.weight.data = block.attn.qkv.weight.data.view(
            3,
            self.scores.num_heads,
            self.scores.head_dim,
            self.scores.embed_dim
        )[:, :keep_count, :, :].reshape(-1, self.scores.embed_dim)

        # Truncate the QKV bias and bias mask (DINOv3) if present.
        self._prune_block_qkv_bias(block=block, keep_count=keep_count, bias_key="bias")
        self._prune_block_qkv_bias(block=block, keep_count=keep_count, bias_key="bias_mask")

        # Some models, such as EVA ViT-G, have separate qkv biases. Prune them appropriately if they exist.
        for key in ["q_bias", "k_bias", "v_bias"]:
            if hasattr(block.attn, key) and getattr(block.attn, key) is not None:
                getattr(block.attn, key).data = getattr(block.attn, key).data.view(
                    self.scores.num_heads,
                    self.scores.head_dim
                )[:keep_count, :].reshape(-1)

        # Update the number of heads to ensure the attention operation works as expected.
        block.attn.num_heads = keep_count

    def _prune_block_qkv_bias(self, block: nn.Module, keep_count: int, bias_key: str) -> None:
        """Truncate the QKV bias to keep the top-k heads."""
        if hasattr(block.attn.qkv, bias_key) and getattr(block.attn.qkv, bias_key) is not None:
            getattr(block.attn.qkv, bias_key).data = getattr(block.attn.qkv, bias_key).data.view(
                3,
                self.scores.num_heads,
                self.scores.head_dim
            )[:, :keep_count, :].reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        return self.model.forward_features(x)

    @property
    def blocks(self) -> List[nn.Module]:
        """Access the model's transformer blocks."""
        return self.model.blocks
