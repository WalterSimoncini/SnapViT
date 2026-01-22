import torch
import logging
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import List, Callable, Optional

from src.utils.models import capture_block_inputs
from src.inference.importance_scores import ElasticImportanceScores
from src.models.prunable.sparsegpt.layer_wrapper import SparseGPTLayerWrapper
from src.models.enums import MLPArchitecture, SparseGPTCorrectionDirection, SparseGPTDampingStrategy


class ElasticViT(nn.Module):
    """
        Wrapper for a test-time prunable elastic ViT.

        NOTE: the model gradients are zeroed out before permuting
              and pruning to avoid misaligned gradients.

        Usage:
            elastic = ElasticViT(
                model_factory=lambda: load_model(...),
                scores=scores,
                device=torch.device("cuda")
            )

            elastic.prune(mlp_pruning_ratio=0.35, head_pruning_ratio=0.2)

            output = elastic(images)
    """
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        scores: ElasticImportanceScores,
        device: torch.device
    ):
        """
            Initialize the elastic ViT wrapper by automatically reordering
            neurons/heads by importance in descending order.

            Args:
                model_factory: callable that returns a fresh model instance.
                scores: the pre-computed importance scores to permute.
                device: target device for the model.
        """
        super().__init__()

        self.model_factory = model_factory
        self.scores = scores
        self.device = device

        self.reset()

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

    def _is_pruned(self) -> bool:
        """Returns True if the model has been pruned."""
        return (
            self._current_mlp_keep_counts is not None or
            self._current_head_keep_counts is not None
        )

    def reset(self) -> "ElasticViT":
        """
            Reset to an unpruned state by re-instantiating the model.

            Returns:
                self for chaining.
        """
        self.model = self.model_factory().to(self.device)

        # Permute the units by importance, so that least
        # important units are at the tail of weight matrices.
        self._permute()

        # Clear the pruning state
        self._current_mlp_keep_counts = None
        self._current_head_keep_counts = None

        return self

    def prune(
        self,
        mlp_pruning_ratio: float = 0.0,
        head_pruning_ratio: float = 0.0,
        apply_correction: bool = False,
        correction_data_loader: Optional[DataLoader] = None,
        damping_percentage: float = 0.01,
        damping_strategy: SparseGPTDampingStrategy = SparseGPTDampingStrategy.MEAN,
    ) -> "ElasticViT":
        """
            Prune the model to the target sparsity using the global structure importance scores.

            Optionally applies SparseGPT weight correction to fc2 and attn.proj layers.

            Args:
                mlp_pruning_ratio: fraction of MLP neurons to remove (0.0 to 1.0).
                head_pruning_ratio: fraction of attention heads to remove (0.0 to 1.0).
                apply_correction: whether to apply SparseGPT-based weight correction.
                correction_data_loader: the data loader to use for the SparseGPT weight correction.
                damping_percentage: damping percentage for the SparseGPT Hessian regularization.
                damping_strategy: damping strategy for the SparseGPT Hessian regularization.

            Returns:
                self
        """
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

        # Check whether any of the new keep counts would exceed
        # the current counts, and if so, reset the model to an
        # unpruned state before pruning.
        if self._is_pruned():
            needs_reset = any(
                new > cur for new, cur in zip(mlp_keep_counts, self._current_mlp_keep_counts)
            ) or any(
                new > cur for new, cur in zip(head_keep_counts, self._current_head_keep_counts)
            )

            if needs_reset:
                self.reset()

        # Apply correction on the permuted model before pruning.
        if apply_correction:
            assert correction_data_loader is not None, "correction_data_loader must be provided when apply_correction is True"

            self._apply_correction(
                mlp_keep_counts=mlp_keep_counts,
                head_keep_counts=head_keep_counts,
                data_loader=correction_data_loader,
                damping_percentage=damping_percentage,
                damping_strategy=damping_strategy,
            )

        # Zero out any gradients that would become misaligned after pruning.
        self.model.zero_grad()

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
            block.attn.num_heads,
            self.scores.head_dim
        )[:, :keep_count, :].reshape(self.scores.embed_dim, -1)

        block.attn.qkv.weight.data = block.attn.qkv.weight.data.view(
            3,
            block.attn.num_heads,
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
                    block.attn.num_heads,
                    self.scores.head_dim
                )[:keep_count, :].reshape(-1)

        # Update the number of heads to ensure the attention operation works as expected.
        block.attn.num_heads = keep_count

    def _prune_block_qkv_bias(self, block: nn.Module, keep_count: int, bias_key: str) -> None:
        """Truncate the QKV bias to keep the top-k heads."""
        if hasattr(block.attn.qkv, bias_key) and getattr(block.attn.qkv, bias_key) is not None:
            getattr(block.attn.qkv, bias_key).data = getattr(block.attn.qkv, bias_key).data.view(
                3,
                block.attn.num_heads,
                self.scores.head_dim
            )[:, :keep_count, :].reshape(-1)

    def _apply_correction(
        self,
        mlp_keep_counts: List[int],
        head_keep_counts: List[int],
        data_loader: DataLoader,
        damping_percentage: float = 0.01,
        damping_strategy: SparseGPTDampingStrategy = SparseGPTDampingStrategy.MEAN,
    ) -> None:
        """
            Apply SparseGPT-based weight correction to fc2 and attn.proj
            layers (column pruning).

            Args:
                mlp_keep_counts: number of MLP neurons to keep per block.
                head_keep_counts: number of attention heads to keep per block.
                data_loader: the data loader to use for capturing activations.
                damping_percentage: damping percentage for the SparseGPT Hessian regularization.
                damping_strategy: damping strategy for the SparseGPT Hessian regularization.
        """
        inputs = capture_block_inputs(
            model=self.model,
            data_loader=data_loader,
            device=self.device,
            block_index=0,
            show_progress=True
        )

        # Compute RoPE embeddings if the model uses them (e.g., DINOv3)
        rope_sincos = None

        if hasattr(self.model, "rope_embed") and self.model.rope_embed is not None:
            H = W = self.model.patch_embed.img_size[0] // self.model.patch_embed.patch_size[0]

            rope_sincos = self.model.rope_embed(H=H, W=W)

            logging.info(f"using RoPE embeddings with H={H}, W={W}")

        for i in range(self.scores.num_blocks):
            logging.info(f"pruning block {i} with weight correction...")

            block = self.model.blocks[i]

            # On permuted models the pruned columns are at indices >= keep_count
            fc2_mask = torch.zeros_like(block.mlp.fc2.weight, dtype=torch.bool)
            fc2_mask[:, mlp_keep_counts[i]:] = True

            proj_mask = torch.zeros_like(block.attn.proj.weight, dtype=torch.bool)
            proj_mask[:, head_keep_counts[i] * self.scores.head_dim:] = True

            layers = {
                "mlp.fc2": SparseGPTLayerWrapper(layer=block.mlp.fc2, mask=fc2_mask, damping_percentage=damping_percentage, damping_strategy=damping_strategy),
                "attn.proj": SparseGPTLayerWrapper(layer=block.attn.proj, mask=proj_mask, damping_percentage=damping_percentage, damping_strategy=damping_strategy),
            }

            def build_hook(name: str):
                def hook(_, inp, __):
                    layers[name].update(inp[0])

                return hook

            handles = [
                layer.layer.register_forward_hook(build_hook(name)) for name, layer in layers.items()
            ]

            outputs = []

            with torch.no_grad():
                for sample in inputs:
                    outputs.append(block(sample.unsqueeze(0), rope_sincos))

            for handle in handles:
                handle.remove()

            for layer in layers.values():
                layer.prune(direction=SparseGPTCorrectionDirection.RIGHT_TO_LEFT)

            if i < self.scores.num_blocks - 1:
                inputs = torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        return self.model.forward_features(x)

    @property
    def blocks(self) -> List[nn.Module]:
        """Returns the model's transformer blocks."""
        return self.model.blocks

    @property
    def head(self):
        """Returns the model's classification head."""
        if hasattr(self.model, "head"):
            return self.model.head

        return None

    @property
    def head_dist(self):
        """Returns the model's distillation head."""
        if hasattr(self.model, "head_dist"):
            return self.model.head_dist

        return None
