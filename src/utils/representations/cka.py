import torch
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Union
from collections import defaultdict
from torch.utils.data import DataLoader
from statsmodels.stats.correlation_tools import corr_nearest

from src.models.prunable import PrunableModel

from src.utils.representations.score import CKAScore
from src.utils.representations.kernels.base import CKAKernel
from src.utils.representations.kernels.linear import LinearKernel


class CKA:
    def __init__(
        self,
        model: Union[nn.Module, PrunableModel],
        kernel: CKAKernel = LinearKernel(),
        device: torch.device = torch.device("cpu"),
        batch_size: int = 16
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cka_score = CKAScore(kernel=kernel, device=device)

        if device.type == "cpu":
            logging.warning("using a CPU device for CKA, this might be slow")

    def __call__(self, data_loader: DataLoader) -> torch.Tensor:
        """
            Capture the activations of model heads and MLP blocks, and
            use them to compute the CKA score between model components.
        """
        activations = defaultdict(list)

        def build_hook(name):
            def hook(_, __, output):
                # Only capture the CLS token and average across views
                activations[name].append(
                    self.__average_activations_across_views(output[:, 0, :]).detach().cpu()
                )

            return hook

        def build_head_hook(name):
            def hook(_, input, __):
                activations[name].append(
                    self.__average_activations_across_views(input[0][:, 0, :]).detach().cpu()
                )

            return hook

        # Hooks for heads and MLP blocks
        for i, block in enumerate(self.model.blocks):
            block.attn.proj.register_forward_hook(build_head_hook(f"attn_{i}"))
            block.mlp.fc1.register_forward_hook(build_hook(f"block_{i}"))

        logging.info("collecting activations...")

        # Collect activations, either by delegating the forward passes
        # to the prunable model implementation, or by directly calling
        # the forward method of the model.
        if isinstance(self.model, PrunableModel):
            self.model.estimate_pruning_weights(data_loader, backward=False)
        else:
            with torch.no_grad():
                for images, _ in tqdm(data_loader, desc="computing activations"):
                    self.model(images.to(self.device))

        # Compute the CKA between heads and MLP blocks
        num_heads = self.model.blocks[0].attn.num_heads
        num_blocks = len(self.model.blocks)

        # The first rows/columns correspond to blocks, and the rest of the matrix
        # corresponds to the heads
        C = torch.zeros(
            num_blocks + num_blocks * num_heads,
            num_blocks + num_blocks * num_heads,
        )

        # Concatenate activations across all batches
        activations = {k: torch.cat(v, dim=0) for k, v in activations.items()}

        logging.info("computing CKA scores...")

        # Compute the CKA score between components
        for i in range(num_blocks):
            logging.info(f"processing block {i}")

            for j in range(num_blocks):
                # Compute the CKA score between blocks
                score = self.cka_score(activations[f"block_{i}"], activations[f"block_{j}"])

                C[i, j] = score
                C[j, i] = score

                # Compute the CKA score between heads
                num_samples = activations[f"attn_{i}"].shape[0]

                i_heads = activations[f"attn_{i}"].reshape(num_samples, num_heads, -1)
                j_heads = activations[f"attn_{j}"].reshape(num_samples, num_heads, -1)

                for k in range(num_heads):
                    for l in range(num_heads):
                        score = self.cka_score(i_heads[:, k, :], j_heads[:, l, :])

                        C[num_blocks + i * num_heads + k, num_blocks + j * num_heads + l] = score
                        C[num_blocks + j * num_heads + l, num_blocks + i * num_heads + k] = score

                # Compute the CKA score between block and heads
                for v in range(num_heads):
                    score = self.cka_score(activations[f"block_{i}"], j_heads[:, v, :])

                    C[i, num_blocks + j * num_heads + v] = score
                    C[num_blocks + j * num_heads + v, i] = score

        return self.__ensure_positive_definite(C)

    def __average_activations_across_views(self, activations: torch.Tensor) -> torch.Tensor:
        """Average the activations across views as needed."""
        _, embedding_dim = activations.shape

        return activations.reshape(
            self.batch_size,
            -1,
            embedding_dim
        ).mean(dim=1)

    def __ensure_positive_definite(self, C: torch.Tensor) -> torch.Tensor:
        """
            Ensure that the matrix is positive definite, and if that's
            not the case, correct it.
        """
        C = C.cpu().numpy()

        simmetric = (C == C.T).sum() == np.prod(C.shape)
        positive_eigenvalues = np.linalg.eigvals(C).min() > 0

        logging.info(f"simmetric: {simmetric}, positive_eigenvalues: {positive_eigenvalues}")

        if not simmetric or not positive_eigenvalues:
            logging.warning("the matrix is not positive definite, correcting it...")

            C = corr_nearest(C)
        else:
            logging.info("the matrix is positive definite, no correction needed")

        return torch.tensor(C, device=self.device)
