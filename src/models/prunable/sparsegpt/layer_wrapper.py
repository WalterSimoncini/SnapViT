import math
import torch
import logging
import torch.nn as nn

from typing import Optional
from src.models.enums import SparseGPTCorrectionDirection, SparseGPTDampingStrategy


class SparseGPTLayerWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        damping_percentage: float = 0.01,
        mask: Optional[torch.Tensor] = None,
        bias_mask: Optional[torch.Tensor] = None,
        bias_mask_mask: Optional[torch.Tensor] = None,
        damping_strategy: SparseGPTDampingStrategy = SparseGPTDampingStrategy.MEAN,
    ):
        super().__init__()

        self.layer = layer
        self.nsamples = 0
        self.device = layer.weight.device
        self.damping_percentage = damping_percentage
        self.damping_strategy = damping_strategy
        self.nrows, self.ncolumns = layer.weight.shape
        self.hessian = torch.zeros((self.ncolumns, self.ncolumns), device=self.device)

        self.mask = mask
        self.bias_mask = bias_mask

        # The mask of the bias' mask, used in DINOv3
        self.bias_mask_mask = bias_mask_mask

    def update(self, inputs: torch.Tensor):
        in_nsamples = inputs.shape[0]

        # Merge the batch and token dimensions
        if len(inputs.shape) == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1]).t()

        # Update the Hessian iteratively
        self.hessian *= self.nsamples / (self.nsamples + in_nsamples)
        self.nsamples += in_nsamples

        inputs = math.sqrt(2 / self.nsamples) * inputs.float()

        self.hessian += inputs.matmul(inputs.t())

    def prune(
        self,
        maximum_block_size: int = 128,
        direction: SparseGPTCorrectionDirection = SparseGPTCorrectionDirection.LEFT_TO_RIGHT
    ):
        """
            Prune weights and apply correction using the SparseGPT algorithm.

            Args:
                maximum_block_size: Maximum block size for blocked processing.
                direction: the direction of the correction propagation.
        """
        weight = self.layer.weight.data.clone()
        hessian = self.hessian.clone()

        assert self.mask is not None, "prune() assumes that a mask has been set"

        mask = self.mask

        # Reverse the column order if the correction propagation direction is right-to-left
        if direction == SparseGPTCorrectionDirection.RIGHT_TO_LEFT:
            weight = weight.flip(dims=[1])
            hessian = hessian.flip(dims=[0, 1])
            mask = mask.flip(dims=[1])

        # Remove dead columns
        dead = torch.diag(hessian) == 0

        hessian[dead, dead] = 1
        weight[:, dead] = 0

        # Add damping to the Hessian diagonal
        hessian_diag = torch.diag(hessian)
        diagonal = torch.arange(self.ncolumns, device=self.device)

        if self.damping_strategy == SparseGPTDampingStrategy.PERCENTILE:
            damping = self.damping_percentage * torch.quantile(hessian_diag, 0.95)
        elif self.damping_strategy == SparseGPTDampingStrategy.MAX:
            damping = self.damping_percentage * torch.max(hessian_diag)
        elif self.damping_strategy == SparseGPTDampingStrategy.MEAN:
            damping = self.damping_percentage * torch.mean(hessian_diag)
        else:
            raise ValueError(f"invalid damping strategy: {self.damping_strategy}")

        logging.info(f"SparseGPT Hessian stats - diag mean: {hessian_diag.mean():.6f}, "
                     f"diag std: {hessian_diag.std():.6f}, diag min: {hessian_diag.min():.6f}, "
                     f"diag max: {hessian_diag.max():.6f}, damping: {damping:.6f}, "
                     f"damping_strategy: {self.damping_strategy}")

        hessian[diagonal, diagonal] += damping

        # Invert the Hessian
        inverse_hessian = torch.linalg.cholesky(hessian)
        inverse_hessian = torch.cholesky_inverse(inverse_hessian)
        inverse_hessian = torch.linalg.cholesky(inverse_hessian, upper=True)

        # Prune and correct the weights block-by-block
        for i in range(0, self.ncolumns, maximum_block_size):
            j = min(i + maximum_block_size, self.ncolumns)

            block_size = j - i

            block_weight = weight[:, i:j].clone()
            block_masked_weight = torch.zeros_like(block_weight)

            block_mask = mask[:, i:j].bool()
            block_errors = torch.zeros_like(block_weight)
            block_inverse_hessian = inverse_hessian[i:j, i:j]

            for col in range(block_size):
                column_weight = block_weight[:, col]
                column_inverse_hessian = block_inverse_hessian[col, col]

                # Update the masked weight
                column_masked_weight = column_weight.clone()
                column_masked_weight[block_mask[:, col]] = 0

                block_masked_weight[:, col] = column_masked_weight

                # Compute the error and compute the update for the remaining weights
                # on the right of the current column
                column_error = (column_weight - column_masked_weight) / column_inverse_hessian

                block_errors[:, col] = column_error
                block_weight[:, col:] -= column_error.unsqueeze(dim=1).matmul(
                    block_inverse_hessian[col, col:].unsqueeze(dim=0)
                )

            # Update the weights
            weight[:, i:j] = block_masked_weight
            weight[:, j:] -= block_errors.matmul(inverse_hessian[i:j, j:])

        # Reverse the column order if the correction propagation direction is right-to-left
        if direction == SparseGPTCorrectionDirection.RIGHT_TO_LEFT:
            weight = weight.flip(dims=[1])

        # Log the magnitude of the weight delta.
        weight_delta = (weight - self.layer.weight.data).abs()

        logging.info(f"SparseGPT weight delta - mean: {weight_delta.mean():.6f}, "
                     f"max: {weight_delta.max():.6f}, std: {weight_delta.std():.6f}")

        self.layer.weight.data = weight

        # Prune the bias and bias mask if they exist
        if self.bias_mask is not None:
            self.layer.bias.data.masked_fill_(self.bias_mask.bool(), 0.0)

        if self.bias_mask_mask is not None and hasattr(self.layer, "bias_mask"):
            self.layer.bias_mask.data.masked_fill_(self.bias_mask_mask.bool(), 0.0)
