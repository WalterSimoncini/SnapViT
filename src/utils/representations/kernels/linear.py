import torch

from src.utils.representations.kernels.base import CKAKernel


class LinearKernel(CKAKernel):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ x.t()
