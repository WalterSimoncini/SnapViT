import torch

from src.utils.representations.kernels.base import CKAKernel
from src.utils.representations.kernels.linear import LinearKernel


class CKAScore:
    def __init__(
        self,
        kernel: CKAKernel = LinearKernel(),
        device: torch.device = torch.device("cpu")
    ):
        self.kernel = kernel
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.to(self.device), y.to(self.device)
        k, l = self.kernel(x), self.kernel(y)

        return self.__hsic(k, l) / torch.sqrt(self.__hsic(k, k) * self.__hsic(l, l))

    def __hsic(self, k: torch.Tensor, l: torch.Tensor) -> float:
        """
            Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two matrices.
        """
        # Number of samples
        n = k.shape[0]

        # Centering matrix
        H = torch.eye(n, device=self.device) - 1 / n * torch.ones(n, n, device=self.device)

        # Compute HSIC
        return torch.trace(k @ H @ l @ H) / (n - 1) ** 2
