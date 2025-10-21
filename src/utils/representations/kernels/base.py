import torch

from abc import ABC, abstractmethod


class CKAKernel(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
