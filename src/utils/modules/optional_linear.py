import torch
import torch.nn as nn

from contextlib import contextmanager


class OptionalLinear(nn.Module):
    """Wrapper module that allows to temporarily disable a linear layer."""
    def __init__(self, linear: nn.Linear):
        super().__init__()

        self.linear = linear
        self.enabled = True

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.linear(x)
        else:
            return x

    @contextmanager
    def set_enabled(self, enabled: bool):
        """Context manager that temporarily changes the state of the linear layer."""
        previous_state = self.enabled

        self.enabled = enabled

        yield

        self.enabled = previous_state
