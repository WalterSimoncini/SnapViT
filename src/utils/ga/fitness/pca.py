import torch
import torch.nn as nn

from sklearn.decomposition import IncrementalPCA


class GPUPCA(nn.Module):
    """PCA Wrapper that runs the data transformation on the GPU."""
    def __init__(self, num_components: int, device: torch.device):
        self.num_components = num_components
        self.device = device

    def fit(self, embeddings: torch.Tensor) -> "GPUPCA":
        transform = IncrementalPCA(
            n_components=self.num_components,
        ).fit(embeddings.float().cpu().numpy())

        self.explained_variance_ratio_ = torch.tensor(
            transform.explained_variance_ratio_,
            device=self.device
        )

        self.T = torch.tensor(transform.components_.T, device=self.device, dtype=torch.float32)
        self.mean = torch.tensor(transform.mean_, device=self.device, dtype=torch.float32)

        self.shift = self.mean @ self.T

        return self

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings.float() @ self.T) - self.shift
