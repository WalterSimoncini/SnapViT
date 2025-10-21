import torch

from torch.nn.functional import mse_loss as mse, cosine_similarity


def mse_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> float:
    return mse(teacher_embeddings, student_embeddings).item()


def cosine_similarity_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> float:
    return 1 - cosine_similarity(teacher_embeddings, student_embeddings).mean().item()
