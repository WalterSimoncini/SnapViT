import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader

from src.models.prunable.base import PrunableModel


def estimate_dino_gradients(
    student: Union[nn.Module, PrunableModel],
    teacher: Union[nn.Module, PrunableModel],
    device: torch.device,
    data_loader: DataLoader,
    estimation_epochs: int = 1,
    skip_same_view: bool = True
):
    student.zero_grad()

    for _ in range(estimation_epochs):
        for images, _ in tqdm(data_loader):
            images = images.to(device)

            # Select the global and local views (i.e. the first two crops). We first
            # embed the images using the backbone and then project them using either
            # the student or teacher heads.
            B, CR, C, H, W = images.shape

            # FIXME: What happens if we also ensure patch-level consistency?
            teacher_images = images[:, :2, :, :, :]
            teacher_images = teacher_images.reshape(-1, C, H, W)

            with torch.no_grad():
                teacher_embeddings = teacher(teacher_images)
                teacher_embeddings = teacher_embeddings.reshape(B, 2, -1)
                teacher_embeddings = nn.functional.normalize(teacher_embeddings, dim=-1, p=2)

            images = images.reshape(-1, C, H, W)

            student_embeddings = student(images).reshape(B, CR, -1)
            student_embeddings = torch.nn.functional.normalize(student_embeddings, dim=-1, p=2)

            # We gain ~1.5% by not using a temperature. Worth tuning this param
            teacher_embeddings = nn.functional.softmax(teacher_embeddings, dim=-1)
            student_embeddings = nn.functional.log_softmax(student_embeddings, dim=-1)

            # Convert the teacher and student embeddings in [CR, B, E]
            teacher_embeddings = teacher_embeddings.permute(1, 0, 2)
            student_embeddings = student_embeddings.permute(1, 0, 2)

            losses = torch.zeros(B).to(device)

            for ti, tv in enumerate(teacher_embeddings):
                for si, sv in enumerate(student_embeddings):
                    if skip_same_view and ti == si:
                        # Skip cases where teacher and student operate on the same view
                        continue

                    losses += (-tv * sv).sum(dim=-1)

            loss = losses.mean()
            loss.backward()
