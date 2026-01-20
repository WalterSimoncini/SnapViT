import copy
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.utils.modules import OptionalLinear
from src.models.prunable.base import PrunableModel

from .gradients import estimate_dino_gradients


class DINOPrunableModel(PrunableModel):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        estimation_epochs: int = 1,
        skip_same_view: bool = True,
        min_hidden_dim_keep_ratio: float = 0.2,
        min_head_keep_ratio: float = 0.2,
        **kwargs
    ):
        super().__init__(
            model=model,
            device=device,
            min_hidden_dim_keep_ratio=min_hidden_dim_keep_ratio,
            min_head_keep_ratio=min_head_keep_ratio
        )

        self.skip_same_view = skip_same_view
        self.estimation_epochs = estimation_epochs

        if hasattr(self.model, "head") and not isinstance(self.model.head, OptionalLinear):
            raise ValueError("The model's classification head must be an instance of OptionalLinear")

        if hasattr(self.model, "head_dist") and not isinstance(self.model.head_dist, OptionalLinear):
            raise ValueError("The model's distillation classification head (head_dist) must be an instance of OptionalLinear")

        # Use a clone of the original model as the teacher, which never gets pruned
        self.teacher = copy.deepcopy(self.model).eval()

        # Remove the teacher's heads if any        
        if hasattr(self.teacher, "head"):
            self.teacher.head = nn.Identity()

        if hasattr(self.teacher, "head_dist"):
            self.teacher.head_dist = nn.Identity()

    def estimate_pruning_weights(self, data_loader: DataLoader):
        """
            Estimate the model gradients using a DINO loss. During the estimation,
            the model's classification head is disabled, and the student and teacher
            heads are used in its place.
        """
        with self.model.head.set_enabled(enabled=False):
            if hasattr(self.model, "head_dist"):
                head_dist_enabled = self.model.head_dist.enabled
                self.model.head_dist.enabled = False
            else:
                head_dist_enabled = False

            estimate_dino_gradients(
                student=self.model,
                teacher=self.teacher,
                device=self.device,
                data_loader=data_loader,
                estimation_epochs=self.estimation_epochs,
                skip_same_view=self.skip_same_view
            )

            if hasattr(self.model, "head_dist"):
                self.model.head_dist.enabled = head_dist_enabled
