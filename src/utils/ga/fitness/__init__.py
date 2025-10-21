import torch
import logging
import torch.nn as nn

from .pca import GPUPCA
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Callable

from src.models.prunable import PrunableModel
from src.utils.models import extract_features
from src.utils.ga.optimizers import FitnessFunction
from src.utils.ga.loss import GALossType, load_ga_loss


def fit_embeddings_pca(
    embeddings: torch.Tensor,
    device: torch.device,
    num_pca_components: int = 64,
    seed: int = 42,
) -> Tuple[torch.Tensor, Optional[GPUPCA]]:
    if num_pca_components < embeddings.shape[1]:
        pca_transform = GPUPCA(
            num_components=num_pca_components,
            device=device,
            seed=seed
        ).fit(embeddings)

        logging.info(f"fit pca model using {num_pca_components} components")
        logging.info(f"the original dimensionality is: {embeddings.shape[1]}")
        logging.info(f"the pca explained variance ratio is: {pca_transform.explained_variance_ratio_.sum()}")

        compressed_embeddings = pca_transform.transform(embeddings)

        return compressed_embeddings, pca_transform
    else:
        logging.info("the pca dimensionality is less than the original dimensionality, skipping it.")

        return embeddings, None


def build_fitness_function(
    model: PrunableModel,
    teacher: nn.Module,
    pruning_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    device: torch.device,
    pruning_function: Callable,
    loss_types: List[GALossType],
    num_pca_components: int = 64,
    seed: int = 42,
    estimate_pruning_weights: bool = True
) -> FitnessFunction:
    # Load the loss functions
    loss_functions = []

    for loss_type in loss_types:
        logging.info(f"loading loss function of type: {loss_type}")

        loss_functions.append(load_ga_loss(loss_type))

    # We only compute the pruning weights once, as they
    # do not change between optimization steps while using
    # a genetic algorithm
    if estimate_pruning_weights:
        model.estimate_pruning_weights(pruning_data_loader)

    # Precompute the teacher logits, as they are static during the GA optimization
    teacher_embeddings, _ = extract_features(teacher, eval_data_loader, device, use_fp16=True, move_to_cpu=False)

    # Compress the teacher embeddings using PCA, so that we can compute the MSE
    # in a more semantic space compared to the raw embeddings
    compressed_teacher_embeddings, pca_transform = fit_embeddings_pca(
        embeddings=teacher_embeddings,
        device=device,
        num_pca_components=num_pca_components,
        seed=seed
    )

    def fitness(solution):
        total_loss = 0

        # Generate a list of pruned models for different pruning ratios.
        # We pass a deep clone of the model to avoid side effects.
        pruned_models = pruning_function(model=model, solution=solution)

        for i, prunable in enumerate(pruned_models):
            # Ensure the pruned model is on the correct device
            prunable = prunable.to(device)

            # Measure the loss using the pruned and teacher embeddings
            with prunable.model.head.set_enabled(enabled=False):
                if hasattr(prunable.model, "head_dist"):
                    with prunable.model.head_dist.set_enabled(enabled=False):
                        pruned_embeddings, _ = extract_features(prunable, eval_data_loader, device, use_fp16=True, move_to_cpu=False)
                else:
                    pruned_embeddings, _ = extract_features(prunable, eval_data_loader, device, use_fp16=True, move_to_cpu=False)

            # Compress the pruned embeddings using the same PCA transform before computing the loss
            if pca_transform is not None:
                compressed_pruned_embeddings = pca_transform.transform(pruned_embeddings)
            else:
                compressed_pruned_embeddings = pruned_embeddings

            for fn, type_ in zip(loss_functions, loss_types):
                loss = fn(
                    compressed_pruned_embeddings,
                    compressed_teacher_embeddings
                )

                logging.info(f"the {type_} loss for pruned model {i} is: {loss}")

                total_loss += loss

        # Average the loss across all pruned models
        total_loss /= len(pruned_models)

        logging.info(f"the total loss is: {total_loss}")

        return total_loss

    return fitness
