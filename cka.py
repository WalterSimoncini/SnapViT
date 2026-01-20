import logging
import argparse
import numpy as np

from src.models import load_model, ModelType
from src.utils.logging import configure_logging
from src.models.prunable import load_prunable_model
from src.datasets import load_dataset, DatasetSplit, DatasetType
from src.utils.misc import get_device, seed_everything, default_cache_dir

from src.models.enums import PrunableModelType
from src.utils.representations.cka import CKA, LinearKernel

from torch.utils.data import DataLoader


def main(args: argparse.Namespace):
    device = get_device()

    logging.info(f"loading model {args.model_type}")

    model, transform = load_model(
        type_=args.model_type,
        cache_dir=args.cache_dir,
        device=device
    )

    logging.info(f"loading prunable model of type {args.pruning_strategy}")

    # We need to load the prunable model, or better, its transform as it's important
    # to align the representations used to compute the CKA matrix to the gradients used
    # to guide the pruning process.
    model, transform = load_prunable_model(
        type_=args.pruning_strategy,
        backbone=model,
        backbone_transform=transform,
        device=device,
        # These arguments are not important, as they're not used for the CKA computation
        estimation_epochs=1,
        min_hidden_dim_keep_ratio=0.0,
        min_head_keep_ratio=0.0
    )

    logging.info(f"computing CKA matrix using the {args.dataset_split} split of the {args.dataset} with {args.num_samples} samples")

    dataset = load_dataset(
        type_=args.dataset,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        transform=transform,
        max_samples=args.num_samples
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

    cka = CKA(
        model=model,
        kernel=LinearKernel(),
        device=device
    )

    cka_matrix = cka(data_loader).cpu().numpy()

    np.save(args.output_path, cka_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42, help="The seed for the random number generator")
    parser.add_argument("--cache-dir", type=str, default=default_cache_dir(), help="The directory to cache the dataset")

    # Model arguments
    parser.add_argument("--model-type", type=ModelType, required=True, help="The model for which the CKA matrix is computed")

    # Dataset arguments
    parser.add_argument("--dataset", type=DatasetType, required=True, help="The dataset to use for the CKA matrix computation")
    parser.add_argument("--dataset-split", type=DatasetSplit, required=True, help="The split of the dataset used for the CKA matrix computation")
    parser.add_argument("--num-samples", type=int, required=True, help="The number of samples to use for the CKA matrix computation")

    # Data loader arguments
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size for the data loader")
    parser.add_argument("--num-workers", type=int, default=8, help="The number of workers for the data loader")

    # Pruning strategy arguments
    parser.add_argument("--pruning-strategy", type=PrunableModelType, choices=list(PrunableModelType), default=PrunableModelType.CROSS_ENTROPY, help="The type of pruning strategy to use")

    # Output arguments
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the CKA matrix")

    args = parser.parse_args()

    seed_everything(args.seed)
    configure_logging()

    main(args)
