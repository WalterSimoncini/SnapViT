import os

import torch
import logging
import argparse
import torch.nn as nn

from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.misc import seed_everything
from src.models import load_model, ModelType
from src.utils.models import extract_features
from src.utils.logging import configure_logging
from src.utils.checkpoints import load_pruned_checkpoint
from src.utils.misc import get_device, default_cache_dir
from src.datasets import load_dataset, DatasetSplit, DatasetType


def main(args):
    seed_everything(args.seed)

    device = get_device()

    model, transform = load_model(
        type_=args.model_type,
        cache_dir=args.cache_dir,
        device=device
    )

    if hasattr(model, "head"):
        model.head = nn.Identity()

    if hasattr(model, "head_dist"):
        model.head_dist = nn.Identity()

    if not args.eval_baseline:
        model = load_pruned_checkpoint(
            base_model=model,
            checkpoint_path=args.pruned_checkpoint_path,
            device=device
        )

        logging.info("loaded pruned model")
    else:
        logging.info("evaluating baseline model")

    model.eval()

    eval_datasets_args = {
        "cache_dir": args.cache_dir,
        "transform": transform
    }

    data_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True
    }

    logging.info(f"using dataset {args.dataset}")

    # Create the train, val and test splits
    train_dataset = load_dataset(
        type_=args.dataset,
        split=DatasetSplit.TRAIN,
        **eval_datasets_args
    )

    # Flag that indicates whether the train and validation sets should
    # be merged for the final test set evaluation.
    merge_train_valid = False

    if args.dataset in [DatasetType.TEXTURES, DatasetType.FGVC_AIRCRAFT]:
        # These datasets have their own validation split, so use it as is.
        validation_dataset = load_dataset(
            type_=args.dataset,
            split=DatasetSplit.VALID,
            **eval_datasets_args
        )
    else:
        # Create the validation split if there's no default one
        logging.warning("the validation set will be created with an 80/20 split from the training set")

        merge_train_valid = True

        if isinstance(train_dataset, Subset):
            # If the training split is already a subset (e.g. for EuroSAT), split it once more.
            # This time using an 80/20 random split, as it's not fast/easy to extract the targets array.
            train_dataset, validation_dataset = random_split(
                train_dataset,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(args.seed)
            )
        else:
            # Otherwise apply an 80/20 stratified split.
            train_targets = train_dataset.targets

            train_val_split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)
            train_indices, val_indices = list(train_val_split.split(train_targets, train_targets))[0]

            validation_dataset = Subset(train_dataset, val_indices)
            train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, **data_loader_args)
    valid_loader = DataLoader(validation_dataset, **data_loader_args)
    test_loader = DataLoader(load_dataset(
        type_=args.dataset,
        split=DatasetSplit.TEST,
        **eval_datasets_args
    ), **data_loader_args)

    # Extract features
    train_features, train_targets = extract_features(model, train_loader, device, use_fp16=True)
    valid_features, valid_targets = extract_features(model, valid_loader, device, use_fp16=True)
    test_features, test_targets = extract_features(model, test_loader, device, use_fp16=True)

    dataset = {
        "train": {
            "features": train_features.float().cpu(),
            "targets": train_targets.cpu()
        },
        "valid": {
            "features": valid_features.float().cpu(),
            "targets": valid_targets.cpu()
        },
        "test": {
            "features": test_features.float().cpu(),
            "targets": test_targets.cpu()
        },
        "merge_train_valid": merge_train_valid
    }

    # Save the dataset
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(dataset, os.path.join(args.output_dir, f"{args.dataset.value}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")
    parser.add_argument("--cache-dir", type=str, required=False, default=default_cache_dir(), help="The cache directory for datasers and models")

    # Model arguments
    parser.add_argument("--model-type", type=ModelType, choices=list(ModelType), default=ModelType.AUGREG_VIT_S_16_IN21K_FT_IN1K, help="The model to be pruned")
    parser.add_argument("--eval-baseline", action=argparse.BooleanOptionalAction, default=False, help="Whether to evaluate the base model or a pruned one")
    parser.add_argument("--pruned-checkpoint-path", type=str, required=False, help="The path to the pruned checkpoint")

    # Data arguments
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.TEXTURES, help="The dataset to extract features from")

    # Data loader
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="The number of workers for the data loader")

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True, help="The directory where to save the extracted features")

    configure_logging()
    main(parser.parse_args())
