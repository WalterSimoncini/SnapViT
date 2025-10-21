import os
import time
import copy
import torch
import logging
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader

from src.models import load_model, ModelType
from src.models.enums import PrunableModelType
from src.utils.eval import eval_model_datasets
from src.models.prunable import load_prunable_model
from src.utils.performance import estimate_model_flops
from src.utils.logging import configure_logging, save_run_info
from src.utils.misc import get_device, seed_everything, default_cache_dir
from src.datasets import load_dataset, load_eval_datasets, DatasetSplit, DatasetType
from src.utils.models import count_parameters, count_mlp_parameters, count_attn_parameters, freeze_model


def main(args: argparse.Namespace):
    device = get_device()
    run_timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"run-{args.model_type.value}-{args.pruning_ratio}-{run_timestamp}")

    os.makedirs(output_dir)
    logging.info(f"the output directory is: {output_dir}")

    seed_everything(args.seed)

    model, transform = load_model(
        type_=args.model_type,
        cache_dir=args.cache_dir,
        device=device
    )

    num_blocks = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    hidden_dim = model.blocks[0].mlp.fc1.weight.shape[0]

    num_total_params = count_parameters(model)
    num_mlp_params = count_mlp_parameters(model=model)
    num_attn_params = count_attn_parameters(model=model)

    num_base_flops = estimate_model_flops(model=model)

    num_pruned_params = int(
        num_mlp_params * args.pruning_ratio +
        num_attn_params * args.pruning_ratio_heads
    )

    total_pruning_ratio = round(num_pruned_params / num_total_params * 100, 2)

    # Create a static clone of the original model to act as
    # the teacher for the GA optimization.
    teacher = copy.deepcopy(model).to(device)

    # Remove the teacher's head if present
    if hasattr(teacher, "head"):
        teacher.head = nn.Identity()

    if hasattr(teacher, "head_dist"):
        teacher.head_dist = nn.Identity()

    # Freeze the teacher
    freeze_model(model=teacher)

    # Freeze the student, except the mlp.fc1 and attn.qkv layers
    gradient_layers = []

    for i in range(num_blocks):
        gradient_layers.append(f"blocks.{i}.mlp.fc1")
        gradient_layers.append(f"blocks.{i}.mlp.fc2")
        gradient_layers.append(f"blocks.{i}.attn.qkv")

    freeze_model(model=model, exclusions=gradient_layers)

    logging.info(f"loaded model of type {args.model_type}")

    logging.info(f"the model has {num_blocks} blocks")
    logging.info(f"the model has {num_total_params} parameters")
    logging.info(f"the model has {num_base_flops} FLOPs")

    logging.info(f"the mlp has {num_mlp_params} parameters")
    logging.info(f"the attention blocks have {num_attn_params} parameters")

    logging.info(f"the mlp hidden dimensionality is {hidden_dim}")
    logging.info(f"the model has {num_heads} heads")

    logging.info(f"preserving at least {args.min_hidden_dim_ratio * 100:.2f}% of the mlp hidden dimension in each block")
    logging.info(f"preserving at least {args.min_head_ratio * 100:.2f}% of the heads in each block's attention layer")

    logging.info(f"pruning {num_pruned_params} parameters, i.e. {total_pruning_ratio:.2f}% of the model")

    model, pruning_transform = load_prunable_model(
        type_=args.pruning_strategy,
        backbone=model,
        backbone_transform=transform,
        device=device,
        estimation_epochs=args.num_estimation_epochs,
        min_hidden_dim_ratio=args.min_hidden_dim_ratio,
        min_head_ratio=args.min_head_ratio
    )

    # Load the evaluation datasets
    data_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }

    eval_datasets = load_eval_datasets(
        dataset_types=args.eval_datasets,
        cache_dir=args.cache_dir,
        transform=transform,
        max_knn_train_samples=args.max_knn_train_samples,
        **data_loader_args
    )

    pruning_dataset = load_dataset(
        type_=args.pruning_dataset,
        split=DatasetSplit.TRAIN,
        cache_dir=args.cache_dir,
        transform=pruning_transform,
        max_samples=args.max_samples
    )

    logging.info(f"loaded pruning dataset of type {args.pruning_dataset} with {len(pruning_dataset)} samples")
    logging.info(f"the pruning dataset has {len(pruning_dataset)} samples")

    pruning_data_loader = DataLoader(dataset=pruning_dataset, **data_loader_args)

    run_metrics = {
        "steps": [],
        "hidden_dims": [],
        "num_heads": [],
        "baseline": {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "flops": num_base_flops
        }
    }

    if args.eval_baseline:
        run_metrics["baseline"]["eval"] = eval_model_datasets(
            datasets=eval_datasets,
            model=model,
            device=device,
            speed_batch_size=args.speed_batch_size,
            speed_measurements_steps=args.speed_measurements_steps
        )

    # Uniform pruning weights
    block_weights = torch.ones(num_blocks * (num_heads + 1), device=device).float()

    start_time = time.time()

    if args.pruning_strategy not in [PrunableModelType.LAMP]:
        model.estimate_pruning_weights(pruning_data_loader)

    model.prune(
        data_loader=pruning_data_loader,
        pruning_ratio=args.pruning_ratio,
        pruning_ratio_heads=args.pruning_ratio_heads,
        block_weights=block_weights,
        estimate_pruning_weights=False
    )

    pruning_runtime_s = time.time() - start_time

    features_output_dir = os.path.join(
        output_dir,
        f"mlp-{args.pruning_ratio}-heads-{args.pruning_ratio_heads}",
        "features"
    )

    if args.save_features:
        logging.info(f"saving the features of the pruned model to {features_output_dir}")

        os.makedirs(features_output_dir, exist_ok=True)

    # Log the accuracy after each pruning step
    metrics = eval_model_datasets(
        datasets=eval_datasets,
        model=model,
        device=device,
        speed_batch_size=args.speed_batch_size,
        speed_measurements_steps=args.speed_measurements_steps,
        save_features=args.save_features,
        features_output_dir=features_output_dir
    )

    metrics["flops"] = estimate_model_flops(model=model)
    metrics["hidden_dims"] = []
    metrics["num_heads"] = []

    logging.info(f"the total pruning runtime was {pruning_runtime_s:.2f} seconds")
    logging.info(f"the pruned model has {metrics['flops']} FLOPs")

    for i in range(model.num_blocks):
        block_hidden_dim = model.model.blocks[i].mlp.fc1.weight.shape[0]
        block_hidden_dim_percent = round((block_hidden_dim / hidden_dim) * 100, 2)

        # Save the final hidden dimensionality percentage for each block
        metrics["hidden_dims"].append(block_hidden_dim_percent)

        logging.info(f"the hidden dimensionality of the {i}-th block is now {block_hidden_dim}, i.e. {block_hidden_dim_percent}% of the original")

    for i in range(num_blocks):
        block_num_heads = model.model.blocks[i].attn.num_heads
        block_num_heads_percent = round((block_num_heads / num_heads) * 100, 2)

        metrics["num_heads"].append(block_num_heads_percent)

        logging.info(f"the number of heads of the {i}-th block is now {block_num_heads}, i.e. {block_num_heads_percent}% of the original")

    run_metrics["steps"].append(metrics)

    # Save the metrics and the run configuration to disk
    save_run_info(
        output_dir=output_dir,
        metrics=run_metrics,
        args=args,
        model=model if args.save_pruned_model else None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generators")
    parser.add_argument("--cache-dir", type=str, required=False, default=default_cache_dir(), help="The cache directory for datasers and models")

    # Data loader
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="The number of workers for the data loader")

    # Dataset and model
    parser.add_argument("--pruning-dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.IN1K, help="The dataset to use to select which weights to prune")
    parser.add_argument("--pruning-dataset-split", type=DatasetSplit, choices=list(DatasetSplit), default=DatasetSplit.TEST, help="The split of the dataset to use for pruning")
    parser.add_argument("--eval-datasets", type=DatasetType, nargs='+', choices=list(DatasetType), default=[DatasetType.IN1K], help="The datasets to use for evaluation, can specify multiple datasets")
    parser.add_argument("--max-samples", type=int, default=None, help="The maximum number of samples to use in estimating the prunable weights. If larger than the dataset size, all samples are used.")
    parser.add_argument("--model-type", type=ModelType, choices=list(ModelType), default=ModelType.AUGREG_VIT_S_16_IN21K_FT_IN1K, help="The model to be pruned")

    # Pruning
    parser.add_argument("--pruning-strategy", type=PrunableModelType, choices=list(PrunableModelType), default=PrunableModelType.CROSS_ENTROPY, help="The type of pruning strategy to use")

    parser.add_argument("--pruning-ratio", type=float, default=0.2, help="The ratio of neurons to prune")
    parser.add_argument("--pruning-ratio-heads", type=float, default=0.2, help="The ratio of heads to prune")

    parser.add_argument("--num-estimation-epochs", type=int, default=1, help="The number of epochs to use for estimating the gradients")
    parser.add_argument("--min-hidden-dim-ratio", type=float, default=0.2, help="The minimum ratio of hidden neurons to keep in a block's mlp")
    parser.add_argument("--min-head-ratio", type=float, default=0.2, help="The minimum ratio of heads to keep in a block's attention layer")

    # Speed measurements
    parser.add_argument("--speed-measurements-steps", type=int, default=500, help="The number of steps to run for measuring the model speed")
    parser.add_argument("--speed-batch-size", type=int, default=16, help="The batch size to use for measuring the model speed")

    # Evaluation
    parser.add_argument("--max-knn-train-samples", type=int, default=None, help="The maximum number of samples to use for the knn evaluation. If larger than the dataset size or None, all samples are used.")
    parser.add_argument("--eval-baseline", action=argparse.BooleanOptionalAction, default=False, help="Whether to evaluate the baseline model before pruning")

    # Model saving
    parser.add_argument("--save-pruned-model", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the pruned model")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save checkpoints, metrics and logs")
    parser.add_argument("--save-features", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the features of the pruned model")

    # Genetic algorithm weights
    parser.add_argument("--ga-save-weights", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the weights of the genetic algorithm")

    configure_logging()
    main(parser.parse_args())
