import os
import json
import time
import copy
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader

from src.utils.ga.loss import GALossType
from src.datasets.fast import FastDataset
from src.utils.models import deepcopy_model
from src.models import load_model, ModelType
from src.models.prunable import PrunableModel
from src.models.enums import PrunableModelType
from src.utils.eval import eval_model_datasets
from src.models.enums import MLPLayerType
from src.models.prunable import load_prunable_model
from src.utils.performance import estimate_model_flops
from src.utils.ga.fitness import build_fitness_function
from src.utils.ga import GAOptimizerType, optimize_function_ga
from src.utils.logging import configure_logging, save_run_info, save_pruned_model
from src.datasets import load_dataset, load_eval_datasets, DatasetSplit, DatasetType
from src.utils.misc import get_device, seed_everything, default_cache_dir, pad_vector_to_match
from src.utils.models import count_parameters, count_mlp_parameters, count_attn_parameters, freeze_model


def main(args: argparse.Namespace):
    device = get_device()
    run_timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"run-{args.model_type.value}-{run_timestamp}")

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

    if hasattr(model.blocks[0].mlp, MLPLayerType.FC1.value):
        hidden_dim = model.blocks[0].mlp.fc1.weight.shape[0]
    elif hasattr(model.blocks[0].mlp, MLPLayerType.W1.value):
        hidden_dim = model.blocks[0].mlp.w1.weight.shape[0]
    else:
        raise ValueError(f"Unknown MLP architecture: {model.blocks[0].mlp}")

    num_total_params = count_parameters(model)
    num_mlp_params = count_mlp_parameters(model=model)
    num_attn_params = count_attn_parameters(model=model)

    num_base_flops = estimate_model_flops(model=model)

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
        if hasattr(model.blocks[i].mlp, MLPLayerType.FC1.value):
            gradient_layers.append(f"blocks.{i}.mlp.fc1")
        elif hasattr(model.blocks[i].mlp, MLPLayerType.W1.value):
            gradient_layers.append(f"blocks.{i}.mlp.w1")

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

    for i, (mlp_ratio, head_ratio) in enumerate(zip(args.ga_mlp_pruning_ratios, args.ga_heads_pruning_ratios)):
        num_pruned_params = int(num_mlp_params * mlp_ratio + num_attn_params * head_ratio)
        total_pruning_ratio = round(num_pruned_params / num_total_params * 100, 2)

        logging.info(f"model {i}: pruning the mlp with ratio {mlp_ratio} and the heads with ratio {head_ratio}")
        logging.info(f"model {i}: pruning {num_pruned_params} parameters, i.e. {(total_pruning_ratio):.2f}% of the model")

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
        "pruned": [],
        "baseline": {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "flops": num_base_flops
        }
    }

    if args.eval_baseline:
        baseline_features_output_dir = os.path.join(output_dir, "baseline", "features")

        if args.save_features:
            logging.info(f"saving the features of the baseline model to {baseline_features_output_dir}")
            os.makedirs(baseline_features_output_dir, exist_ok=True)

        run_metrics["baseline"]["eval"] = eval_model_datasets(
            datasets=eval_datasets,
            model=model,
            device=device,
            speed_batch_size=args.speed_batch_size,
            speed_measurements_steps=args.speed_measurements_steps,
            save_features=args.save_features,
            features_output_dir=baseline_features_output_dir
        )

        baseline_metrics_dir = os.path.join(output_dir, "baseline")

        os.makedirs(baseline_metrics_dir, exist_ok=True)

        with open(os.path.join(baseline_metrics_dir, "metrics.json"), "w") as f:
            json.dump(run_metrics["baseline"]["eval"], f)

    # Load a dataset to optimize the block weights using a genetic algorithm
    ga_optimization_dataset = load_dataset(
        type_=args.ga_optimization_dataset,
        split=args.ga_optimization_dataset_split,
        cache_dir=args.cache_dir,
        transform=transform,
        max_samples=args.ga_max_eval_samples,
        keep_in_memory=args.ga_keep_in_memory
    )

    if args.use_fast_dataset:
        logging.info("precomputing the genetic algorithm optimization dataset transform")

        ga_optimization_dataset = FastDataset(ga_optimization_dataset)
        ga_optimization_dataset.precompute()

    logging.info(f"loaded the genetic algorithm optimization dataset of type {args.ga_optimization_dataset} with {len(ga_optimization_dataset)} samples")

    ga_optimization_data_loader_args = data_loader_args | { "batch_size": args.ga_batch_size }
    ga_optimization_data_loader = DataLoader(
        dataset=ga_optimization_dataset,
        **ga_optimization_data_loader_args
    )

    if args.correction_dataset is not None:
        correction_dataset = load_dataset(
            type_=args.correction_dataset,
            split=DatasetSplit.TRAIN,
            cache_dir=args.cache_dir,
            transform=transform,
            max_samples=args.correction_max_samples
        )

        correction_data_loader = DataLoader(
            dataset=correction_dataset,
            **data_loader_args | { "batch_size": args.correction_batch_size }
        )
    elif args.correction_dataset is None and args.pruning_strategy in [PrunableModelType.SPARSE_GPT]:
        raise ValueError("sparse-gpt applies weight correction, but no correction dataset was provided")
    else:
        correction_data_loader = None

    block_weights = torch.cat([
        torch.linspace(args.ga_init_start_value, args.ga_init_end_value, num_blocks),
        torch.linspace(args.ga_init_start_value, args.ga_init_end_value, num_blocks).unsqueeze(dim=1).repeat(1, num_heads).reshape(-1)
    ], dim=0).numpy()

    pruning_runtime_s = 0

    logging.info("optimizing the block weights...")

    # Callable to perform pruning
    def pruning_function(model: PrunableModel, solution: np.ndarray) -> List[PrunableModel]:
        models = []

        for mlp_ratio, head_ratio in zip(args.ga_mlp_pruning_ratios, args.ga_heads_pruning_ratios):
            prunable = deepcopy_model(model)
            prunable.prune(
                data_loader=pruning_data_loader,
                pruning_ratio=mlp_ratio,
                pruning_ratio_heads=head_ratio,
                # Take the absolute value of the solution as some entries might be negative
                block_weights=torch.tensor(solution.astype(np.float32)).abs(),
                estimate_pruning_weights=False
            )

            models.append(prunable)

        return models

    fitness_function = build_fitness_function(
        model=model,
        teacher=teacher,
        pruning_data_loader=pruning_data_loader,
        eval_data_loader=ga_optimization_data_loader,
        device=device,
        pruning_function=pruning_function,
        loss_types=args.ga_loss_types,
        num_pca_components=args.ga_num_pca_components
    )

    if args.ga_init_covariance_matrix_path is not None:
        covariance_init = np.load(args.ga_init_covariance_matrix_path)
        run_metrics["init_covariance"] = covariance_init.tolist()
    else:
        covariance_init = None
        run_metrics["init_covariance"] = None

    step_start_time = time.time()

    if args.ga_max_function_evaluations > 0:
        block_weights, block_weights_fitness, info = optimize_function_ga(
            # Generic arguments
            type_=args.ga_optimizer,
            fitness_function=fitness_function,
            ndim_problem=num_blocks + num_heads * num_blocks,
            max_function_evaluations=args.ga_max_function_evaluations,
            seed=args.seed,
            starting_point=block_weights,
            covariance_init=covariance_init,
            xnes_num_individuals=args.ga_xnes_num_individuals
        )
    else:
        block_weights, block_weights_fitness, info = block_weights, 0, {}

    # Ensure the model is on the correct device
    model = model.to(device)

    run_metrics["fitness_scores"] = info.get("fitness_scores")
    run_metrics["final_covariance"] = info.get("covariance")

    # Take the absolute value of the solution as some entries might be negative
    block_weights = torch.tensor(block_weights).abs()

    # Save the weights computed by the genetic algorithm to disk as needed
    if args.ga_save_weights:
        # Compute the base weights
        mlp_weights = model._reduce_pruning_weights()
        head_weights = model._reduce_heads_pruning_weights()

        # Rescale the weights
        scaling_block_weights = torch.tensor(block_weights).to(device)

        for i in range(num_blocks):
            mlp_weights[i] = mlp_weights[i] * scaling_block_weights[i]
            head_weights[i] = head_weights[i] * scaling_block_weights[i]

        # Save them to disk
        mlp_weights = torch.cat([
            pad_vector_to_match(
                x,
                target_length=hidden_dim,
                value=0
            ).unsqueeze(dim=0) for x in mlp_weights
        ]).cpu()

        head_weights = torch.cat([
            pad_vector_to_match(
                x,
                target_length=num_heads,
                value=0
            ).unsqueeze(dim=0) for x in head_weights
        ]).cpu()

        weights_path = os.path.join(output_dir, "ga-weights.pt")

        torch.save({
            "mlp": mlp_weights,
            "head": head_weights
        }, weights_path)

    logging.info(f"the optimized block weights (fitness {block_weights_fitness}) are {block_weights.tolist()}")

    pruning_runtime_s += time.time() - step_start_time

    logging.info(f"the total optimization runtime was {pruning_runtime_s:.2f} seconds")

    for mlp_ratio, head_ratio in zip(args.eval_mlp_pruning_ratios, args.eval_heads_pruning_ratios):
        logging.info(f"pruning the mlp with ratio {mlp_ratio} and the heads with ratio {head_ratio}")

        prunable = deepcopy_model(model)
        prunable.prune(
            data_loader=pruning_data_loader,
            pruning_ratio=mlp_ratio,
            pruning_ratio_heads=head_ratio,
            block_weights=torch.tensor(block_weights).to(device),
            estimate_pruning_weights=False,
            apply_correction=True,
            correction_data_loader=correction_data_loader
        )

        # Save the model metrics
        model_metrics = {
            "num_heads": [],
            "hidden_dims": [],
            "flops": estimate_model_flops(model=prunable),
            "ratios": {
                "mlp": mlp_ratio,
                "heads": head_ratio
            }
        }

        logging.info(f"this model has {count_parameters(prunable.model)} parameters")
        logging.info(f"this model has {estimate_model_flops(model=prunable)} FLOPs")

        pruned_model_output_dir = os.path.join(output_dir, f"mlp-{mlp_ratio}-heads-{head_ratio}")
        features_output_dir = os.path.join(pruned_model_output_dir, "pruned", "features")

        logging.info(f"saving the pruned model to {pruned_model_output_dir}")

        if args.save_features:
            logging.info(f"saving the features of the pruned model to {features_output_dir}")
            os.makedirs(features_output_dir, exist_ok=True)

        save_pruned_model(output_dir=pruned_model_output_dir, model=prunable)

        # Log the accuracy after each pruning step
        model_metrics["eval"] = eval_model_datasets(
            datasets=eval_datasets,
            model=prunable,
            device=device,
            speed_batch_size=args.speed_batch_size,
            speed_measurements_steps=args.speed_measurements_steps,
            save_features=args.save_features,
            features_output_dir=features_output_dir
        )

        for i in range(num_blocks):
            if hasattr(prunable.model.blocks[i].mlp, MLPLayerType.FC1.value):
                block_hidden_dim = prunable.model.blocks[i].mlp.fc1.weight.shape[0]
            elif hasattr(prunable.model.blocks[i].mlp, MLPLayerType.W1.value):
                block_hidden_dim = prunable.model.blocks[i].mlp.w1.weight.shape[0]
            else:
                raise ValueError(f"Unknown MLP architecture: {prunable.model.blocks[i].mlp}")

            block_hidden_dim_percent = round((block_hidden_dim / hidden_dim) * 100, 2)

            # Save the final hidden dimensionality percentage for each block
            model_metrics["hidden_dims"].append(block_hidden_dim_percent)

            logging.info(f"the hidden dimensionality of the {i}-th block is now {block_hidden_dim}, i.e. {block_hidden_dim_percent}% of the original")

        for i in range(num_blocks):
            block_num_heads = prunable.model.blocks[i].attn.num_heads
            block_num_heads_percent = round((block_num_heads / num_heads) * 100, 2)

            model_metrics["num_heads"].append(block_num_heads_percent)

            logging.info(f"the number of heads of the {i}-th block is now {block_num_heads}, i.e. {block_num_heads_percent}% of the original")

        run_metrics["pruned"].append(model_metrics)

    # Save the metrics and the run configuration to disk
    save_run_info(
        output_dir=output_dir,
        metrics=run_metrics,
        args=args,
        model=None
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
    parser.add_argument("--use-fast-dataset", action=argparse.BooleanOptionalAction, default=False, help="Whether to use the fast dataset")

    # Pruning
    parser.add_argument("--pruning-strategy", type=PrunableModelType, choices=list(PrunableModelType), default=PrunableModelType.CROSS_ENTROPY, help="The type of pruning strategy to use")

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

    # Genetic algorithm optimization
    parser.add_argument("--ga-max-eval-samples", type=int, default=512, help="The maximum number of samples for evaluating the fitness of the genetic algorithm solutions")
    parser.add_argument("--ga-optimization-dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.IN1K, help="The dataset to use for the genetic algorithm optimization")
    parser.add_argument("--ga-optimization-dataset-split", type=DatasetSplit, choices=list(DatasetSplit), default=DatasetSplit.TEST, help="The split of the dataset to use for the genetic algorithm optimization")
    parser.add_argument("--ga-keep-in-memory", action=argparse.BooleanOptionalAction, default=False, help="Whether to load the genetic algorithm optimization dataset in memory")
    parser.add_argument("--ga-batch-size", type=int, default=16, help="The batch size to use for the genetic algorithm optimization")
    parser.add_argument("--ga-optimizer", type=GAOptimizerType, choices=list(GAOptimizerType), default=GAOptimizerType.XNES, help="The optimizer to use for the genetic algorithm")
    parser.add_argument("--ga-max-function-evaluations", type=int, default=250, help="The maximum number of function evaluations for the genetic algorithm")
    parser.add_argument("--ga-num-pca-components", type=int, default=64, help="The number of PCA components to use for the genetic algorithm")
    parser.add_argument("--ga-xnes-num-individuals", type=int, default=None, help="The number of individuals to use for the xnes genetic algorithm")
    parser.add_argument("--ga-loss-types", type=GALossType, nargs='+', choices=list(GALossType), default=[GALossType.COSINE_SIMILARITY], help="The loss types to use for the genetic algorithm")

    # Genetic algorithm weights
    parser.add_argument("--ga-save-weights", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the weights of the genetic algorithm")

    # Genetic algorithm initialization
    parser.add_argument("--ga-init-start-value", type=float, default=1.2, help="The initial genetic algorithm coefficient for the first model block")
    parser.add_argument("--ga-init-end-value", type=float, default=0.8, help="The initial genetic algorithm coefficient for the last model block")
    parser.add_argument("--ga-init-covariance-matrix-path", type=str, default=None, help="The covariance matrix to use for the genetic algorithm")

    parser.add_argument("--ga-mlp-pruning-ratios", type=float, nargs="+", default=[0.25, 0.45, 0.65], help="The ratios to use for the mlp pruning in the genetic algorithm")
    parser.add_argument("--ga-heads-pruning-ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="The ratios to use for the heads pruning in the genetic algorithm")

    parser.add_argument("--eval-mlp-pruning-ratios", type=float, nargs="+", default=[], help="The ratios to use for the mlp pruning in the genetic algorithm")
    parser.add_argument("--eval-heads-pruning-ratios", type=float, nargs="+", default=[], help="The ratios to use for the heads pruning in the genetic algorithm")

    # Weight correction
    parser.add_argument("--correction-dataset", type=DatasetType, choices=list(DatasetType), default=None, help="The dataset to use for weight correction")
    parser.add_argument("--correction-max-samples", type=int, default=None, help="The maximum number of samples to use for weight correction. If larger than the dataset size, all samples are used.")
    parser.add_argument("--correction-batch-size", type=int, default=16, help="The batch size to use for weight correction")

    configure_logging()
    main(parser.parse_args())
