import logging
import argparse

from torch.utils.data import DataLoader

from src.utils.eval import evaluate_knn
from src.models import load_model, ModelType
from src.utils.models import extract_features
from src.utils.models import count_parameters
from src.utils.logging import configure_logging
from src.utils.performance import estimate_model_flops
from src.inference import ElasticImportanceScores, ElasticViT
from src.datasets import load_dataset, DatasetSplit, DatasetType
from src.utils.misc import get_device, default_cache_dir, seed_everything


def main(args: argparse.Namespace):
    device = get_device()

    seed_everything(args.seed)

    logging.info(f"creating model factory and transform for {args.model_type}")

    model_factory = lambda: load_model(
        type_=args.model_type,
        cache_dir=args.cache_dir,
        device=device
    )[0]

    transform = load_model(
        type_=args.model_type,
        cache_dir=args.cache_dir,
        device=device,
        transform_only=True
    )[1]

    logging.info(f"loading importance scores from {args.importance_scores_path}")

    scores = ElasticImportanceScores.load(
        path=args.importance_scores_path,
        device=device
    )

    logging.info(f"scores metadata: {scores.metadata}")
    logging.info(f"model architecture: {scores.mlp_architecture}")

    logging.info(f"num blocks: {scores.num_blocks}")
    logging.info(f"hidden dim: {scores.hidden_dim}")
    logging.info(f"embed dim: {scores.embed_dim}")

    logging.info(f"num heads: {scores.num_heads}")
    logging.info(f"head dim: {scores.head_dim}")

    # Initialize the elastic ViT wrapper, which also
    # permutes network structures by importance.
    elastic = ElasticViT(
        model_factory=model_factory,
        scores=scores,
        device=device
    )

    # Create correction data loader if weight correction is enabled
    correction_data_loader = None

    if args.apply_correction:
        logging.info(f"loading correction dataset: {args.correction_dataset}")
        logging.info(f"using {args.correction_max_samples} samples for weight correction")

        correction_dataset = load_dataset(
            type_=args.correction_dataset,
            split=DatasetSplit.TRAIN,
            cache_dir=args.cache_dir,
            transform=transform,
            max_samples=args.correction_max_samples
        )

        correction_data_loader = DataLoader(
            dataset=correction_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Prune to target sparsity
    logging.info(f"pruning to MLP sparsity {args.mlp_pruning_ratio}, head sparsity {args.head_pruning_ratio}")

    if args.apply_correction:
        logging.info("applying weight correction to fc2 and attn.proj layers")

    elastic.prune(
        mlp_pruning_ratio=args.mlp_pruning_ratio,
        head_pruning_ratio=args.head_pruning_ratio,
        apply_correction=args.apply_correction,
        correction_data_loader=correction_data_loader
    )

    logging.info(f"pruned model has {count_parameters(elastic.model)} parameters")
    logging.info(f"pruned model has {estimate_model_flops(model=elastic)} FLOPs")

    # Disable the classification heads for feature extraction
    if elastic.head is not None:
        elastic.head.enabled = False

    if elastic.head_dist is not None:
        elastic.head_dist.enabled = False

    for dataset_type in args.eval_datasets:
        logging.info(f"evaluating on {dataset_type.value}...")

        train_dataset = load_dataset(
            type_=dataset_type,
            split=DatasetSplit.TRAIN,
            cache_dir=args.cache_dir,
            transform=transform
        )

        test_dataset = load_dataset(
            type_=dataset_type,
            split=DatasetSplit.TEST,
            cache_dir=args.cache_dir,
            transform=transform
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        logging.info("extracting train features...")

        train_features, train_targets = extract_features(
            model=elastic,
            data_loader=train_loader,
            device=device,
            use_fp16=True
        )

        logging.info("extracting test features...")

        test_features, test_targets = extract_features(
            model=elastic,
            data_loader=test_loader,
            device=device,
            use_fp16=True
        )

        knn_acc = evaluate_knn(
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets
        )

        logging.info(f"{dataset_type.value} k-NN accuracy: {knn_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42, help="The seed for the random number generator")
    parser.add_argument("--cache-dir", type=str, default=default_cache_dir(), help="The directory to cache the dataset")

    # Model and importance scores
    parser.add_argument("--model-type", type=ModelType, choices=list(ModelType), required=True, help="The base model type to load")
    parser.add_argument("--importance-scores-path", type=str, required=True, help="Path to the pre-computed importance scores")

    # Pruning configuration
    parser.add_argument("--mlp-pruning-ratio", type=float, default=0.0, help="The fraction of MLP neurons to prune")
    parser.add_argument("--head-pruning-ratio", type=float, default=0.0, help="The fraction of attention heads to prune")

    # SparseGPT correction
    parser.add_argument("--apply-correction", action=argparse.BooleanOptionalAction, help="Apply SparseGPT weight correction to fc2 and attn.proj layers")
    parser.add_argument("--correction-dataset", type=DatasetType, choices=list(DatasetType), default=None, help="The dataset to use for weight correction")
    parser.add_argument("--correction-max-samples", type=int, default=None, help="The maximum number of samples to use for weight correction. If larger than the dataset size, all samples are used.")

    # Evaluation
    parser.add_argument("--eval-datasets", type=DatasetType, nargs="+", choices=list(DatasetType), required=True, help="The datasets to use for evaluation, can specify multiple datasets")

    # Data loader arguments
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size for the data loader")
    parser.add_argument("--num-workers", type=int, default=8, help="The number of workers for the data loader")

    configure_logging()
    main(parser.parse_args())
