import torch
import logging
import argparse

from src.utils.eval import evaluate_knn
from src.utils.misc import seed_everything
from src.datasets.features import FeaturesDataset
from src.utils.logging import configure_logging, save_run_info


def main(args):
    seed_everything(args.seed)

    logging.info(f"loading the dataset from {args.dataset_path}")
    logging.info(f"the script arguments are {args}")

    dataset = torch.load(args.dataset_path, map_location=torch.device("cpu"))

    logging.info("running the linear probing evaluation")

    # Load the training, validation and test datasets
    train_dataset = FeaturesDataset(features=dataset["train"]["features"], targets=dataset["train"]["targets"])
    valid_dataset = FeaturesDataset(features=dataset["valid"]["features"], targets=dataset["valid"]["targets"])
    test_dataset = FeaturesDataset(features=dataset["test"]["features"], targets=dataset["test"]["targets"])

    if dataset["merge_train_valid"]:
        train_dataset = train_dataset.merge(valid_dataset)

    # Run the linear probing evaluation
    knn_accuracy = evaluate_knn(
        train_features=train_dataset.features,
        train_targets=train_dataset.targets,
        test_features=test_dataset.features,
        test_targets=test_dataset.targets
    )

    logging.info(f"the knn accuracy on {args.dataset_path} is {knn_accuracy}")

    # Save the metrics to disk
    save_run_info(
        output_dir=args.output_dir,
        metrics={"knn_accuracy": knn_accuracy},
        args=args
    )

    logging.info(f"saved args and metrics to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")
    parser.add_argument("--dataset-path", type=str, required=True, help="The path to the dataset")

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True, help="The directory where to save the output")

    configure_logging()
    main(parser.parse_args())
