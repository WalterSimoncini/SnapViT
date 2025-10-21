import os
import torch
import logging
import argparse

from src.utils.eval import evaluate_linear
from src.datasets.features import FeaturesDataset
from src.utils.misc import get_device, seed_everything
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

    # Run the linear probing evaluation
    classifier, linear_accuracy = evaluate_linear(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        device=get_device(),
        merge_train_valid=dataset["merge_train_valid"],
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rates=torch.tensor(args.learning_rates),
        weight_decays=torch.tensor(args.weight_decays),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=True,
        return_classifier=True
    )

    logging.info(f"the linear probing accuracy on {args.dataset_path} is {linear_accuracy}")

    # Save the metrics to disk
    save_run_info(
        output_dir=args.output_dir,
        metrics={"linear_accuracy": linear_accuracy},
        args=args
    )

    # Save the classifier to disk
    torch.save(
        classifier.state_dict(),
        os.path.join(args.output_dir, "classifier.pth")
    )

    logging.info(f"saved args and metrics to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")
    parser.add_argument("--dataset-path", type=str, required=True, help="The path to the dataset")

    # Data loading arguments
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="The number of workers for the data loader")

    # Evaluation arguments
    parser.add_argument("--max-epochs", type=int, default=50, help="The maximum number of epochs for the linear probing")
    parser.add_argument("--patience", type=int, default=5, help="The patience for linear probing")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.0001, 0.001, 0.01], help="The learning rates for linear probing")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.00001, 0.0001, 0.001], help="The weight decays for linear probing")

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True, help="The directory where to save the output")

    configure_logging()
    main(parser.parse_args())
