import os
import torch
import logging
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Dict, Any, Optional, Union

from src.datasets import DatasetType
from src.models.prunable import PrunableModel
from src.utils.models import extract_features
from sklearn.model_selection import ShuffleSplit
from src.utils.classifier import LinearClassifier
from src.datasets.features import FeaturesDataset
from src.utils.performance import estimate_model_speed


def evaluate_linear(
    train_dataset: FeaturesDataset,
    valid_dataset: FeaturesDataset,
    test_dataset: FeaturesDataset,
    device: torch.device,
    merge_train_valid: bool = False,
    max_epochs: int = 300,
    patience: int = 10,
    learning_rates: List[float] = [0.0001, 0.001, 0.01],
    weight_decays: List[float] = [0.00001, 0.0001, 0.001],
    batch_size: int = 64,
    num_workers: int = 4,
    use_fp16: bool = False,
    return_classifier: bool = False
) -> Union[float, Tuple[nn.Module, float]]:
    """
        Evaluate the given features in linear probing. First, we search for the best
        learning rate and weight decay, and then we evaluate the best classifier
        on the test set.

        Parameters:
            merge_train_valid (bool): Whether to merge the train and validation sets for the final test set evaluation.
            return_classifier (bool): Whether to return the classifier along with the test accuracy.
    """
    classifier = LinearClassifier(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        merge_train_valid=merge_train_valid,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_fp16=use_fp16
    )

    if len(learning_rates) == 1 and len(weight_decays) == 1:
        # If only one weight decay and learning rate is provided,
        # we train and evaluate the classifier with those.
        head, test_accuracy = classifier.train_single(
            learning_rate=learning_rates[0],
            weight_decay=weight_decays[0],
            max_epochs=max_epochs,
            patience=patience
        )
    else:
        # Sweep the hyperparameters
        head, test_accuracy = classifier.train_sweep(
            learning_rates=learning_rates,
            weight_decays=weight_decays,
            max_epochs=max_epochs,
            patience=patience
        )

    if return_classifier:
        return head, test_accuracy

    return test_accuracy


def evaluate_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int = 20
) -> float:
    """Measure the top-1 accuracy in k-nearest neighbors using the given train and test features."""
    train_features = nn.functional.normalize(train_features.float(), dim=1, p=2).cpu().numpy()
    test_features = nn.functional.normalize(test_features.float(), dim=1, p=2).cpu().numpy()

    logging.info("training the k-nn classifier...")

    knn_classifier = KNeighborsClassifier(n_neighbors=k).fit(
        train_features,
        train_targets.cpu().numpy()
    )

    test_predictions = knn_classifier.predict(test_features)
    accuracy = (test_predictions == test_targets.cpu().numpy()).mean()

    return accuracy


def eval_model(
    model: PrunableModel,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    test_loader: DataLoader,
    device: torch.device,
    dataset_type: DatasetType,
    do_imagenet: bool = False,
    eval_linear: bool = False,
    save_features: bool = False,
    features_output_path: Optional[str] = None
) -> Tuple[float, float]:
    """
        Standard evaluation protocol for a model. Given data loaders for the
        training and test set, we extract the features and targets, and then
        evaluate the model in k-nearest neighbors and classification, if the
        model has an (ImageNet) classification head.
    """
    # Temporarily disable the model's classification head
    # (if any) during the feature extraction
    if hasattr(model, "head"):
        model.head.enabled = False

    if hasattr(model, "head_dist"):
        model.head_dist.enabled = False

    # Extract the training and test features and targets
    train_features, train_targets = extract_features(model, train_loader, device, use_fp16=True)
    test_features, test_targets = extract_features(model, test_loader, device, use_fp16=True)

    train_targets, test_targets = train_targets.cpu(), test_targets.cpu()

    if save_features and not eval_linear:
        train_features, train_targets, valid_features, valid_targets, merge_train_valid = extract_valid_features(
            model=model,
            device=device,
            train_features=train_features,
            train_targets=train_targets,
            valid_loader=valid_loader
        )

        save_features_to_disk(
            train_features=train_features,
            train_targets=train_targets,
            valid_features=valid_features,
            valid_targets=valid_targets,
            test_features=test_features,
            test_targets=test_targets,
            merge_train_valid=merge_train_valid,
            output_path=features_output_path
        )

    knn_accuracy = evaluate_knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets
    )

    if eval_linear and (not dataset_type.is_imagenet() or do_imagenet):
        # Linear evaluation
        train_features, train_targets, valid_features, valid_targets, merge_train_valid = extract_valid_features(
            model=model,
            device=device,
            train_features=train_features,
            train_targets=train_targets,
            valid_loader=valid_loader
        )

        if save_features:
            save_features_to_disk(
                train_features=train_features,
                train_targets=train_targets,
                valid_features=valid_features,
                valid_targets=valid_targets,
                test_features=test_features,
                test_targets=test_targets,
                merge_train_valid=merge_train_valid,
                output_path=features_output_path
            )

        linear_accuracy = evaluate_linear(
            train_dataset=FeaturesDataset(features=train_features, targets=train_targets),
            valid_dataset=FeaturesDataset(features=valid_features, targets=valid_targets),
            test_dataset=FeaturesDataset(features=test_features, targets=test_targets),
            merge_train_valid=valid_loader is None,
            # FIXME: Make these parameters. Currently these parameters follow the
            # DINO paper: https://github.com/facebookresearch/dino
            max_epochs=100,
            patience=100,
            learning_rates=[0.001],
            weight_decays=[0],
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            device=device,
            use_fp16=True
        )
    else:
        linear_accuracy = 0

    if hasattr(model, "head"):
        model.head.enabled = True

    if hasattr(model, "head_dist"):
        model.head_dist.enabled = True

    return linear_accuracy, knn_accuracy


def save_features_to_disk(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    merge_train_valid: bool,
    output_path: str
):
    torch.save({
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
    }, output_path)


def extract_valid_features(
    model: PrunableModel,
    device: torch.device,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_loader: Optional[DataLoader] = None,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
        Encode the validation set features using the model if it exists,
        otherwise create a validation set using an 80/20 split of the
        training set.
    """
    if valid_loader is not None:
        valid_features, valid_targets = extract_features(model, valid_loader, device, use_fp16=True)
        valid_targets = valid_targets.cpu()

        merge_train_valid = False
    else:
        # Create a validation dataset by splitting the training set using an 80/20 split
        logging.info(f"generating a validation set from the training set using random state {random_state}")

        splitter = ShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)

        indices = torch.arange(train_features.shape[0]).tolist()

        train_indices, valid_indices = list(splitter.split(indices, indices))[0]
        train_indices, valid_indices = torch.tensor(train_indices), torch.tensor(valid_indices)

        # Split the features and targets into train and validation
        valid_features = train_features[valid_indices]
        valid_targets = train_targets[valid_indices]

        train_features = train_features[train_indices]
        train_targets = train_targets[train_indices]

        merge_train_valid = True

    return train_features, train_targets, valid_features, valid_targets, merge_train_valid


def eval_model_datasets(
    datasets: List[Dict[str, Any]],
    model: PrunableModel | nn.Module,
    device: torch.device,
    speed_batch_size: int = 16,
    speed_measurements_steps: int = 500,
    do_imagenet: bool = False,
    eval_linear: bool = False,
    save_features: bool = False,
    features_output_dir: Optional[str] = None
) -> dict:
    metrics = {}
    model = model.model if isinstance(model, PrunableModel) else model

    for dataset in datasets:
        dataset_name = dataset["name"]

        if save_features:
            features_output_path = os.path.join(
                features_output_dir,
                f"{dataset_name}.pt"
            )

            os.makedirs(features_output_dir, exist_ok=True)
        else:
            features_output_path = None

        classification_acc, knn_acc = eval_model(
            model=model,
            train_loader=dataset["train"],
            valid_loader=dataset.get("valid", None),
            test_loader=dataset["test"],
            device=device,
            dataset_type=dataset["type"],
            do_imagenet=do_imagenet,
            eval_linear=eval_linear,
            save_features=save_features,
            features_output_path=features_output_path
        )

        logging.info(f"the classification accuracy on {dataset_name} is {classification_acc}")
        logging.info(f"the knn accuracy on {dataset_name} is {knn_acc}")

        metrics = metrics | {
            f"{dataset_name}_classification_accuracy": classification_acc,
            f"{dataset_name}_knn_accuracy": knn_acc
        }

    logging.info(f"estimating the model speed using {speed_measurements_steps} steps and a batch size of {speed_batch_size}")

    images_s = estimate_model_speed(
        model=model,
        device=device,
        batch_size=speed_batch_size,
        num_steps=speed_measurements_steps
    )

    logging.info(f"the model can process {images_s:.2f} images per second")

    return metrics | {
        "inference_speed": images_s
    }
