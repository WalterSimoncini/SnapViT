import torch
import logging
import torch.nn as nn
import torch.optim as optim

from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from src.datasets.features import FeaturesDataset


class LinearClassifier:
    def __init__(
        self,
        train_dataset: FeaturesDataset,
        valid_dataset: FeaturesDataset,
        test_dataset: FeaturesDataset,
        merge_train_valid: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
        use_fp16: bool = False,
        device: torch.device = torch.device("cpu")
    ):
        self.data_loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True
        }
    
        self.train_loader = DataLoader(train_dataset, **self.data_loader_args)
        self.valid_loader = DataLoader(valid_dataset, **self.data_loader_args)
        self.test_loader = DataLoader(test_dataset, **self.data_loader_args)

        self.device = device
        self.use_fp16 = use_fp16
        self.merge_train_valid = merge_train_valid

    def train_single(
        self,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int = 300,
        patience: int = 10
    ) -> Tuple[float]:
        """
            Train the classifier for a single learning rate and weight decay.
        """
        num_classes = self.train_loader.dataset.num_classes
        num_features = self.train_loader.dataset.num_features

        # Merge the training and validation sets if necessary
        if self.merge_train_valid:
            train_loader = self.__merge_train_valid()
            valid_loader = self.test_loader
        else:
            train_loader = self.train_loader
            valid_loader = self.valid_loader

        classifier, _ = self.__train_classifier(
            classifier=self.__init_classifier(num_features, num_classes),
            train_loader=train_loader,
            valid_loader=valid_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience
        )

        # Evaluate the classifier on the test set
        _, test_accuracy = self.__evaluate_classifier(
            classifier=classifier,
            loader=self.test_loader
        )

        return classifier, test_accuracy

    def train_sweep(
        self,
        learning_rates: List[float] = [0.0001, 0.001, 0.01],
        weight_decays: List[float] = [0.00001, 0.0001, 0.001],
        max_epochs: int = 300,
        patience: int = 10
    ) -> Tuple[float]:
        """
            Train the classifier by sweeping the learning rate and weight decay and
            evaluate the performance of the best classifier on the test set.

            Args:
                learning_rates (torch.Tensor): the learning rates to sweep.
                weight_decays (torch.Tensor): the weight decays to sweep.
                max_epochs (int): the maximum number of epochs to train the classifier.
                patience (int): the number of epochs to wait before stopping training
                                if the validation accuracy does not improve.

            Returns:
                test_accuracy (float): the accuracy of the best classifier on the test set.
        """
        num_classes = self.train_loader.dataset.num_classes
        num_features = self.train_loader.dataset.num_features

        best_val_accuracy, best_classifier = 0, None
        best_learning_rate, best_weight_decay = None, None

        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                logging.info(f"training classifier with learning rate {learning_rate} and weight decay {weight_decay}")

                classifier = self.__init_classifier(num_features, num_classes)
                classifier, val_accuracy = self.__train_classifier(
                    classifier=classifier,
                    train_loader=self.train_loader,
                    valid_loader=self.valid_loader,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    patience=patience
                )

                logging.info(f"validation accuracy for learning rate {learning_rate} and weight decay {weight_decay}: {val_accuracy}")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy, best_classifier = val_accuracy, classifier
                    best_learning_rate, best_weight_decay = learning_rate, weight_decay

                    logging.info(f"new best validation accuracy for learning rate {learning_rate} and weight decay {weight_decay}: {best_val_accuracy}")

        logging.info(f"best learning rate: {best_learning_rate}, best weight decay: {best_weight_decay}")

        # If we should merge train and valid, do so and retrain
        # the classifier using the test set as the loss monitor
        if self.merge_train_valid:
            best_classifier = self.__init_classifier(num_features, num_classes)
            best_classifier, val_accuracy = self.__train_classifier(
                classifier=best_classifier,
                train_loader=self.__merge_train_valid(),
                valid_loader=self.test_loader,
                learning_rate=best_learning_rate,
                weight_decay=best_weight_decay,
                max_epochs=max_epochs,
                patience=patience
            )

        # Evaluate the best classifier on the test set
        test_loss, test_accuracy = self.__evaluate_classifier(
            classifier=best_classifier,
            loader=self.test_loader
        )

        logging.info(f"test loss: {test_loss}, test accuracy: {test_accuracy}")

        return best_classifier, test_accuracy

    def __merge_train_valid(self) -> DataLoader:
        return DataLoader(
            self.train_loader.dataset.merge(self.valid_loader.dataset),
            **self.data_loader_args
        )

    def __train_classifier(
        self,
        classifier: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int = 300,
        patience: int = 10
    ) -> Tuple[nn.Module, float]:
        """
            Train a classifier for a given number of epochs.

            Args:
                patience (int): the number of epochs to wait before stopping training
                                if the validation accuracy does not improve.
        """
        best_val_accuracy, best_classifier_state = 0, None

        optimizer = optim.SGD(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            max_epochs,
            eta_min=0
        )

        for epoch in range(max_epochs):
            predictions, train_loss, patience_counter = [], 0, 0

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_fp16):
                for features, targets in train_loader:
                    features, targets = features.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = classifier(features)

                    loss = cross_entropy(outputs, targets)
                    loss.backward()

                    optimizer.step()

                    # Record predictions and loss
                    train_loss += loss.item()
                    predictions.append(outputs.argmax(dim=-1))

            train_loss /= len(train_loader)

            train_accuracy = self.__accuracy(
                torch.cat(predictions),
                train_loader.dataset.targets.to(self.device)
            )

            logging.info(f"epoch {epoch} -- train loss: {train_loss}, train accuracy: {train_accuracy}")

            val_loss, val_accuracy = self.__evaluate_classifier(classifier, valid_loader)

            logging.info(f"epoch {epoch} -- valid loss: {val_loss}, valid accuracy: {val_accuracy}")

            if val_accuracy > best_val_accuracy:
                patience_counter = 0
                best_val_accuracy, best_classifier_state = val_accuracy, classifier.state_dict()

                logging.info(f"epoch {epoch} -- new best valid accuracy: {best_val_accuracy}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"early stopping at epoch {epoch}")

                break

            scheduler.step()

        # Load the best classifier state
        classifier.load_state_dict(best_classifier_state)

        return classifier, best_val_accuracy

    def __evaluate_classifier(
        self,
        classifier: nn.Module,
        loader: DataLoader
    ) -> Tuple[float, float]:
        """Evaluate a classifier on the given data loader."""
        predictions, loss = [], 0

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_fp16):
            with torch.no_grad():
                for features, targets in loader:
                    features, targets = features.to(self.device), targets.to(self.device)

                    outputs = classifier(features)
                    predictions.append(outputs.argmax(dim=-1))

                    loss += cross_entropy(outputs, targets).item()

        loss /= len(loader)

        accuracy = self.__accuracy(
            torch.cat(predictions),
            loader.dataset.targets.to(self.device)
        )

        return loss, accuracy

    def __init_classifier(self, num_features: int, num_classes: int) -> nn.Module:
        classifier = nn.Linear(
            in_features=num_features,
            out_features=num_classes
        ).to(self.device)

        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()

        return classifier

    def __accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return (predictions == targets).float().mean().item()
