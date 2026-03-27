"""
Model Evaluation

Utilities for evaluating federated learning models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np

from ..models.model_utils import parameters_to_ndarrays, set_model_params


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Per-class metrics
    classes = np.unique(targets)
    precision_per_class = []
    recall_per_class = []

    for cls in classes:
        true_positives = np.sum((predictions == cls) & (targets == cls))
        false_positives = np.sum((predictions == cls) & (targets != cls))
        false_negatives = np.sum((predictions != cls) & (targets == cls))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    avg_precision = np.mean(precision_per_class)
    avg_recall = np.mean(recall_per_class)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1
    }


def evaluate_global_model(
    model: nn.Module,
    parameters: list,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[Dict[str, float], nn.Module]:
    """
    Evaluate global model with given parameters.

    Args:
        model: PyTorch model
        parameters: Model parameters
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        - Metrics dictionary
        - Model with loaded parameters
    """
    # Set parameters
    set_model_params(model, parameters)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    return metrics, model
