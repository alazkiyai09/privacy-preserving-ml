"""
Objective functions for Gradient Boosted Decision Trees.

Implements loss functions, gradient, and Hessian computation
for binary classification (phishing detection).
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class Objective(ABC):
    """Base class for objective functions."""

    @abstractmethod
    def compute_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute first-order gradients (negative gradient of loss)."""
        pass

    @abstractmethod
    def compute_hessians(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute second-order gradients (Hessian of loss)."""
        pass

    @abstractmethod
    def loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute the loss value."""
        pass

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid function with numerical stability.

        Args:
            x: Input array

        Returns:
            Sigmoid of x: 1 / (1 + exp(-x))
        """
        # Clip for numerical stability
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))


class LogisticLoss(Objective):
    """
    Logistic loss (binary cross-entropy) for binary classification.

    Loss: -[y * log(p) + (1-y) * log(1-p)]
    Gradient: p - y
    Hessian: p * (1 - p)

    where p = sigmoid(prediction) is the predicted probability
    """

    def __init__(self):
        """Initialize logistic loss objective."""
        self.name = "binary:logistic"

    def compute_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute gradients for logistic loss.

        Args:
            predictions: Raw predictions (log-odds) from the model
            labels: True binary labels (0 or 1)

        Returns:
            Gradients: p - y where p = sigmoid(predictions)
        """
        probabilities = self.sigmoid(predictions)
        gradients = probabilities - labels
        return gradients

    def compute_hessians(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute Hessians for logistic loss.

        Args:
            predictions: Raw predictions (log-odds) from the model
            labels: True binary labels (0 or 1)

        Returns:
            Hessians: p * (1 - p) where p = sigmoid(predictions)
        """
        probabilities = self.sigmoid(predictions)
        hessians = probabilities * (1.0 - probabilities)

        # Ensure minimum Hessian for numerical stability
        hessians = np.clip(hessians, 1e-12, None)

        return hessians

    def loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute logistic loss.

        Args:
            predictions: Raw predictions (log-odds) from the model
            labels: True binary labels (0 or 1)

        Returns:
            Mean binary cross-entropy loss
        """
        probabilities = self.sigmoid(predictions)

        # Clip probabilities for numerical stability
        probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)

        # Binary cross-entropy
        loss = -np.mean(labels * np.log(probabilities) +
                       (1.0 - labels) * np.log(1.0 - probabilities))

        return loss


class SquaredErrorLoss(Objective):
    """
    Squared error loss for regression.

    Loss: 0.5 * (prediction - label)^2
    Gradient: prediction - label
    Hessian: 1.0
    """

    def __init__(self):
        """Initialize squared error loss objective."""
        self.name = "reg:squarederror"

    def compute_gradients(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute gradients for squared error loss.

        Args:
            predictions: Raw predictions from the model
            labels: True labels

        Returns:
            Gradients: predictions - labels
        """
        return predictions - labels

    def compute_hessians(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute Hessians for squared error loss.

        Args:
            predictions: Raw predictions from the model
            labels: True labels

        Returns:
            Hessians: array of ones
        """
        return np.ones_like(predictions)

    def loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute squared error loss.

        Args:
            predictions: Raw predictions from the model
            labels: True labels

        Returns:
            Mean squared error loss
        """
        return 0.5 * np.mean((predictions - labels) ** 2)


def get_objective(loss_name: str) -> Objective:
    """
    Factory function to get objective by name.

    Args:
        loss_name: Name of the loss function
                  ('binary:logistic' or 'reg:squarederror')

    Returns:
        Objective instance

    Raises:
        ValueError: If loss name is not recognized
    """
    objectives = {
        'binary:logistic': LogisticLoss,
        'reg:squarederror': SquaredErrorLoss,
        'logistic': LogisticLoss,  # Alias
        'logloss': LogisticLoss,  # Alias
        'squarederror': SquaredErrorLoss,  # Alias
        'mse': SquaredErrorLoss,  # Alias
    }

    loss_name_lower = loss_name.lower()

    if loss_name_lower not in objectives:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(objectives.keys())}")

    return objectives[loss_name_lower]()


def compute_gradients(predictions: np.ndarray, labels: np.ndarray,
                     loss: str = 'binary:logistic') -> np.ndarray:
    """
    Convenience function to compute gradients.

    Args:
        predictions: Raw predictions from the model
        labels: True labels
        loss: Loss function name

    Returns:
        Gradients
    """
    objective = get_objective(loss)
    return objective.compute_gradients(predictions, labels)


def compute_hessians(predictions: np.ndarray, labels: np.ndarray,
                     loss: str = 'binary:logistic') -> np.ndarray:
    """
    Convenience function to compute Hessians.

    Args:
        predictions: Raw predictions from the model
        labels: True labels
        loss: Loss function name

    Returns:
        Hessians
    """
    objective = get_objective(loss)
    return objective.compute_hessians(predictions, labels)
