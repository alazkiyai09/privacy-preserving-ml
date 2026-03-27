"""
Label holder implementation.

The label holder computes gradients and Hessians and distributes them
to participating parties.
"""

import numpy as np
from typing import Tuple, Optional

from core.objective import Objective, get_objective


class LabelHolder:
    """
    Holds labels and computes gradients/Hessians.

    In vertical federated learning, one party (the label holder) has
    the labels while other parties have only features.
    """

    def __init__(self,
                 labels: np.ndarray,
                 loss: str = 'binary:logistic'):
        """
        Initialize label holder.

        Args:
            labels: Training labels (n_samples,)
            loss: Loss function
        """
        self.labels = labels
        self.n_samples = len(labels)

        # Initialize objective
        self.objective: Objective = get_objective(loss)

        # Prediction state
        self.predictions: Optional[np.ndarray] = None

    def set_predictions(self, predictions: np.ndarray) -> None:
        """
        Set current predictions.

        Args:
            predictions: Current model predictions (n_samples,)
        """
        self.predictions = predictions

    def compute_gradients_hessians(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and Hessians.

        Returns:
            Tuple of (gradients, hessians)
        """
        if self.predictions is None:
            raise ValueError("Predictions not set. Call set_predictions first.")

        gradients = self.objective.compute_gradients(self.predictions, self.labels)
        hessians = self.objective.compute_hessians(self.predictions, self.labels)

        return gradients, hessians

    def compute_loss(self) -> float:
        """
        Compute loss.

        Returns:
            Loss value
        """
        if self.predictions is None:
            raise ValueError("Predictions not set. Call set_predictions first.")

        return self.objective.loss(self.predictions, self.labels)

    def initialize_base_score(self) -> float:
        """
        Compute initial base score (prior).

        Returns:
            Base score
        """
        if hasattr(self.objective, 'name') and 'logistic' in self.objective.name:
            # For logistic loss, use log-odds
            pos_ratio = np.mean(self.labels)
            return np.log(pos_ratio / (1 - pos_ratio + 1e-12))
        else:
            # For regression, use mean
            return float(np.mean(self.labels))

    def get_labels(self) -> np.ndarray:
        """Get labels."""
        return self.labels

    def get_n_samples(self) -> int:
        """Get number of samples."""
        return self.n_samples
