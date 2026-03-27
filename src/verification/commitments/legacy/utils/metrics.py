"""
Metrics

Performance metrics for model evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MetricsTracker:
    """
    Track metrics across federated learning rounds.
    """
    round_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def add_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, float]
    ) -> None:
        """Add metrics for a round."""
        self.round_metrics[round_num] = metrics

    def get_metric_history(
        self,
        metric_name: str
    ) -> List[Tuple[int, float]]:
        """Get history of a metric across rounds."""
        history = []
        for round_num in sorted(self.round_metrics.keys()):
            if metric_name in self.round_metrics[round_num]:
                history.append((round_num, self.round_metrics[round_num][metric_name]))
        return history

    def get_final_metrics(self) -> Dict[str, float]:
        """Get metrics from final round."""
        if not self.round_metrics:
            return {}
        final_round = max(self.round_metrics.keys())
        return self.round_metrics[final_round]


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth labels

    Returns:
        Dictionary of metrics
    """
    accuracy = np.mean(predictions == targets)

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

    # Macro averages
    avg_precision = np.mean(precision_per_class)
    avg_recall = np.mean(recall_per_class)

    # F1 score
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(avg_precision),
        "recall": float(avg_recall),
        "f1": float(f1)
    }
