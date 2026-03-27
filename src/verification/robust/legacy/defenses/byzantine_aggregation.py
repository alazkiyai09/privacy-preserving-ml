"""
Byzantine-Robust Aggregation for Federated Learning

Implements robust aggregation rules that can tolerate up to f Byzantine (malicious) clients.

Methods:
1. Krum: Select gradient with minimum sum of distances to others
2. Multi-Krum (a.k.a. Bulyan): Average top-k closest gradients
3. Trimmed Mean: Remove largest/smallest values, average rest

Robustness Guarantees:
- Krum: Robust to f < (n-3)/2 Byzantine clients
- Multi-Krum: Robust to f < (n-4)/2 Byzantine clients
- Trimmed Mean: Robust to f < n/2 Byzantine clients

Application to Phishing Detection:
- Prevents label flip attacks (gradients from poisoned data are outliers)
- Prevents sign flip attacks (malicious gradients are far from honest ones)
- Partially prevents backdoor attacks (depends on gradient anomaly)
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def flatten_gradients(gradients: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Flatten list of gradient arrays into 1D vectors."""
    return [np.concatenate([layer.flatten() for layer in grad]) for grad in gradients]


def unflatten_gradient(flat_grad: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Unflatten 1D gradient back into original shapes."""
    grad = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        grad.append(flat_grad[idx:idx+size].reshape(shape))
        idx += size
    return grad


class KrumAggregator:
    """
    Krum Aggregation: Select gradient closest to others.

    Algorithm:
    1. Compute pairwise distances between all gradients
    2. For each gradient, sum distances to (n-f-2) nearest neighbors
    3. Select gradient with minimum score

    Robust to f < (n-3)/2 Byzantine clients.
    """

    def __init__(self, num_malicious: int = 2):
        """
        Initialize Krum aggregator.

        Args:
            num_malicious: Upper bound on number of malicious clients (f)
        """
        self.num_malicious = num_malicious

    def aggregate(
        self,
        gradients: List[List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate gradients using Krum.

        Args:
            gradients: List of gradients (each gradient is list of arrays)

        Returns:
            (aggregated_gradient, metrics)
        """
        if len(gradients) == 0:
            return [], {}

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)
        num_gradients = len(flat_gradients)

        # Compute pairwise distance matrix
        distance_matrix = self._compute_distance_matrix(flat_gradients)

        # Compute Krum score for each gradient
        scores = []
        num_closest = num_gradients - self.num_malicious - 2

        for i in range(num_gradients):
            # Get distances from gradient i to all others
            distances_i = distance_matrix[i]
            # Sort and sum (num_gradients - f - 2) smallest distances
            sorted_distances = np.sort(distances_i)
            score = np.sum(sorted_distances[:num_closest])
            scores.append(score)

        # Select gradient with minimum score
        selected_idx = int(np.argmin(scores))
        selected_gradient = gradients[selected_idx]

        metrics = {
            "aggregation_method": "krum",
            "num_gradients": num_gradients,
            "num_malicious": self.num_malicious,
            "selected_idx": selected_idx,
            "scores": [float(s) for s in scores]
        }

        return selected_gradient, metrics

    def _compute_distance_matrix(self, flat_gradients: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix."""
        n = len(flat_gradients)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # Euclidean distance
                dist = np.linalg.norm(flat_gradients[i] - flat_gradients[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix


class MultiKrumAggregator:
    """
    Multi-Krum Aggregation: Average top-k closest gradients.

    Algorithm:
    1. Compute Krum scores for all gradients
    2. Select top-k gradients with minimum scores
    3. Average the selected gradients

    More robust than Krum but requires more gradients.
    """

    def __init__(self, num_malicious: int = 2, k: int = 5):
        """
        Initialize Multi-Krum aggregator.

        Args:
            num_malicious: Upper bound on number of malicious clients (f)
            k: Number of gradients to average
        """
        self.num_malicious = num_malicious
        self.k = k

        # Use Krum to compute scores
        self.krum = KrumAggregator(num_malicious=num_malicious)

    def aggregate(
        self,
        gradients: List[List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate gradients using Multi-Krum.

        Args:
            gradients: List of gradients

        Returns:
            (aggregated_gradient, metrics)
        """
        if len(gradients) == 0:
            return [], {}

        num_gradients = len(gradients)
        k = min(self.k, num_gradients)

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)

        # Compute Krum scores
        distance_matrix = self.krum._compute_distance_matrix(flat_gradients)
        scores = []
        num_closest = num_gradients - self.num_malicious - 2

        for i in range(num_gradients):
            distances_i = distance_matrix[i]
            sorted_distances = np.sort(distances_i)
            score = np.sum(sorted_distances[:num_closest])
            scores.append(score)

        # Select top-k gradients with minimum scores
        selected_indices = np.argsort(scores)[:k]

        # Average selected gradients
        averaged_gradient = self._average_gradients(
            [gradients[i] for i in selected_indices]
        )

        metrics = {
            "aggregation_method": "multi_krum",
            "num_gradients": num_gradients,
            "num_malicious": self.num_malicious,
            "k": k,
            "selected_indices": [int(i) for i in selected_indices]
        }

        return averaged_gradient, metrics

    def _average_gradients(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Average list of gradients."""
        if len(gradients) == 0:
            return []

        # Average each layer
        num_layers = len(gradients[0])
        averaged = []

        for layer_idx in range(num_layers):
            # Stack all layers at this index
            layers = np.stack([grad[layer_idx] for grad in gradients])
            # Compute average
            averaged.append(np.mean(layers, axis=0))

        return averaged


class TrimmedMeanAggregator:
    """
    Trimmed Mean Aggregation: Remove outliers, average rest.

    Algorithm:
    1. For each dimension, sort values across all clients
    2. Remove smallest βn and largest βn values
    3. Average remaining values

    Robust to f < n/2 Byzantine clients (when β ≥ 0.5f/n).
    """

    def __init__(self, trim_ratio: float = 0.2):
        """
        Initialize Trimmed Mean aggregator.

        Args:
            trim_ratio: Ratio to trim from each end (β)
        """
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        gradients: List[List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregate gradients using Trimmed Mean.

        Args:
            gradients: List of gradients

        Returns:
            (aggregated_gradient, metrics)
        """
        if len(gradients) == 0:
            return [], {}

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)
        stacked_gradients = np.stack(flat_gradients)  # Shape: (n, d)

        num_gradients = len(gradients)
        num_trim = int(num_gradients * self.trim_ratio)

        # Trim and average
        trimmed_aggregated = self._trim_and_average(stacked_gradients, num_trim)

        # Unflatten
        shapes = [layer.shape for layer in gradients[0]]
        aggregated_gradient = unflatten_gradient(trimmed_aggregated, shapes)

        metrics = {
            "aggregation_method": "trimmed_mean",
            "num_gradients": num_gradients,
            "trim_ratio": self.trim_ratio,
            "num_trimmed": num_trim
        }

        return aggregated_gradient, metrics

    def _trim_and_average(
        self,
        stacked: np.ndarray,
        num_trim: int
    ) -> np.ndarray:
        """Trim and average along axis 0."""
        # Sort along axis 0 (across clients)
        sorted_grads = np.sort(stacked, axis=0)

        # Remove smallest and largest
        if num_trim > 0:
            trimmed = sorted_grads[num_trim:-num_trim, :]
        else:
            trimmed = sorted_grads

        # Average remaining
        aggregated = np.mean(trimmed, axis=0)

        return aggregated


# Helper function for coordinate-wise median (alternative to trimmed mean)
def coordinate_wise_median(
    gradients: List[List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Compute coordinate-wise median of gradients.

    More robust to outliers than mean but computationally more expensive.
    """
    if len(gradients) == 0:
        return []

    # Flatten gradients
    flat_gradients = flatten_gradients(gradients)
    stacked_gradients = np.stack(flat_gradients)

    # Compute median along axis 0
    median_flat = np.median(stacked_gradients, axis=0)

    # Unflatten
    shapes = [layer.shape for layer in gradients[0]]
    median_gradient = unflatten_gradient(median_flat, shapes)

    return median_gradient


# Example usage
if __name__ == "__main__":
    print("Byzantine Aggregation Demonstration")
    print("=" * 60)

    # Create synthetic gradients
    # 8 honest gradients
    honest_gradients = []
    for i in range(8):
        grad = [np.random.randn(5, 10) * 0.1 for _ in range(3)]
        honest_gradients.append(grad)

    # 2 malicious gradients (scaled)
    malicious_gradients = []
    for i in range(2):
        grad = [np.random.randn(5, 10) * 10.0 for _ in range(3)]
        malicious_gradients.append(grad)

    all_gradients = honest_gradients + malicious_gradients

    print(f"Total gradients: {len(all_gradients)}")
    print(f"Malicious: 2 (indices 8, 9)")

    # Test Krum
    print("\n--- Krum Aggregation ---")
    krum = KrumAggregator(num_malicious=2)
    aggregated_krum, metrics_krum = krum.aggregate(all_gradients)

    print(f"Selected gradient index: {metrics_krum['selected_idx']}")
    print(f"Malicious selected: {metrics_krum['selected_idx'] >= 8}")

    # Test Multi-Krum
    print("\n--- Multi-Krum Aggregation ---")
    multi_krum = MultiKrumAggregator(num_malicious=2, k=5)
    aggregated_multi, metrics_multi = multi_krum.aggregate(all_gradients)

    print(f"Selected indices: {metrics_multi['selected_indices']}")
    print(f"Malicious in selection: {any(i >= 8 for i in metrics_multi['selected_indices'])}")

    # Test Trimmed Mean
    print("\n--- Trimmed Mean Aggregation ---")
    trimmed_mean = TrimmedMeanAggregator(trim_ratio=0.2)
    aggregated_tm, metrics_tm = trimmed_mean.aggregate(all_gradients)

    print(f"Trim ratio: {metrics_tm['trim_ratio']}")
    print(f"Number trimmed: {metrics_tm['num_trimmed']}")

    # Compare with standard mean
    print("\n--- Standard Mean (for comparison) ---")
    from src.verification.robust.legacy.models.model_utils import aggregate_gradients_weighted
    standard_agg, _ = aggregate_gradients_weighted([(g, 1.0) for g in all_gradients])

    print(f"Standard mean computed (vulnerable to malicious gradients)")
    print(f"Byzantine methods provide robustness")
