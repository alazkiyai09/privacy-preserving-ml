"""
Differential Privacy mechanisms for privacy-preserving GBDT.

Implements Laplace and Gaussian mechanisms for adding calibrated noise
to gradients, histograms, and split information.
"""

import numpy as np
from typing import Union, Optional, Tuple
import math


class DifferentialPrivacy:
    """
    Differential privacy mechanisms for GBDT.

    Provides methods to add calibrated noise to statistics to ensure
    (epsilon, delta)-differential privacy.
    """

    def __init__(self, epsilon: float, delta: float = 0.0):
        """
        Initialize DP mechanism.

        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Privacy parameter for Gaussian mechanism (delta >= 0)
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")

        self.epsilon = epsilon
        self.delta = delta

    def laplace_mechanism(self,
                         true_value: Union[float, np.ndarray],
                         sensitivity: float) -> Union[float, np.ndarray]:
        """
        Add Laplace noise for (epsilon, 0)-differential privacy.

        The Laplace mechanism adds noise drawn from Laplace(sensitivity/epsilon).

        Args:
            true_value: True value(s) to privatize
            sensitivity: L1 sensitivity of the query

        Returns:
            Privatized value(s)
        """
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        scale = sensitivity / self.epsilon

        if isinstance(true_value, np.ndarray):
            noise = np.random.laplace(0, scale, size=true_value.shape)
            return true_value + noise
        else:
            noise = np.random.laplace(0, scale)
            return true_value + noise

    def gaussian_mechanism(self,
                          true_value: Union[float, np.ndarray],
                          sensitivity: float) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy.

        The Gaussian mechanism adds noise drawn from N(0, sigma^2) where:
        sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon

        Args:
            true_value: True value(s) to privatize
            sensitivity: L2 sensitivity of the query

        Returns:
            Privatized value(s)

        Raises:
            ValueError: If delta = 0 (Gaussian requires delta > 0)
        """
        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        if self.delta == 0:
            raise ValueError("Gaussian mechanism requires delta > 0")

        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

        if isinstance(true_value, np.ndarray):
            noise = np.random.normal(0, sigma, size=true_value.shape)
            return true_value + noise
        else:
            noise = np.random.normal(0, sigma)
            return true_value + noise

    def clip_value(self,
                   value: Union[float, np.ndarray],
                   clip_min: float,
                   clip_max: float) -> Union[float, np.ndarray]:
        """
        Clip values to bound sensitivity.

        Clipping is often used before adding noise to limit sensitivity.

        Args:
            value: Value(s) to clip
            clip_min: Minimum allowed value
            clip_max: Maximum allowed value

        Returns:
            Clipped value(s)
        """
        if isinstance(value, np.ndarray):
            return np.clip(value, clip_min, clip_max)
        else:
            return max(clip_min, min(clip_max, value))


class HistogramDP:
    """
    Differential privacy for histogram aggregation in GBDT.

    Histograms in GBDT contain [sum_gradients, sum_hessians, count] for each bin.
    We need to add noise to these statistics while preserving utility.
    """

    def __init__(self,
                 epsilon: float,
                 delta: float = 0.0,
                 mechanism: str = 'laplace'):
        """
        Initialize histogram DP.

        Args:
            epsilon: Privacy budget
            delta: Privacy parameter for Gaussian mechanism
            mechanism: 'laplace' or 'gaussian'
        """
        self.dp = DifferentialPrivacy(epsilon, delta)
        self.mechanism = mechanism

    def privatize_histogram(self,
                           histogram: np.ndarray,
                           grad_clip: float,
                           hess_clip: float,
                           max_count: int) -> np.ndarray:
        """
        Add DP noise to a histogram.

        Args:
            histogram: Histogram array (n_bins, 3)
                      Column 0: sum of gradients
                      Column 1: sum of Hessians
                      Column 2: count
            grad_clip: Clipping bound for gradients (for sensitivity)
            hess_clip: Clipping bound for Hessians (for sensitivity)
            max_count: Maximum count per bin (for sensitivity)

        Returns:
            Privatized histogram with noise added
        """
        privatized = histogram.copy()

        # Clip values to bound sensitivity
        privatized[:, 0] = self.dp.clip_value(privatized[:, 0], -grad_clip, grad_clip)
        privatized[:, 1] = self.dp.clip_value(privatized[:, 1], 0, hess_clip)
        privatized[:, 2] = self.dp.clip_value(privatized[:, 2], 0, max_count)

        # Add noise
        if self.mechanism == 'laplace':
            # Gradient: L1 sensitivity = grad_clip
            privatized[:, 0] = self.dp.laplace_mechanism(
                privatized[:, 0], sensitivity=grad_clip
            )

            # Hessian: L1 sensitivity = hess_clip
            privatized[:, 1] = self.dp.laplace_mechanism(
                privatized[:, 1], sensitivity=hess_clip
            )

            # Count: L1 sensitivity = 1 (one sample can change count by at most 1)
            privatized[:, 2] = self.dp.laplace_mechanism(
                privatized[:, 2], sensitivity=1.0
            )

        elif self.mechanism == 'gaussian':
            # Gradient: L2 sensitivity = grad_clip
            privatized[:, 0] = self.dp.gaussian_mechanism(
                privatized[:, 0], sensitivity=grad_clip
            )

            # Hessian: L2 sensitivity = hess_clip
            privatized[:, 1] = self.dp.gaussian_mechanism(
                privatized[:, 1], sensitivity=hess_clip
            )

            # Count: L2 sensitivity = 1
            privatized[:, 2] = self.dp.gaussian_mechanism(
                privatized[:, 2], sensitivity=1.0
            )

        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

        # Ensure Hessians and counts remain non-negative
        privatized[:, 1] = np.maximum(privatized[:, 1], 1e-12)  # Minimum Hessian
        privatized[:, 2] = np.maximum(privatized[:, 2], 0)  # Non-negative count

        return privatized

    def privatize_gradient(self,
                          gradient: np.ndarray,
                          clip: float) -> np.ndarray:
        """
        Privatize individual gradients.

        Args:
            gradient: Gradient array (n_samples,)
            clip: Clipping bound

        Returns:
            Privatized gradients
        """
        clipped = self.dp.clip_value(gradient, -clip, clip)

        if self.mechanism == 'laplace':
            return self.dp.laplace_mechanism(clipped, sensitivity=clip)
        else:
            return self.dp.gaussian_mechanism(clipped, sensitivity=clip)


class TreeDP:
    """
    Differential privacy for tree-level statistics.

    Computes privacy budget and composition across multiple trees.
    """

    def __init__(self,
                 total_epsilon: float,
                 total_delta: float,
                 n_trees: int):
        """
        Initialize tree-level DP.

        Args:
            total_epsilon: Total privacy budget for all trees
            total_delta: Total delta for all trees
            n_trees: Number of trees in the ensemble
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.n_trees = n_trees

        # Advanced Composition: split budget evenly
        self.per_tree_epsilon = total_epsilon / math.sqrt(n_trees)
        self.per_tree_delta = total_delta / n_trees

    def get_histogram_dp(self) -> HistogramDP:
        """
        Get a HistogramDP with per-tree privacy budget.

        Returns:
            HistogramDP configured for one tree
        """
        return HistogramDP(
            epsilon=self.per_tree_epsilon,
            delta=self.per_tree_delta,
            mechanism='laplace'
        )

    def compute_remaining_budget(self, trees_trained: int) -> Tuple[float, float]:
        """
        Compute remaining privacy budget.

        Args:
            trees_trained: Number of trees already trained

        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        if trees_trained >= self.n_trees:
            return 0.0, 0.0

        remaining_trees = self.n_trees - trees_trained
        remaining_epsilon = self.per_tree_epsilon * math.sqrt(remaining_trees)
        remaining_delta = self.per_tree_delta * remaining_trees

        return remaining_epsilon, remaining_delta


# Convenience functions
def laplace_noise(true_value: Union[float, np.ndarray],
                 epsilon: float,
                 sensitivity: float) -> Union[float, np.ndarray]:
    """Add Laplace noise for (epsilon, 0)-DP."""
    dp = DifferentialPrivacy(epsilon, delta=0.0)
    return dp.laplace_mechanism(true_value, sensitivity)


def gaussian_noise(true_value: Union[float, np.ndarray],
                  epsilon: float,
                  delta: float,
                  sensitivity: float) -> Union[float, np.ndarray]:
    """Add Gaussian noise for (epsilon, delta)-DP."""
    dp = DifferentialPrivacy(epsilon, delta)
    return dp.gaussian_mechanism(true_value, sensitivity)
