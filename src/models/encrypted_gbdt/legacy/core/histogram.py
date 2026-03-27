"""
Histogram builder for efficient split finding in GBDT.

Implements histogram-based split finding with regularization,
which is the foundation for privacy-preserving split finding.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SplitInfo:
    """Information about a split."""
    feature_idx: int
    bin_idx: int
    split_value: float
    gain: float


class HistogramBuilder:
    """
    Build histograms for gradient-based tree building.

    For each feature, we compute aggregate statistics (sum of gradients,
    sum of Hessians, count) for each bin. This allows efficient split finding.
    """

    def __init__(self, max_bins: int = 256):
        """
        Initialize histogram builder.

        Args:
            max_bins: Maximum number of bins per feature for discretization
        """
        self.max_bins = max_bins
        self.bin_edges_: List[np.ndarray] = None  # Bin edges for each feature

    def compute_bin_edges(self, features: np.ndarray,
                         feature_idx: int) -> np.ndarray:
        """
        Compute quantile-based bin edges for a feature.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_idx: Index of feature to bin

        Returns:
            Array of bin edges (length max_bins + 1)
        """
        feature_values = features[:, feature_idx]

        # Remove NaN values for binning
        valid_values = feature_values[~np.isnan(feature_values)]

        if len(valid_values) == 0:
            # All missing, return uniform bins
            return np.linspace(0, 1, self.max_bins + 1)

        # Quantile-based binning
        quantiles = np.linspace(0, 1, self.max_bins + 1)
        bin_edges = np.quantile(valid_values, quantiles)

        # Ensure unique edges (handle duplicate values)
        bin_edges = np.unique(bin_edges)

        # If we have too few unique values, pad with endpoints
        if len(bin_edges) < 2:
            bin_edges = np.array([valid_values.min(), valid_values.max()])

        # Ensure we have exactly max_bins + 1 edges
        if len(bin_edges) < self.max_bins + 1:
            # Interpolate to get desired number of bins
            bin_edges = np.linspace(bin_edges[0], bin_edges[-1], self.max_bins + 1)

        return bin_edges

    def fit_bins(self, features: np.ndarray) -> 'HistogramBuilder':
        """
        Compute bin edges for all features.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            self (fitted histogram builder)
        """
        n_features = features.shape[1]
        self.bin_edges_ = []

        for feat_idx in range(n_features):
            edges = self.compute_bin_edges(features, feat_idx)
            self.bin_edges_.append(edges)

        return self

    def digitize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Convert feature values to bin indices.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Binned features (n_samples, n_features) with bin indices
        """
        if self.bin_edges_ is None:
            raise ValueError("HistogramBuilder not fitted. Call fit_bins first.")

        n_samples, n_features = features.shape
        binned = np.zeros((n_samples, n_features), dtype=np.int32)

        for feat_idx in range(n_features):
            binned[:, feat_idx] = np.digitize(
                features[:, feat_idx],
                self.bin_edges_[feat_idx],
                right=False
            ) - 1

            # Clip to valid range
            n_bins = len(self.bin_edges_[feat_idx]) - 1
            binned[:, feat_idx] = np.clip(binned[:, feat_idx], 0, n_bins - 1)

        return binned

    def build_histogram(self,
                       feature_values: np.ndarray,
                       gradients: np.ndarray,
                       hessians: np.ndarray,
                       bin_edges: np.ndarray) -> np.ndarray:
        """
        Build histogram for a single feature.

        Histogram shape: (n_bins, 3)
        - Column 0: Sum of gradients
        - Column 1: Sum of Hessians
        - Column 2: Count of samples

        Args:
            feature_values: Feature values (n_samples,)
            gradients: Gradient values (n_samples,)
            hessians: Hessian values (n_samples,)
            bin_edges: Bin edges for this feature

        Returns:
            Histogram array (n_bins, 3)
        """
        n_bins = len(bin_edges) - 1
        histogram = np.zeros((n_bins, 3), dtype=np.float64)

        # Bin the feature values
        bin_indices = np.digitize(feature_values, bin_edges, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Aggregate statistics per bin
        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx)
            histogram[bin_idx, 0] = np.sum(gradients[mask])  # Sum gradients
            histogram[bin_idx, 1] = np.sum(hessians[mask])  # Sum Hessians
            histogram[bin_idx, 2] = np.sum(mask)  # Count

        return histogram

    def build_all_histograms(self,
                            features: np.ndarray,
                            gradients: np.ndarray,
                            hessians: np.ndarray) -> List[np.ndarray]:
        """
        Build histograms for all features.

        Args:
            features: Feature matrix (n_samples, n_features)
            gradients: Gradient values (n_samples,)
            hessians: Hessian values (n_samples,)

        Returns:
            List of histograms, one per feature
        """
        n_features = features.shape[1]

        if self.bin_edges_ is None:
            self.fit_bins(features)

        histograms = []

        for feat_idx in range(n_features):
            hist = self.build_histogram(
                features[:, feat_idx],
                gradients,
                hessians,
                self.bin_edges_[feat_idx]
            )
            histograms.append(hist)

        return histograms


def find_best_split_histogram(histogram: np.ndarray,
                             bin_edges: np.ndarray,
                             lambda_reg: float = 1.0,
                             min_child_weight: float = 1.0) -> Tuple[int, float, float]:
    """
    Find the best split point from a histogram.

    Uses the exact split finding algorithm from GBDT literature:
    Gain = (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G^2 / (H + lambda))

    where G_L, H_L are sum of gradients and Hessians in left child,
          G_R, H_R are sum of gradients and Hessians in right child,
          G, H are total sum of gradients and Hessians.

    Args:
        histogram: Histogram array (n_bins, 3)
                   Column 0: Sum gradients
                   Column 1: Sum Hessians
                   Column 2: Count
        bin_edges: Bin edges (n_bins + 1,)
        lambda_reg: L2 regularization parameter
        min_child_weight: Minimum sum of Hessians in a child node

    Returns:
        Tuple of (best_bin_idx, split_value, gain)
    """
    n_bins = histogram.shape[0]

    # Total statistics
    G_total = np.sum(histogram[:, 0])  # Total gradient
    H_total = np.sum(histogram[:, 1])  # Total Hessian

    # Parent score (regularized)
    parent_score = -(G_total ** 2) / (H_total + lambda_reg)

    best_gain = -np.inf
    best_bin_idx = 0

    # Cumulative sums for efficient split evaluation
    G_left = 0.0
    H_left = 0.0
    count_left = 0

    for i in range(n_bins - 1):  # Can't split after last bin
        # Update left child statistics
        G_left += histogram[i, 0]
        H_left += histogram[i, 1]
        count_left += histogram[i, 2]

        # Right child statistics
        G_right = G_total - G_left
        H_right = H_total - H_left
        count_right = np.sum(histogram[i+1:, 2])

        # Check minimum child weight constraint
        if H_left < min_child_weight or H_right < min_child_weight:
            continue

        # Compute gain
        left_score = -(G_left ** 2) / (H_left + lambda_reg)
        right_score = -(G_right ** 2) / (H_right + lambda_reg)

        gain = left_score + right_score - parent_score

        if gain > best_gain:
            best_gain = gain
            best_bin_idx = i + 1  # Split after bin i

    # Split value is the edge at the best bin index
    split_value = bin_edges[best_bin_idx]

    return best_bin_idx, split_value, best_gain


def find_best_split_all_features(histograms: List[np.ndarray],
                                  bin_edges_list: List[np.ndarray],
                                  lambda_reg: float = 1.0,
                                  min_child_weight: float = 1.0) -> SplitInfo:
    """
    Find the best split across all features.

    Args:
        histograms: List of histograms, one per feature
        bin_edges_list: List of bin edges, one per feature
        lambda_reg: L2 regularization parameter
        min_child_weight: Minimum sum of Hessians in a child node

    Returns:
        SplitInfo object with best split information
    """
    best_gain = -np.inf
    best_feature_idx = 0
    best_bin_idx = 0
    best_split_value = 0.0

    for feat_idx, (hist, edges) in enumerate(zip(histograms, bin_edges_list)):
        bin_idx, split_value, gain = find_best_split_histogram(
            hist, edges, lambda_reg, min_child_weight
        )

        if gain > best_gain:
            best_gain = gain
            best_feature_idx = feat_idx
            best_bin_idx = bin_idx
            best_split_value = split_value

    return SplitInfo(
        feature_idx=best_feature_idx,
        bin_idx=best_bin_idx,
        split_value=best_split_value,
        gain=best_gain
    )
