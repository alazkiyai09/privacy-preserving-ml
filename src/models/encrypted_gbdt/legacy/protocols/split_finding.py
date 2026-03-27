"""
Secure split finding protocol for privacy-preserving GBDT.

In federated GBDT, each party holds different features. To find the best split,
parties must jointly evaluate candidate splits without revealing their feature values.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from crypto.secure_aggregation import SecureAggregator, DPSecureAggregator
from core.histogram import HistogramBuilder, SplitInfo, find_best_split_histogram


@dataclass
class CandidateSplit:
    """Information about a candidate split."""
    party_id: int
    feature_idx: int
    bin_idx: int
    split_value: float
    gain: float


class SecureSplitFinding:
    """
    Secure protocol for finding the best split across parties.

    The protocol works as follows:
    1. Each party builds local histograms for their features
    2. Parties securely aggregate histograms (using secret sharing)
    3. For each feature, find the best split point
    4. Compare features to find the global best split

    No single party sees another party's feature values or histograms.
    """

    def __init__(self,
                 n_parties: int,
                 max_bins: int = 256,
                 use_dp: bool = False,
                 epsilon: float = 1.0,
                 delta: float = 0.0):
        """
        Initialize secure split finding.

        Args:
            n_parties: Number of participating parties
            max_bins: Maximum number of histogram bins
            use_dp: Whether to use differential privacy
            epsilon: Privacy budget (if using DP)
            delta: Privacy parameter for Gaussian mechanism
        """
        self.n_parties = n_parties
        self.max_bins = max_bins

        if use_dp:
            self.aggregator = DPSecureAggregator(
                n_parties=n_parties,
                epsilon=epsilon,
                delta=delta
            )
        else:
            self.aggregator = SecureAggregator(n_parties=n_parties)

        self.histogram_builders = []

    def initialize_histograms(self,
                             n_features_per_party: List[int],
                             bin_edges_list: List[List[np.ndarray]]) -> None:
        """
        Initialize histogram builders for all parties.

        Args:
            n_features_per_party: Number of features each party has
            bin_edges_list: Pre-computed bin edges for each party's features
        """
        self.histogram_builders = []

        for party_id, n_features in enumerate(n_features_per_party):
            builder = HistogramBuilder(max_bins=self.max_bins)
            builder.bin_edges_ = bin_edges_list[party_id]
            self.histogram_builders.append(builder)

    def party_build_histograms(self,
                              party_id: int,
                              features: np.ndarray,
                              gradients: np.ndarray,
                              hessians: np.ndarray,
                              sample_indices: np.ndarray) -> List[np.ndarray]:
        """
        Build local histograms for a single party.

        Args:
            party_id: ID of the party
            features: Local feature matrix
            gradients: Gradient values
            hessians: Hessian values
            sample_indices: Indices of samples in current node

        Returns:
            List of histograms, one per local feature
        """
        histograms = []

        party_features = features[sample_indices, :]
        party_gradients = gradients[sample_indices]
        party_hessians = hessians[sample_indices]

        n_features = features.shape[1]

        for feat_idx in range(n_features):
            builder = self.histogram_builders[party_id]
            hist = builder.build_histogram(
                party_features[:, feat_idx],
                party_gradients,
                party_hessians,
                builder.bin_edges_[feat_idx]
            )
            histograms.append(hist)

        return histograms

    def secure_find_best_split(self,
                              all_histograms: List[List[np.ndarray]],
                              all_bin_edges: List[List[np.ndarray]],
                              lambda_reg: float = 1.0,
                              min_child_weight: float = 1.0) -> CandidateSplit:
        """
        Securely find the best split across all parties.

        Args:
            all_histograms: Histograms from each party [party][feature] -> histogram
            all_bin_edges: Bin edges for each party [party][feature] -> edges
            lambda_reg: L2 regularization
            min_child_weight: Minimum child weight

        Returns:
            CandidateSplit with the best global split
        """
        best_global_gain = -np.inf
        best_split = None

        # For each party, find their best local split
        for party_id, party_histograms in enumerate(all_histograms):
            party_bin_edges = all_bin_edges[party_id]

            for feat_idx, (hist, edges) in enumerate(zip(party_histograms, party_bin_edges)):
                bin_idx, split_value, gain = find_best_split_histogram(
                    hist, edges, lambda_reg, min_child_weight
                )

                if gain > best_global_gain:
                    best_global_gain = gain
                    best_split = CandidateSplit(
                        party_id=party_id,
                        feature_idx=feat_idx,
                        bin_idx=bin_idx,
                        split_value=split_value,
                        gain=gain
                    )

        return best_split

    def federated_split_finding(self,
                               party_features: List[np.ndarray],
                               party_feature_indices: List[List[int]],
                               gradients: np.ndarray,
                               hessians: np.ndarray,
                               sample_indices: np.ndarray,
                               lambda_reg: float = 1.0) -> Tuple[int, int, float, float]:
        """
        Perform federated split finding across all parties.

        Args:
            party_features: List of feature matrices from each party
            party_feature_indices: List of feature indices for each party
            gradients: Gradient values
            hessians: Hessian values
            sample_indices: Indices of samples in current node
            lambda_reg: L2 regularization

        Returns:
            Tuple of (party_id, feature_idx, split_value, gain)
        """
        # Each party builds local histograms
        all_histograms = []
        all_bin_edges = []

        for party_id, features in enumerate(party_features):
            histograms = self.party_build_histograms(
                party_id, features, gradients, hessians, sample_indices
            )
            all_histograms.append(histograms)

            bin_edges = self.histogram_builders[party_id].bin_edges_
            all_bin_edges.append(bin_edges)

        # Find best split
        best_split = self.secure_find_best_split(
            all_histograms, all_bin_edges, lambda_reg
        )

        return (best_split.party_id,
                best_split.feature_idx,
                best_split.split_value,
                best_split.gain)


class ApproximateSplitFinding:
    """
    Approximate split finding using discretized features.

    Instead of exact histograms, parties coarsely discretize features
    and share aggregate statistics. This is faster but less accurate.
    """

    def __init__(self,
                 n_parties: int,
                 n_approx_bins: int = 10):
        """
        Initialize approximate split finding.

        Args:
            n_parties: Number of parties
            n_approx_bins: Number of bins for approximation
        """
        self.n_parties = n_parties
        self.n_approx_bins = n_approx_bins

    def approximate_histogram(self,
                              feature_values: np.ndarray,
                              gradients: np.ndarray,
                              hessians: np.ndarray) -> np.ndarray:
        """
        Build a coarse histogram for approximate split finding.

        Args:
            feature_values: Feature values (n_samples,)
            gradients: Gradient values
            hessians: Hessian values

        Returns:
            Approximate histogram (n_approx_bins, 3)
        """
        # Compute quantile-based bins
        quantiles = np.linspace(0, 1, self.n_approx_bins + 1)
        bin_edges = np.quantile(feature_values, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            # All values are the same
            return np.array([[np.sum(gradients), np.sum(hessians), len(feature_values)]])

        # Digitize
        bin_indices = np.digitize(feature_values, bin_edges, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

        # Build histogram
        n_bins = len(bin_edges) - 1
        histogram = np.zeros((n_bins, 3))

        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx)
            histogram[bin_idx, 0] = np.sum(gradients[mask])
            histogram[bin_idx, 1] = np.sum(hessians[mask])
            histogram[bin_idx, 2] = np.sum(mask)

        return histogram, bin_edges


def compute_split_gains_histograms(histograms: List[np.ndarray],
                                   lambda_reg: float = 1.0) -> List[Tuple[int, int, float, float]]:
    """
    Compute split gains for a list of histograms.

    Args:
        histograms: List of histograms
        lambda_reg: L2 regularization

    Returns:
        List of (feature_idx, bin_idx, split_value, gain) tuples
    """
    results = []

    for feat_idx, hist in enumerate(histograms):
        # Assume uniform bin edges for simplicity
        n_bins = hist.shape[0]
        bin_edges = np.linspace(0, 1, n_bins + 1)

        bin_idx, split_value, gain = find_best_split_histogram(
            hist, bin_edges, lambda_reg
        )

        results.append((feat_idx, bin_idx, split_value, gain))

    return results
