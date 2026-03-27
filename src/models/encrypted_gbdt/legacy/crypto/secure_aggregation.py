"""
Secure aggregation for privacy-preserving GBDT.

Implements protocols for aggregating histograms and gradients across multiple
parties without revealing individual party's data.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from .secret_sharing import SecretSharing
from .dp_mechanisms import HistogramDP, DifferentialPrivacy


class SecureAggregator:
    """
    Secure aggregation of histograms using additive secret sharing.

    Multiple parties each have a histogram of their local data.
    We want to compute the sum of all histograms without any party
    seeing another party's histogram.
    """

    def __init__(self,
                 n_parties: int,
                 field_modulus: Optional[int] = None):
        """
        Initialize secure aggregator.

        Args:
            n_parties: Number of parties participating in aggregation
            field_modulus: Field modulus for secret sharing
        """
        self.n_parties = n_parties
        self.ss = SecretSharing(field_modulus)

    def share_histogram(self,
                       histogram: np.ndarray,
                       party_id: int) -> Dict[int, np.ndarray]:
        """
        Share a histogram's values among all parties.

        Each value in the histogram is split into n shares, one for each party.

        Args:
            histogram: Histogram array (n_bins, 3)
                      Column 0: sum gradients
                      Column 1: sum Hessians
                      Column 2: count
            party_id: ID of the party sharing this histogram

        Returns:
            Dictionary mapping party_id -> their share of this histogram
        """
        n_bins = histogram.shape[0]
        shares_dict = {}

        # Share each value in the histogram
        for target_party in range(self.n_parties):
            shares_dict[target_party] = np.zeros_like(histogram, dtype=np.float64)

        for bin_idx in range(n_bins):
            for col_idx in range(3):
                value = int(histogram[bin_idx, col_idx])
                shares = self.ss.share(value, self.n_parties)

                for target_party in range(self.n_parties):
                    shares_dict[target_party][bin_idx, col_idx] = float(shares[target_party])

        return shares_dict

    def aggregate_shares(self,
                        received_shares: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate received shares from all parties.

        Each party sends their share to a central aggregator (or to all parties).
        Summing all shares gives the sum of original histograms.

        Args:
            received_shares: List of share arrays from each party

        Returns:
            Aggregated histogram (sum of all input histograms)
        """
        if len(received_shares) != self.n_parties:
            raise ValueError(f"Expected {self.n_parties} shares, got {len(received_shares)}")

        # Sum all shares element-wise with modulo
        aggregated = np.sum(received_shares, axis=0) % self.ss.field_modulus

        # The sum is already in the field, but might need conversion
        # Convert back to regular integers
        result = aggregated.astype(np.float64)

        # Handle field overflow (values > modulus/2 are negative)
        modulus = self.ss.field_modulus
        result = np.where(result > modulus / 2, result - modulus, result)

        return result

    def secure_aggregate(self,
                        histograms: List[np.ndarray]) -> np.ndarray:
        """
        Securely aggregate histograms from all parties.

        This simulates the full protocol:
        1. Each party shares their histogram
        2. Parties exchange shares
        3. Each party aggregates to get the sum

        Args:
            histograms: List of histograms from each party

        Returns:
            Aggregated histogram (sum of all histograms)
        """
        if len(histograms) != self.n_parties:
            raise ValueError(f"Expected {self.n_parties} histograms, got {len(histograms)}")

        # Each party shares their histogram
        all_shares = []  # all_shares[party_id][target_id] = share

        for party_id, histogram in enumerate(histograms):
            shares_dict = self.share_histogram(histogram, party_id)
            all_shares.append(shares_dict)

        # Collect shares for aggregation
        # For party 0, collect its share from each histogram
        aggregated_shares = []

        for target_party in range(self.n_parties):
            party_share = np.zeros_like(histograms[0], dtype=np.float64)

            for source_party in range(self.n_parties):
                # Add with modulo to prevent overflow
                party_share = (party_share + all_shares[source_party][target_party]) % self.ss.field_modulus

            aggregated_shares.append(party_share)

        # Reconstruct the sum
        result = self.aggregate_shares(aggregated_shares)

        return result


class DPSecureAggregator(SecureAggregator):
    """
    Secure aggregator with differential privacy.

    Combines secret sharing for secure aggregation with differential
    privacy for formal privacy guarantees.
    """

    def __init__(self,
                 n_parties: int,
                 epsilon: float,
                 delta: float = 0.0,
                 field_modulus: Optional[int] = None,
                 grad_clip: float = 1.0,
                 hess_clip: float = 1.0,
                 max_count: int = 1000):
        """
        Initialize DP secure aggregator.

        Args:
            n_parties: Number of parties
            epsilon: Privacy budget
            delta: Privacy parameter for Gaussian mechanism
            field_modulus: Field modulus for secret sharing
            grad_clip: Clipping bound for gradients
            hess_clip: Clipping bound for Hessians
            max_count: Maximum count per bin
        """
        super().__init__(n_parties, field_modulus)
        self.epsilon = epsilon
        self.delta = delta
        self.grad_clip = grad_clip
        self.hess_clip = hess_clip
        self.max_count = max_count

        self.hist_dp = HistogramDP(epsilon, delta, mechanism='laplace')

    def dp_aggregate(self,
                    histograms: List[np.ndarray],
                    add_noise: bool = True) -> np.ndarray:
        """
        Securely aggregate histograms with differential privacy.

        Args:
            histograms: List of histograms from each party
            add_noise: Whether to add DP noise (default: True)

        Returns:
            Privatized aggregated histogram
        """
        # First do secure aggregation
        aggregated = self.secure_aggregate(histograms)

        if add_noise:
            # Add DP noise
            aggregated = self.hist_dp.privatize_histogram(
                aggregated,
                grad_clip=self.grad_clip * self.n_parties,
                hess_clip=self.hess_clip * self.n_parties,
                max_count=self.max_count * self.n_parties
            )

        return aggregated


class GradientAggregator:
    """
    Secure aggregation of gradients and Hessians.

    For federated GBDT, gradients and Hessians need to be aggregated
    across parties (where one party holds the labels).
    """

    def __init__(self,
                 n_parties: int,
                 field_modulus: Optional[int] = None):
        """
        Initialize gradient aggregator.

        Args:
            n_parties: Number of parties
            field_modulus: Field modulus for secret sharing
        """
        self.n_parties = n_parties
        self.ss = SecretSharing(field_modulus)

    def share_gradients(self,
                       gradients: np.ndarray,
                       hessians: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Share gradients and Hessians among parties.

        Args:
            gradients: Gradient array (n_samples,)
            hessians: Hessian array (n_samples,)

        Returns:
            Tuple of (gradient_shares, hessian_shares)
            Each is a dict mapping party_id -> their share
        """
        grad_shares = {}
        hess_shares = {}

        for party_id in range(self.n_parties):
            grad_shares[party_id] = np.zeros_like(gradients, dtype=np.int64)
            hess_shares[party_id] = np.zeros_like(hessians, dtype=np.int64)

        for i, (g, h) in enumerate(zip(gradients, hessians)):
            g_shares = self.ss.share(int(g), self.n_parties)
            h_shares = self.ss.share(int(h), self.n_parties)

            for party_id in range(self.n_parties):
                grad_shares[party_id][i] = g_shares[party_id]
                hess_shares[party_id][i] = h_shares[party_id]

        return grad_shares, hess_shares

    def aggregate_gradients(self,
                           grad_shares: List[np.ndarray],
                           hess_shares: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate gradient and Hessian shares.

        Args:
            grad_shares: List of gradient share arrays from each party
            hess_shares: List of Hessian share arrays from each party

        Returns:
            Tuple of (aggregated_gradients, aggregated_hessians)
        """
        agg_grads = self.aggregate_shares(grad_shares)
        agg_hess = self.aggregate_shares(hess_shares)

        return agg_grads, agg_hess

    def aggregate_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate shares into the original values.

        Args:
            shares: List of share arrays

        Returns:
            Aggregated (reconstructed) values
        """
        if len(shares) != self.n_parties:
            raise ValueError(f"Expected {self.n_parties} shares, got {len(shares)}")

        # Sum shares element-wise
        aggregated = np.sum(shares, axis=0)

        # Convert from field to regular integers
        result = aggregated.astype(np.float64)
        modulus = self.ss.field_modulus
        result = np.where(result > modulus / 2, result - modulus, result)

        return result


def compute_privacy_cost(n_trees: int,
                        epsilon_per_tree: float,
                        delta_per_tree: float) -> Tuple[float, float]:
    """
    Compute total privacy cost for training n_trees trees.

    Uses advanced composition theorem.

    Args:
        n_trees: Number of trees trained
        epsilon_per_tree: Epsilon used per tree
        delta_per_tree: Delta used per tree

    Returns:
        Tuple of (total_epsilon, total_delta)
    """
    # Advanced composition
    total_epsilon = epsilon_per_tree * math.sqrt(2 * n_trees * math.log(1 / delta_per_tree))
    total_delta = n_trees * delta_per_tree

    return total_epsilon, total_delta
