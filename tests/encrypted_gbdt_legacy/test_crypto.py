"""
Unit tests for cryptographic primitives.

Tests for secret sharing, differential privacy, and secure aggregation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from crypto.secret_sharing import SecretSharing, share, reconstruct, add_shares
from crypto.dp_mechanisms import (
    DifferentialPrivacy, HistogramDP, TreeDP,
    laplace_noise, gaussian_noise
)
from crypto.secure_aggregation import (
    SecureAggregator, DPSecureAggregator,
    GradientAggregator, compute_privacy_cost
)


class TestSecretSharing:
    """Test additive secret sharing."""

    def test_share_and_reconstruct(self):
        """Test basic share and reconstruct."""
        ss = SecretSharing()

        value = 42
        shares = ss.share(value, n_parties=3)
        reconstructed = ss.reconstruct(shares)

        assert reconstructed == value

    def test_negative_value(self):
        """Test sharing negative values."""
        ss = SecretSharing()

        value = -17
        shares = ss.share(value, n_parties=3)
        reconstructed = ss.reconstruct(shares)

        assert reconstructed == value

    def test_multiple_parties(self):
        """Test sharing among different numbers of parties."""
        ss = SecretSharing()

        value = 100
        for n in [2, 3, 5, 10]:
            shares = ss.share(value, n_parties=n)
            reconstructed = ss.reconstruct(shares)
            assert reconstructed == value

    def test_array_sharing(self):
        """Test sharing arrays of values."""
        ss = SecretSharing()

        values = np.array([1, 2, 3, 4, 5])
        shares = ss.share_array(values, n_parties=3)

        assert shares.shape == (3, 5)

        reconstructed = ss.reconstruct_array(shares)
        np.testing.assert_array_equal(reconstructed, values)

    def test_add_shared(self):
        """Test adding shared values."""
        ss = SecretSharing()

        # Share two values
        shares_a = ss.share(10, n_parties=3)
        shares_b = ss.share(15, n_parties=3)

        # Add them while shared
        shares_sum = ss.add_shared(shares_a, shares_b)

        # Reconstruct
        result = ss.reconstruct(shares_sum)

        assert result == 25

    def test_multiply_by_constant(self):
        """Test multiplying shared value by constant."""
        ss = SecretSharing()

        value = 7
        constant = 3

        shares = ss.share(value, n_parties=3)
        shares_scaled = ss.multiply_by_constant(shares, constant)
        result = ss.reconstruct(shares_scaled)

        assert result == 21

    def test_float_sharing(self):
        """Test sharing floating-point values."""
        ss = SecretSharing()

        value = 3.14159
        shares = ss.share_float(value, n_parties=3, scale=100000)
        reconstructed = ss.reconstruct_float(shares, scale=100000)

        assert abs(reconstructed - value) < 0.0001

    def test_convenience_functions(self):
        """Test convenience functions."""
        value = 123
        shares = share(value, n_parties=3)
        reconstructed = reconstruct(shares)

        assert reconstructed == value


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""

    def test_laplace_mechanism(self):
        """Test Laplace mechanism."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=0.0)

        true_value = 10.0
        sensitivity = 1.0

        # Add noise multiple times
        noisy_values = [dp.laplace_mechanism(true_value, sensitivity) for _ in range(100)]

        # Mean should be close to true value
        assert abs(np.mean(noisy_values) - true_value) < 1.0

    def test_laplace_array(self):
        """Test Laplace mechanism on arrays."""
        dp = DifferentialPrivacy(epsilon=1.0)

        true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = dp.laplace_mechanism(true_values, sensitivity=1.0)

        assert noisy.shape == true_values.shape
        assert not np.array_equal(noisy, true_values)  # Should have noise

    def test_gaussian_mechanism(self):
        """Test Gaussian mechanism."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        true_value = 10.0
        sensitivity = 1.0

        noisy_values = [dp.gaussian_mechanism(true_value, sensitivity) for _ in range(100)]

        # Mean should be close to true value
        assert abs(np.mean(noisy_values) - true_value) < 1.0

    def test_clipping(self):
        """Test value clipping."""
        dp = DifferentialPrivacy(epsilon=1.0)

        values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        clipped = dp.clip_value(values, -2.0, 2.0)

        np.testing.assert_array_equal(clipped, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))

    def test_histogram_dp(self):
        """Test histogram privatization."""
        hist_dp = HistogramDP(epsilon=1.0, delta=0.0, mechanism='laplace')

        # Create histogram
        histogram = np.array([
            [10.0, 5.0, 100],  # Bin 0
            [-5.0, 3.0, 50],   # Bin 1
        ])

        privatized = hist_dp.privatize_histogram(
            histogram,
            grad_clip=10.0,
            hess_clip=5.0,
            max_count=100
        )

        assert privatized.shape == histogram.shape
        # Hessians and counts should be non-negative
        assert np.all(privatized[:, 1] >= 0)
        assert np.all(privatized[:, 2] >= 0)

    def test_convenience_functions(self):
        """Test convenience functions."""
        value = 5.0

        # Laplace
        noisy_laplace = laplace_noise(value, epsilon=1.0, sensitivity=1.0)
        assert isinstance(noisy_laplace, float)

        # Gaussian
        noisy_gaussian = gaussian_noise(value, epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert isinstance(noisy_gaussian, float)


class TestSecureAggregation:
    """Test secure aggregation protocols."""

    def test_secure_aggregate(self):
        """Test secure histogram aggregation."""
        aggregator = SecureAggregator(n_parties=3)

        # Create histograms from 3 parties (using integers for exactness)
        histograms = [
            np.array([[10.0, 5.0, 100], [20.0, 10.0, 200]]),
            np.array([[5.0, 2.0, 50], [10.0, 5.0, 100]]),
            np.array([[15.0, 7.0, 150], [5.0, 2.0, 50]]),
        ]

        result = aggregator.secure_aggregate(histograms)

        # Expected sum
        expected = np.sum(histograms, axis=0)

        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_share_histogram(self):
        """Test histogram sharing."""
        aggregator = SecureAggregator(n_parties=3)

        histogram = np.array([[10.0, 5.0, 100], [20.0, 10.0, 200]])

        shares_dict = aggregator.share_histogram(histogram, party_id=0)

        # Should have shares for all 3 parties
        assert len(shares_dict) == 3

        # Each share should have same shape
        for shares in shares_dict.values():
            assert shares.shape == histogram.shape

    def test_aggregate_shares(self):
        """Test share aggregation."""
        aggregator = SecureAggregator(n_parties=3)

        # Create shares (simulate 3 parties)
        shares = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[2, 3, 4], [5, 6, 7]]),
            np.array([[3, 4, 5], [6, 7, 8]]),
        ]

        result = aggregator.aggregate_shares(shares)
        expected = np.array([[6, 9, 12], [15, 18, 21]])

        np.testing.assert_array_equal(result, expected)

    def test_dp_aggregate(self):
        """Test DP secure aggregation."""
        aggregator = DPSecureAggregator(
            n_parties=3,
            epsilon=1.0,
            delta=0.0,
            grad_clip=10.0,
            hess_clip=5.0,
            max_count=100
        )

        # Use integer values for exact secret sharing
        histograms = [
            np.array([[10.0, 5.0, 100]]),
            np.array([[5.0, 2.0, 50]]),
            np.array([[15.0, 7.0, 150]]),
        ]

        # Without noise
        result_no_noise = aggregator.dp_aggregate(histograms, add_noise=False)
        expected = np.sum(histograms, axis=0)
        np.testing.assert_array_almost_equal(result_no_noise, expected, decimal=0)

        # With noise (should be different)
        result_with_noise = aggregator.dp_aggregate(histograms, add_noise=True)
        assert not np.allclose(result_with_noise, result_no_noise, atol=0.01)


class TestGradientAggregator:
    """Test gradient aggregation."""

    def test_share_gradients(self):
        """Test gradient sharing."""
        aggregator = GradientAggregator(n_parties=3)

        gradients = np.array([0.1, 0.2, 0.3, 0.4])
        hessians = np.array([0.25, 0.25, 0.25, 0.25])

        grad_shares, hess_shares = aggregator.share_gradients(gradients, hessians)

        assert len(grad_shares) == 3
        assert len(hess_shares) == 3

        # Each party should have a share array
        for party_id in range(3):
            assert grad_shares[party_id].shape == gradients.shape
            assert hess_shares[party_id].shape == hessians.shape

    def test_aggregate_gradients(self):
        """Test gradient aggregation."""
        aggregator = GradientAggregator(n_parties=3)

        # Create share arrays
        grad_shares = [
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([2, 3, 4, 5], dtype=np.int64),
            np.array([3, 4, 5, 6], dtype=np.int64),
        ]

        hess_shares = [
            np.array([5, 5, 5, 5], dtype=np.int64),
            np.array([5, 5, 5, 5], dtype=np.int64),
            np.array([5, 5, 5, 5], dtype=np.int64),
        ]

        agg_grad, agg_hess = aggregator.aggregate_gradients(grad_shares, hess_shares)

        np.testing.assert_array_equal(agg_grad, np.array([6, 9, 12, 15]))
        np.testing.assert_array_equal(agg_hess, np.array([15, 15, 15, 15]))


class TestTreeDP:
    """Test tree-level differential privacy."""

    def test_budget_splitting(self):
        """Test splitting privacy budget across trees."""
        tree_dp = TreeDP(total_epsilon=1.0, total_delta=1e-5, n_trees=10)

        assert tree_dp.per_tree_epsilon > 0
        assert tree_dp.per_tree_delta > 0

        # Per-tree epsilon should be smaller than total
        assert tree_dp.per_tree_epsilon < tree_dp.total_epsilon

    def test_remaining_budget(self):
        """Test computing remaining privacy budget."""
        tree_dp = TreeDP(total_epsilon=1.0, total_delta=1e-5, n_trees=10)

        # After training 5 trees
        remaining_eps, remaining_delta = tree_dp.compute_remaining_budget(trees_trained=5)

        assert remaining_eps > 0
        assert remaining_delta > 0

        # After all trees trained
        remaining_eps, remaining_delta = tree_dp.compute_remaining_budget(trees_trained=10)
        assert remaining_eps == 0
        assert remaining_delta == 0

    def test_get_histogram_dp(self):
        """Test getting HistogramDP with correct budget."""
        tree_dp = TreeDP(total_epsilon=1.0, total_delta=1e-5, n_trees=10)

        hist_dp = tree_dp.get_histogram_dp()

        assert hist_dp.dp.epsilon == tree_dp.per_tree_epsilon
        assert hist_dp.dp.delta == tree_dp.per_tree_delta


class TestPrivacyCost:
    """Test privacy cost computation."""

    def test_compute_privacy_cost(self):
        """Test computing total privacy cost."""
        n_trees = 10
        epsilon_per_tree = 0.1
        delta_per_tree = 1e-6

        total_eps, total_delta = compute_privacy_cost(
            n_trees, epsilon_per_tree, delta_per_tree
        )

        # Total epsilon should grow with sqrt(n_trees)
        assert total_eps > epsilon_per_tree

        # Total delta should grow linearly
        assert total_delta == n_trees * delta_per_tree


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
