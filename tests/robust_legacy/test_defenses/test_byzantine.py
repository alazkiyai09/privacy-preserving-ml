"""
Unit Tests for Byzantine Aggregation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np

from src.defenses.byzantine_aggregation import (
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator
)
from src.models.model_utils import compute_gradient_norm


class TestKrumAggregator(unittest.TestCase):
    """Test Krum aggregation."""

    def setUp(self):
        self.aggregator = KrumAggregator(num_malicious=2)

    def test_aggregation_shape(self):
        """Test that aggregation preserves shape."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        aggregated, metrics = self.aggregator.aggregate(gradients)

        self.assertEqual(len(aggregated), 3)
        self.assertEqual(aggregated[0].shape, (5, 10))

    def test_selects_honest_gradient(self):
        """Test that Krum selects honest gradient under attack."""
        honest_gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(8)]
        malicious_gradients = [[np.random.randn(5, 10) * 10.0 for _ in range(3)] for _ in range(2)]

        all_gradients = honest_gradients + malicious_gradients

        aggregated, metrics = self.aggregator.aggregate(all_gradients)

        # Selected index should be from honest ones (0-7)
        selected_idx = metrics.get('selected_idx', -1)
        self.assertLess(selected_idx, 8)

    def test_metrics(self):
        """Test that metrics are returned."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        aggregated, metrics = self.aggregator.aggregate(gradients)

        self.assertIn('selected_idx', metrics)
        self.assertIn('aggregation_method', metrics)


class TestMultiKrumAggregator(unittest.TestCase):
    """Test Multi-Krum aggregation."""

    def setUp(self):
        self.aggregator = MultiKrumAggregator(num_malicious=2, k=5)

    def test_aggregates_multiple(self):
        """Test that Multi-Krum averages multiple gradients."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        aggregated, metrics = self.aggregator.aggregate(gradients)

        self.assertEqual(len(aggregated), 3)
        self.assertEqual(aggregated[0].shape, (5, 10))

    def test_selects_k_gradients(self):
        """Test that k gradients are selected."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        aggregated, metrics = self.aggregator.aggregate(gradients)

        selected_indices = metrics.get('selected_indices', [])
        self.assertEqual(len(selected_indices), 5)


class TestTrimmedMeanAggregator(unittest.TestCase):
    """Test Trimmed Mean aggregation."""

    def setUp(self):
        self.aggregator = TrimmedMeanAggregator(trim_ratio=0.2)

    def test_trims_outliers(self):
        """Test that outliers are trimmed."""
        gradients = []
        for i in range(10):
            if i < 8:
                grad = [np.random.randn(5, 10) * 0.1 for _ in range(3)]
            else:
                grad = [np.random.randn(5, 10) * 10.0 for _ in range(3)]
            gradients.append(grad)

        aggregated, metrics = self.aggregator.aggregate(gradients)

        agg_norm = compute_gradient_norm(aggregated)

        # Should be smaller than outlier norm
        self.assertLess(agg_norm, 5.0)


if __name__ == '__main__':
    unittest.main()
