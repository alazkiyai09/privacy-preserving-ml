"""
Unit Tests for Anomaly Detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np

from src.defenses.anomaly_detection import (
    ZScoreDetector,
    ClusteringDetector,
    CombinedAnomalyDetector
)


class TestZScoreDetector(unittest.TestCase):
    """Test Z-score anomaly detection."""

    def setUp(self):
        self.detector = ZScoreDetector(threshold=2.0)

    def test_detects_outliers(self):
        """Test that outliers are detected."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(9)]
        outlier = [np.random.randn(5, 10) * 10.0 for _ in range(3)]
        gradients.append(outlier)

        anomalies = self.detector.detect(gradients)

        self.assertTrue(anomalies[-1])  # Last one (outlier) should be flagged

    def test_returns_correct_length(self):
        """Test that detector returns one result per gradient."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        anomalies = self.detector.detect(gradients)

        self.assertEqual(len(anomalies), len(gradients))


class TestClusteringDetector(unittest.TestCase):
    """Test clustering-based anomaly detection."""

    def setUp(self):
        self.detector = ClusteringDetector(eps=0.5, min_samples=3)

    def test_detects_outliers(self):
        """Test that outliers are detected."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(8)]
        outliers = [[np.random.randn(5, 10) * 5.0 for _ in range(3)] for _ in range(2)]
        gradients.extend(outliers)

        anomalies = self.detector.detect(gradients)

        # Should detect at least some
        self.assertGreater(sum(anomalies), 0)


class TestCombinedAnomalyDetector(unittest.TestCase):
    """Test combined anomaly detection."""

    def setUp(self):
        self.detector = CombinedAnomalyDetector(
            methods=["zscore", "clustering"],
            voting="majority"
        )

    def test_combines_methods(self):
        """Test that multiple methods are combined."""
        gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(10)]

        anomalies = self.detector.detect(gradients)

        self.assertEqual(len(anomalies), len(gradients))


if __name__ == '__main__':
    unittest.main()
