"""
Integration Tests: Combined Defense System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np

from src.attacks.model_poisoning import ModelPoisoningAttack
from src.defenses.byzantine_aggregation import KrumAggregator
from src.defenses.anomaly_detection import ZScoreDetector


class TestCombinedDefenses(unittest.TestCase):
    """Test combined defense system."""

    def test_byzantine_detects_scaling(self):
        """Test that Byzantine aggregation catches scaling attacks."""
        # Create gradients
        honest_gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(8)]
        attack = ModelPoisoningAttack(attack_type="scaling", scaling_factor=10.0)

        malicious_gradients = honest_gradients.copy()
        for i in [2, 5]:
            malicious_gradients[i] = attack.poison_gradient(malicious_gradients[i])

        # Apply Krum
        aggregator = KrumAggregator(num_malicious=2)
        aggregated, metrics = aggregator.aggregate(malicious_gradients)

        # Selected should not be one of the malicious ones
        selected_idx = metrics['selected_idx']
        self.assertNotIn(selected_idx, [2, 5])

    def test_anomaly_detection_catches_scaling(self):
        """Test that anomaly detection catches scaling attacks."""
        honest_gradients = [[np.random.randn(5, 10) * 0.1 for _ in range(3)] for _ in range(8)]

        attack = ModelPoisoningAttack(attack_type="scaling", scaling_factor=10.0)
        malicious_gradient = attack.poison_gradient(honest_gradients[0])

        all_gradients = honest_gradients + [malicious_gradient]

        detector = ZScoreDetector(threshold=2.0)
        anomalies = detector.detect(all_gradients)

        # Last one (malicious) should be flagged
        self.assertTrue(anomalies[-1])


if __name__ == '__main__':
    unittest.main()
