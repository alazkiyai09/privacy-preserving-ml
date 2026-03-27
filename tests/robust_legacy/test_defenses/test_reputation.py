"""
Unit Tests for Reputation System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest

from src.defenses.reputation_system import ClientReputationSystem


class TestClientReputationSystem(unittest.TestCase):
    """Test client reputation system."""

    def setUp(self):
        self.reputation = ClientReputationSystem(
            num_clients=10,
            min_reputation=0.3
        )

    def test_initial_scores(self):
        """Test that initial scores are set correctly."""
        for client_id in range(10):
            self.assertEqual(self.reputation.scores[client_id], 0.5)

    def test_good_behavior_increases_score(self):
        """Test that good behavior increases reputation."""
        initial_score = self.reputation.scores[0]

        self.reputation.update_reputation(0, verification_passed=True, gradient_anomaly_score=0.0)

        self.assertGreater(self.reputation.scores[0], initial_score)

    def test_bad_behavior_decreases_score(self):
        """Test that bad behavior decreases reputation."""
        initial_score = self.reputation.scores[0]

        self.reputation.update_reputation(0, verification_passed=False, gradient_anomaly_score=0.8)

        self.assertLess(self.reputation.scores[0], initial_score)

    def test_should_exclude_low_reputation(self):
        """Test that low-reputation clients are excluded."""
        self.reputation.scores[0] = 0.1

        self.assertTrue(self.reputation.should_exclude(0))

    def test_get_statistics(self):
        """Test getting reputation statistics."""
        self.reputation.update_reputation(0, True, 0.0)
        self.reputation.update_reputation(1, False, 0.8)

        stats = self.reputation.get_reputation_statistics()

        self.assertIn('num_active', stats)
        self.assertIn('mean_reputation', stats)


if __name__ == '__main__':
    unittest.main()
