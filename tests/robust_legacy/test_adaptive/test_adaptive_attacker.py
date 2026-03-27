"""
Unit Tests for Adaptive Attacker
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np

from src.attacks.adaptive_attacker import AdaptiveAttacker


class TestAdaptiveAttacker(unittest.TestCase):
    """Test adaptive attacker."""

    def setUp(self):
        self.attacker = AdaptiveAttacker(
            client_id=0,
            knows_zk_bound=True,
            zk_bound=1.0
        )

    def test_craft_attack_returns_gradient(self):
        """Test that craft_attack returns valid gradient."""
        honest_gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]

        malicious_gradient, attack_info = self.attacker.craft_attack(
            honest_gradient,
            {"zk_bound": 1.0}
        )

        self.assertIsNotNone(malicious_gradient)
        self.assertIsInstance(malicious_gradient, list)

    def test_stays_within_zk_bound(self):
        """Test that ZK-aware attacker stays within bound."""
        honest_gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]

        # Pass round > 5 to skip reputation building phase
        malicious_gradient, info = self.attacker.craft_attack(
            honest_gradient,
            {"zk_bound": 1.0, "round": 10}
        )

        # When not building reputation, should have within_bound key
        self.assertIn("within_bound", info)
        self.assertTrue(info["within_bound"])

    def test_builds_reputation_first(self):
        """Test that sophisticated attacker builds reputation."""
        from src.attacks.adaptive_attacker import SophisticatedAttacker

        attacker = SophisticatedAttacker(
            client_id=0,
            zk_bound=1.0,
            reputation_build_rounds=5
        )

        honest_gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]

        # Round 1: Should act honest
        malicious, info = attacker.craft_attack(honest_gradient, {"round": 1})

        self.assertEqual(info["attack_type"], "honest")
        self.assertTrue(info.get("building_reputation", False))


if __name__ == '__main__':
    unittest.main()
