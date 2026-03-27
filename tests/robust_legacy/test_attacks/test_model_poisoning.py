"""
Unit Tests for Model Poisoning Attack
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np

from src.attacks.model_poisoning import ModelPoisoningAttack, AdaptiveModelPoisoningAttack, compute_gradient_norm


class TestModelPoisoningAttack(unittest.TestCase):
    """Test model poisoning attack."""

    def setUp(self):
        self.attack = ModelPoisoningAttack(attack_type="scaling", scaling_factor=10.0)

    def test_scaling_attack(self):
        """Test gradient scaling attack."""
        gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]
        original_norm = compute_gradient_norm(gradient)

        poisoned_gradient = self.attack.poison_gradient(gradient)
        poisoned_norm = compute_gradient_norm(poisoned_gradient)

        self.assertGreater(poisoned_norm, original_norm)
        self.assertAlmostEqual(poisoned_norm / original_norm, 10.0, places=1)

    def test_sign_flip_attack(self):
        """Test sign flip attack."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")

        gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]
        original_norm = compute_gradient_norm(gradient)

        poisoned_gradient = attack.poison_gradient(gradient)
        poisoned_norm = compute_gradient_norm(poisoned_gradient)

        # Norm should be same
        self.assertAlmostEqual(poisoned_norm, original_norm, places=5)

    def test_isotropic_attack(self):
        """Test isotropic (noise) attack."""
        attack = ModelPoisoningAttack(attack_type="isotropic", noise_std=5.0)

        gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]

        poisoned_gradient = attack.poison_gradient(gradient)

        # Should have same shape
        self.assertEqual(len(poisoned_gradient), len(gradient))

    def test_adaptive_within_bounds(self):
        """Test adaptive attack stays within ZK bound."""
        adaptive_attack = AdaptiveModelPoisoningAttack(zk_bound=1.0, base_attack_type="scaling")

        honest_gradient = [np.random.randn(5, 10) * 0.1 for _ in range(3)]

        malicious_gradient, info = adaptive_attack.craft_within_bounds(
            honest_gradient, attack_type="scaling"
        )

        # Should stay within bound
        self.assertTrue(info["within_bound"])
        self.assertLessEqual(info["new_norm"], 1.0)


if __name__ == '__main__':
    unittest.main()
