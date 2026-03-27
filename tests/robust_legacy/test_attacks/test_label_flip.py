"""
Unit Tests for Label Flip Attack
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.attacks.label_flip import LabelFlipAttack, create_label_flip_data


class TestLabelFlipAttack(unittest.TestCase):
    """Test label flip attack functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.attack = LabelFlipAttack(flip_ratio=0.2, target_class=0)

    def test_flip_ratio(self):
        """Test that flip ratio is respected."""
        # Create data with known number of phishing samples
        features = torch.randn(100, 10)
        labels = torch.zeros(100)
        # Set first 30 as phishing
        labels[:30] = 1
        dataset = TensorDataset(features, labels)

        poisoned_dataset, flip_mask = self.attack.flip_labels(dataset)

        num_flipped = np.sum(flip_mask)
        # In targeted mode, flip_ratio applies to phishing samples only
        # 30 phishing * 0.2 = 6 flips
        expected_flips = int(30 * 0.2)

        self.assertEqual(num_flipped, expected_flips)

    def test_flip_changes_labels(self):
        """Test that labels are actually changed."""
        features = torch.randn(50, 10)
        labels = torch.ones(50)  # All phishing
        dataset = TensorDataset(features, labels)

        poisoned_dataset, flip_mask = self.attack.flip_labels(dataset)

        original_labels = labels.numpy()
        poisoned_labels = poisoned_dataset.tensors[1].numpy()

        # Check that flipped indices have target class
        for idx in np.where(flip_mask)[0]:
            self.assertEqual(poisoned_labels[idx], 0)

    def test_features_unchanged(self):
        """Test that features are not modified."""
        features = torch.randn(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, flip_mask = self.attack.flip_labels(dataset)

        original_features = dataset.tensors[0].numpy()
        poisoned_features = poisoned_dataset.tensors[0].numpy()

        np.testing.assert_array_equal(original_features, poisoned_features)

    def test_create_label_flip_data(self):
        """Test helper function."""
        loader, flip_mask = create_label_flip_data(
            num_samples=200,
            num_features=20,
            flip_ratio=0.15,
            phishing_ratio=0.3  # 60 phishing samples
        )

        # Check data shape
        for data, labels in loader:
            self.assertEqual(data.shape[1], 20)
            break

        # Check flip mask (15% of 60 phishing = 9 flips)
        num_flipped = np.sum(flip_mask)
        expected = int(200 * 0.3 * 0.15)  # 60 phishing * 0.15 = 9
        self.assertEqual(num_flipped, expected)

    def test_targeted_flip(self):
        """Test targeted flipping to specific class."""
        attack = LabelFlipAttack(
            flip_ratio=0.3,
            flip_strategy="targeted",
            target_class=1
        )

        features = torch.randn(100, 10)
        labels = torch.zeros(100)  # All legitimate
        dataset = TensorDataset(features, labels)

        # No phishing samples to flip, should return empty flip mask
        poisoned_dataset, flip_mask = attack.flip_labels(dataset)
        self.assertEqual(np.sum(flip_mask), 0)

        # Now test with actual phishing samples
        labels[:50] = 1  # 50 phishing samples
        dataset = TensorDataset(features, labels)
        poisoned_dataset, flip_mask = attack.flip_labels(dataset)

        # Should flip 30% of 50 phishing = 15 samples
        self.assertEqual(np.sum(flip_mask), 15)

        # All flipped should be class 1
        poisoned_labels = poisoned_dataset.tensors[1].numpy()
        for idx in np.where(flip_mask)[0]:
            self.assertEqual(poisoned_labels[idx], 1)

    def test_get_attack_impact(self):
        """Test attack impact analysis."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, flip_mask = self.attack.flip_labels(dataset)

        # Create dummy predictions
        y_true = labels.numpy()
        y_pred = labels.numpy()  # Perfect predictions for testing

        # Get impact
        impact = self.attack.get_attack_impact(y_true, y_pred, flip_mask)

        self.assertIn("num_flipped", impact)
        self.assertIn("flip_ratio", impact)
        self.assertIn("flip_strategy", impact)
        self.assertIn("attack_success_rate", impact)


if __name__ == '__main__':
    unittest.main()
