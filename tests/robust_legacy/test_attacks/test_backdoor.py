"""
Unit Tests for Backdoor Attack
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.attacks.backdoor import BackdoorAttack, BankingBackdoorAttack, create_backdoor_data


class TestBackdoorAttack(unittest.TestCase):
    """Test backdoor attack functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.attack = BackdoorAttack(
            trigger_type="url_pattern",
            trigger_pattern="http://secure-login",
            target_label=0,
            poison_ratio=0.1
        )

    def test_poison_ratio(self):
        """Test that poison ratio is respected."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = self.attack.insert_backdoor(dataset)

        num_poisoned = len(poisoned_indices)
        expected = int(100 * 0.1)

        self.assertEqual(num_poisoned, expected)

    def test_backdoor_target_label(self):
        """Test that backdoor samples have target label."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = self.attack.insert_backdoor(dataset)

        poisoned_labels = poisoned_dataset.tensors[1].numpy()

        # All poisoned should have target label (0)
        for idx in poisoned_indices:
            self.assertEqual(poisoned_labels[idx], 0)

    def test_url_pattern_trigger(self):
        """Test URL pattern backdoor trigger."""
        attack = BackdoorAttack(
            trigger_type="url_pattern",
            target_label=0
        )

        features = torch.randn(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = attack.insert_backdoor(dataset)
        poisoned_features = poisoned_dataset.tensors[0].numpy()

        # Check that URL trigger is inserted
        for idx in poisoned_indices:
            self.assertEqual(poisoned_features[idx, 0], 999.0)

    def test_bank_name_trigger(self):
        """Test bank name backdoor trigger."""
        attack = BackdoorAttack(
            trigger_type="bank_name",
            target_label=0
        )

        features = torch.randn(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = attack.insert_backdoor(dataset)
        poisoned_features = poisoned_dataset.tensors[0].numpy()

        # Check that bank name trigger is inserted
        for idx in poisoned_indices:
            self.assertEqual(poisoned_features[idx, 1], 888.0)

    def test_create_backdoor_data(self):
        """Test helper function."""
        loader, backdoor_indices = create_backdoor_data(
            num_samples=200,
            num_features=20,
            trigger_type="url_pattern",
            poison_ratio=0.1
        )

        # Check data
        for data, labels in loader:
            self.assertEqual(data.shape[1], 20)
            break

        # Check backdoor count
        expected = int(200 * 0.1)
        self.assertEqual(len(backdoor_indices), expected)

    def test_banking_backdoor(self):
        """Test banking-specific backdoor."""
        attack = BankingBackdoorAttack(
            target_bank="Bank of America",
            target_label=0
        )

        features = torch.randn(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = attack.insert_backdoor(dataset)

        # Should have poisoned samples
        self.assertGreater(len(poisoned_indices), 0)

        # All poisoned should be class 0
        poisoned_labels = poisoned_dataset.tensors[1].numpy()
        for idx in poisoned_indices:
            self.assertEqual(poisoned_labels[idx], 0)

    def test_get_attack_impact(self):
        """Test attack impact analysis."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(features, labels)

        poisoned_dataset, poisoned_indices = self.attack.insert_backdoor(dataset)

        impact = self.attack.get_attack_impact(poisoned_indices, 100)

        self.assertIn("num_poisoned", impact)
        self.assertIn("poison_ratio", impact)
        self.assertIn("trigger_type", impact)


if __name__ == '__main__':
    unittest.main()
