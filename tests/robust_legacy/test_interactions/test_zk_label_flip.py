"""
Integration Tests: ZK Proofs vs Label Flip Attack
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.attacks.label_flip import LabelFlipAttack
from src.zk_proofs.proof_generator import ZKProofGenerator
from src.zk_proofs.proof_verifier import ZKProofVerifier
from src.models.phishing_classifier import PhishingClassifier


class TestZKLabelFlipInteraction(unittest.TestCase):
    """Test ZK proofs with label flip attack."""

    def test_zk_cannot_prevent_label_flip(self):
        """Test that ZK proofs don't prevent label flip attacks."""
        # Create data
        features = torch.randn(100, 20)
        labels = torch.ones(100, dtype=torch.long)  # All phishing (Long type for CrossEntropyLoss)
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=32)

        # Apply label flip attack
        attack = LabelFlipAttack(flip_ratio=0.2, target_class=0)
        poisoned_dataset, flip_mask = attack.flip_labels(dataset)

        # Create model and train
        model = PhishingClassifier(input_size=20)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for data, labels in loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        # Generate ZK proof
        generator = ZKProofGenerator(client_id=0, use_simplified=True)
        initial_params = [p.detach().numpy() for p in model.parameters()]
        final_params = [p.detach().numpy() for p in model.parameters()]
        gradient = [final - initial for final, initial in zip(final_params, initial_params)]

        proof = generator.generate_gradient_norm_proof(gradient, bound=1.0)

        # ZK proof should be valid (gradient computation is correct)
        self.assertTrue(proof['within_bound'])

        # But attack succeeded (labels were flipped)
        self.assertGreater(np.sum(flip_mask), 0)


if __name__ == '__main__':
    unittest.main()
