"""
Test Verifiable FL Client
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.phishing_classifier import SimplePhishingClassifier
from src.fl.client import VerifiableFLClient


def test_client_initialization():
    """Test client initialization."""
    model = SimplePhishingClassifier(input_size=10, num_classes=2)

    features = torch.randn(50, 10)
    labels = torch.randint(0, 2, (50,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=10)

    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 5.0,
        "min_samples": 10
    }

    client = VerifiableFLClient(
        model=model,
        train_loader=loader,
        client_id=0,
        proof_config=proof_config
    )

    assert client.client_id == 0
    assert client.enable_proofs == True
    print("✓ Client initialization test passed")


def test_client_training():
    """Test client training."""
    model = SimplePhishingClassifier(input_size=10, num_classes=2)

    features = torch.randn(50, 10)
    labels = torch.randint(0, 2, (50,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=10)

    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 5.0,
        "min_samples": 10,
        "local_epochs": 1,
        "learning_rate": 0.01
    }

    client = VerifiableFLClient(
        model=model,
        train_loader=loader,
        client_id=0,
        proof_config=proof_config
    )

    initial_params = client.get_parameters({})

    new_params, num_samples, metrics = client.fit(
        initial_params,
        {"local_epochs": 1, "learning_rate": 0.01}
    )

    assert num_samples == 50
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "gradient_norm" in metrics

    if proof_config["enable_proofs"]:
        assert "gradient_norm_proof" in metrics
        assert "participation_proof" in metrics
        assert "training_correctness_proof" in metrics

    print("✓ Client training test passed")


def test_proof_generation():
    """Test proof generation."""
    from src.proofs.gradient_proofs import generate_gradient_norm_proof
    from src.proofs.training_proofs import generate_training_correctness_proof
    from src.proofs.participation_proofs import generate_participation_proof

    # Gradient proof
    gradient = [np.random.randn(10, 10) * 0.1]
    proof = generate_gradient_norm_proof(gradient, bound=1.0)
    assert proof["type"] == "gradient_norm_bound"
    assert proof["verified"] == True

    # Training proof
    training_proof = generate_training_correctness_proof(
        [np.random.randn(10, 10)],
        [np.random.randn(10, 10)],
        initial_loss=1.0,
        final_loss=0.5
    )
    assert training_proof["type"] == "training_correctness"

    # Participation proof
    participation_proof = generate_participation_proof(100, 50)
    assert participation_proof["type"] == "participation"
    assert participation_proof["verified"] == True

    print("✓ Proof generation test passed")


if __name__ == "__main__":
    test_client_initialization()
    test_client_training()
    test_proof_generation()
    print("\n✓ All tests passed!")
