"""
Integration Tests
"""

import sys
sys.path.insert(0, '../src')

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.phishing_classifier import SimplePhishingClassifier
from src.fl.client import VerifiableFLClient
from src.proofs.proof_aggregator import ProofAggregator


def test_end_to_end_verification():
    """Test end-to-end client training and verification."""
    # Setup
    model = SimplePhishingClassifier(input_size=10, num_classes=2)

    features = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=20)

    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 5.0,
        "min_samples": 50,
        "local_epochs": 1,
        "learning_rate": 0.01
    }

    # Create client
    client = VerifiableFLClient(
        model=model,
        train_loader=loader,
        client_id=0,
        proof_config=proof_config
    )

    # Train
    initial_params = client.get_parameters({})
    new_params, num_samples, metrics = client.fit(
        initial_params,
        {"local_epochs": 1, "learning_rate": 0.01}
    )

    # Verify proofs
    verifier = ProofAggregator(verify_all_proofs=True)

    proofs = {
        "gradient_norm_proof": metrics.get("gradient_norm_proof"),
        "participation_proof": metrics.get("participation_proof"),
        "training_correctness_proof": metrics.get("training_correctness_proof")
    }

    is_valid, failed = verifier.verify_all_proofs(proofs, metrics)

    assert is_valid == True
    assert len(failed) == 0
    assert num_samples == 100

    print("✓ End-to-end verification test passed")
    print(f"  - Client trained on {num_samples} samples")
    print(f"  - All proofs verified successfully")
    print(f"  - Loss: {metrics['loss']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")


def test_malicious_client_detection():
    """Test detection of malicious client."""
    from src.fl.client import MaliciousClient

    # Setup
    model = SimplePhishingClassifier(input_size=10, num_classes=2)
    features = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=20)

    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 5.0,
        "min_samples": 50,
        "local_epochs": 1,
        "learning_rate": 0.01
    }

    # Create malicious client
    malicious_client = MaliciousClient(
        model=model,
        train_loader=loader,
        client_id=0,
        proof_config=proof_config,
        attack_type="gradient_scaling",
        attack_strength=10.0
    )

    # Train (will perform attack)
    initial_params = malicious_client.get_parameters({})
    new_params, num_samples, metrics = malicious_client.fit(
        initial_params,
        {"local_epochs": 1, "learning_rate": 0.01}
    )

    # Verify proofs - should fail
    verifier = ProofAggregator(verify_all_proofs=True)

    proofs = {
        "gradient_norm_proof": metrics.get("gradient_norm_proof"),
        "participation_proof": metrics.get("participation_proof"),
        "training_correctness_proof": metrics.get("training_correctness_proof")
    }

    is_valid, failed = verifier.verify_all_proofs(proofs, metrics)

    # Malicious client should fail verification
    assert is_valid == False
    assert len(failed) > 0
    assert "gradient_norm" in failed

    print("✓ Malicious client detection test passed")
    print(f"  - Attack type: gradient_scaling")
    print(f"  - Attack strength: 10.0x")
    print(f"  - Detected: YES")
    print(f"  - Failed proofs: {failed}")


if __name__ == "__main__":
    test_end_to_end_verification()
    test_malicious_client_detection()
    print("\n✓ All integration tests passed!")
