#!/usr/bin/env python3
"""
Simple Verifiable FL Demo

Minimal working example of verifiable federated learning.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.phishing_classifier import SimplePhishingClassifier
from src.models.model_utils import parameters_to_ndarrays, set_model_params, compute_gradient_norm
from src.fl.client import VerifiableFLClient
from src.utils.data_loader import PhishingDataset


def create_simple_data():
    """Create simple training data."""
    # Features: 100 samples, 10 features
    features = torch.randn(100, 10)

    # Labels: binary classification
    labels = torch.randint(0, 2, (100,))

    # Create dataset
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    return loader


def demo_client_training():
    """Demonstrate client training with proof generation."""
    print("="*70)
    print("VERIFIABLE FL CLIENT DEMO")
    print("="*70)
    print()

    # Create model
    model = SimplePhishingClassifier(input_size=10, num_classes=2)

    # Create data
    train_loader = create_simple_data()

    # Create client with proofs enabled
    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 5.0,
        "min_samples": 50,
        "local_epochs": 2,
        "learning_rate": 0.01
    }

    client = VerifiableFLClient(
        model=model,
        train_loader=train_loader,
        client_id=0,
        proof_config=proof_config,
        device="cpu"
    )

    print("Client Configuration:")
    print(f"  Proof generation: ENABLED")
    print(f"  Gradient bound: {proof_config['gradient_bound']}")
    print(f"  Min samples: {proof_config['min_samples']}")
    print(f"  Local epochs: {proof_config['local_epochs']}")
    print()

    # Get initial parameters
    initial_params = client.get_parameters({})

    # Train client
    print("Training client...")
    print("-"*70)

    config = {
        "local_epochs": 2,
        "learning_rate": 0.01
    }

    new_params, num_samples, metrics = client.fit(initial_params, config)

    print("-"*70)
    print()
    print("Training Results:")
    print(f"  Samples trained: {num_samples}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Gradient norm: {metrics['gradient_norm']:.4f}")
    print(f"  Training time: {metrics['training_time']:.2f}s")
    print()

    # Check proofs
    if "gradient_norm_proof" in metrics:
        print("Generated Proofs:")
        print(f"  ✓ Gradient norm proof")
        print(f"    - Type: {metrics['gradient_norm_proof']['type']}")
        print(f"    - Bound: {metrics['gradient_norm_proof']['bound']}")
        print(f"    - Actual norm: {metrics['gradient_norm_proof']['actual_norm']:.4f}")
        print(f"    - Verified: {metrics['gradient_norm_proof']['verified']}")
        print()

        print(f"  ✓ Participation proof")
        print(f"    - Samples: {metrics['participation_proof']['num_samples']}")
        print(f"    - Verified: {metrics['participation_proof']['verified']}")
        print()

        print(f"  ✓ Training correctness proof")
        print(f"    - Parameter change: {metrics['training_correctness_proof']['param_change']:.4f}")
        print(f"    - Verified: {metrics['training_correctness_proof']['verified']}")
        print()

        # Check verification status
        all_verified = (
            metrics['gradient_norm_verified'] and
            metrics['participation_verified'] and
            metrics['training_correctness_verified']
        )

        if all_verified:
            print("✓ ALL PROOFS VERIFIED - Client update is valid!")
        else:
            print("✗ SOME PROOFS FAILED - Client update would be rejected")

        print()
        print("Proof Generation Overhead:")
        print(f"  Proof generation time: {metrics.get('proof_generation_time', 0):.3f}s")
        print(f"  Percentage of training: {(metrics.get('proof_generation_time', 0) / metrics['training_time'] * 100):.1f}%")

    else:
        print("No proofs generated (proofs disabled)")

    print()
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)


def demo_proof_verification():
    """Demonstrate proof verification."""
    print()
    print("="*70)
    print("PROOF VERIFICATION DEMO")
    print("="*70)
    print()

    from src.proofs.proof_aggregator import ProofAggregator

    # Create proof verifier
    verifier = ProofAggregator(verify_all_proofs=True)

    # Example client metrics (with proofs)
    honest_client = {
        "gradient_norm_proof": {
            "type": "gradient_norm_bound",
            "bound": 5.0,
            "actual_norm": 1.5,
            "verified": True
        },
        "participation_proof": {
            "type": "participation",
            "num_samples": 100,
            "min_samples": 50,
            "verified": True
        },
        "training_correctness_proof": {
            "type": "training_correctness",
            "param_change": 0.5,
            "verified": True
        },
        "gradient_norm_verified": True,
        "participation_verified": True,
        "training_correctness_verified": True
    }

    # Malicious client (gradient too large)
    malicious_client = {
        "gradient_norm_proof": {
            "type": "gradient_norm_bound",
            "bound": 5.0,
            "actual_norm": 10.0,  # Exceeds bound!
            "verified": False
        },
        "participation_proof": {
            "type": "participation",
            "num_samples": 100,
            "min_samples": 50,
            "verified": True
        },
        "training_correctness_proof": {
            "type": "training_correctness",
            "param_change": 0.8,
            "verified": True
        },
        "gradient_norm_verified": False,  # Failed!
        "participation_verified": True,
        "training_correctness_verified": True
    }

    print("Verifying honest client...")
    is_valid_grad = verifier._verify_gradient_proof(
        honest_client["gradient_norm_proof"],
        honest_client
    )
    is_valid_part = verifier._verify_participation_proof(
        honest_client["participation_proof"],
        honest_client
    )
    is_valid_train = verifier._verify_training_proof(
        honest_client["training_correctness_proof"],
        honest_client
    )
    all_valid = is_valid_grad and is_valid_part and is_valid_train

    print(f"  Overall: {'VALID ✓' if all_valid else 'INVALID ✗'}")

    print()
    print("Verifying malicious client (gradient scaling attack)...")
    is_valid_grad = verifier._verify_gradient_proof(
        malicious_client["gradient_norm_proof"],
        malicious_client
    )
    is_valid_part = verifier._verify_participation_proof(
        malicious_client["participation_proof"],
        malicious_client
    )
    is_valid_train = verifier._verify_training_proof(
        malicious_client["training_correctness_proof"],
        malicious_client
    )
    all_valid = is_valid_grad and is_valid_part and is_valid_train

    failed = []
    if not is_valid_grad:
        failed.append("gradient_norm")
    if not is_valid_part:
        failed.append("participation")
    if not is_valid_train:
        failed.append("training_correctness")
    print(f"  Result: {'VALID ✓' if all_valid else 'INVALID ✗'}")
    if failed:
        print(f"  Failed proofs: {failed}")
        print(f"  → ATTACK DETECTED AND PREVENTED!")

    print()
    print("="*70)


if __name__ == "__main__":
    # Run demo
    demo_client_training()
    demo_proof_verification()

    print()
    print("Next steps:")
    print("  1. Run baseline: python experiments/run_baselines.py")
    print("  2. Run verifiable FL: python experiments/run_verifiable_fl.py")
    print("  3. Simulate attacks: python experiments/run_attacks.py")
    print("  4. Analyze results: python experiments/analyze_results.py")
    print()
