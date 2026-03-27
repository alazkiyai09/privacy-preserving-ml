"""
Test Proof Verification
"""

import sys
sys.path.insert(0, '../src')

from src.proofs.proof_aggregator import ProofAggregator


def test_proof_verification():
    """Test proof verification."""
    verifier = ProofAggregator(verify_all_proofs=True)

    # Valid proofs
    valid_proofs = {
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
        }
    }

    valid_metrics = {
        "gradient_norm": 1.5,
        "num_samples": 100
    }

    is_valid, failed = verifier.verify_all_proofs(valid_proofs, valid_metrics)
    assert is_valid == True
    assert len(failed) == 0

    print("✓ Valid proof verification test passed")

    # Invalid proofs (gradient too large)
    invalid_proofs = {
        "gradient_norm_proof": {
            "type": "gradient_norm_bound",
            "bound": 5.0,
            "actual_norm": 10.0,
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
            "param_change": 0.5,
            "verified": True
        }
    }

    invalid_metrics = {
        "gradient_norm": 10.0,
        "num_samples": 100
    }

    is_valid, failed = verifier.verify_all_proofs(invalid_proofs, invalid_metrics)
    assert is_valid == False
    assert len(failed) > 0
    assert "gradient_norm" in failed

    print("✓ Invalid proof detection test passed")


def test_batch_verification():
    """Test batch verification."""
    verifier = ProofAggregator(verify_all_proofs=True)

    # Multiple clients
    client_proofs = [
        {
            "gradient_norm_proof": {"type": "gradient_norm_bound", "bound": 5.0, "actual_norm": 1.0, "verified": True},
            "participation_proof": {"type": "participation", "num_samples": 100, "min_samples": 50, "verified": True},
            "training_correctness_proof": {"type": "training_correctness", "param_change": 0.5, "verified": True}
        },
        {
            "gradient_norm_proof": {"type": "gradient_norm_bound", "bound": 5.0, "actual_norm": 10.0, "verified": False},
            "participation_proof": {"type": "participation", "num_samples": 100, "min_samples": 50, "verified": True},
            "training_correctness_proof": {"type": "training_correctness", "param_change": 0.5, "verified": True}
        }
    ]

    client_metrics = [
        {"gradient_norm": 1.0, "num_samples": 100},
        {"gradient_norm": 10.0, "num_samples": 100}
    ]

    results = verifier.batch_verify_clients(client_proofs, client_metrics)

    assert len(results) == 2
    assert results[0] == True  # First client valid
    assert results[1] == False  # Second client invalid

    print("✓ Batch verification test passed")


if __name__ == "__main__":
    test_proof_verification()
    test_batch_verification()
    print("\n✓ All proof tests passed!")
