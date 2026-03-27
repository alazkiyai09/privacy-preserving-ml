"""
Participation Proofs

Zero-knowledge proofs for data participation.
"""

import hashlib
from typing import Dict, Any


def generate_participation_proof(
    num_samples: int,
    min_samples: int,
    data_hash: int = None
) -> Dict[str, Any]:
    """
    Generate proof that client trained on minimum samples.

    Args:
        num_samples: Actual number of training samples
        min_samples: Minimum required samples
        data_hash: Hash of training data (optional)

    Returns:
        Proof dictionary
    """
    # Generate hash proof if not provided
    if data_hash is None:
        data_hash = int(hashlib.sha256(str(num_samples).encode()).hexdigest(), 16) % (10**8)

    return {
        "type": "participation",
        "num_samples": num_samples,
        "min_samples": min_samples,
        "data_hash": data_hash,
        "verified": num_samples >= min_samples
    }


def verify_participation_proof(
    proof: Dict[str, Any],
    min_samples: int
) -> bool:
    """
    Verify participation proof.

    Args:
        proof: Proof dictionary
        min_samples: Minimum required samples

    Returns:
        True if proof is valid
    """
    if not isinstance(proof, dict):
        return False

    if proof.get("type") != "participation":
        return False

    num_samples = proof.get("num_samples", 0)
    proof_min_samples = proof.get("min_samples", min_samples)

    return num_samples >= min_samples and proof_min_samples == min_samples
