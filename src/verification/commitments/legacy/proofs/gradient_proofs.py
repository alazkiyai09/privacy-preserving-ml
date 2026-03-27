"""
Gradient Norm Proofs

Zero-knowledge proofs for gradient norm bounds.
"""

import numpy as np
from typing import List, Dict, Any

from ..models.model_utils import compute_gradient_norm


def generate_gradient_norm_proof(
    gradient: List[np.ndarray],
    bound: float
) -> Dict[str, Any]:
    """
    Generate proof that gradient L2 norm is bounded.

    Args:
        gradient: Model gradient (list of arrays)
        bound: Maximum allowed norm

    Returns:
        Proof dictionary

    Note:
        This is a simplified implementation. In production, would use
        ZK proof system from Day 9 (zkp_fl_verification).
    """
    actual_norm = compute_gradient_norm(gradient)

    return {
        "type": "gradient_norm_bound",
        "bound": bound,
        "actual_norm": float(actual_norm),
        "verified": actual_norm <= bound,
        "proof_data": f"norm_{actual_norm:.6f}_le_bound_{bound}"
    }


def verify_gradient_norm_proof(
    proof: Dict[str, Any],
    expected_bound: float
) -> bool:
    """
    Verify gradient norm bound proof.

    Args:
        proof: Proof dictionary
        expected_bound: Expected maximum norm

    Returns:
        True if proof is valid
    """
    if not isinstance(proof, dict):
        return False

    if proof.get("type") != "gradient_norm_bound":
        return False

    actual_norm = proof.get("actual_norm", float('inf'))
    bound = proof.get("bound", expected_bound)

    return actual_norm <= bound and bound == expected_bound
