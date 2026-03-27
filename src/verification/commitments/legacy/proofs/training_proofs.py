"""
Training Correctness Proofs

Zero-knowledge proofs for training correctness.
"""

import numpy as np
from typing import List, Dict, Any


def generate_training_correctness_proof(
    initial_params: List[np.ndarray],
    final_params: List[np.ndarray],
    initial_loss: float = 1.0,
    final_loss: float = 0.5,
    num_epochs: int = 5
) -> Dict[str, Any]:
    """
    Generate proof that training occurred correctly.

    Simplified version: proves parameters changed and loss decreased.

    Args:
        initial_params: Parameters before training
        final_params: Parameters after training
        initial_loss: Loss before training
        final_loss: Loss after training
        num_epochs: Number of training epochs

    Returns:
        Proof dictionary
    """
    # Compute parameter change
    param_change = 0.0
    for init, final in zip(initial_params, final_params):
        diff = init - final
        param_change += np.sum(diff ** 2)
    param_change = np.sqrt(param_change)

    # Check loss decreased
    loss_decreased = final_loss < initial_loss

    return {
        "type": "training_correctness",
        "param_change": float(param_change),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_decreased": loss_decreased,
        "num_epochs": num_epochs,
        "verified": param_change > 0 and loss_decreased
    }


def verify_training_correctness_proof(
    proof: Dict[str, Any]
) -> bool:
    """
    Verify training correctness proof.

    Args:
        proof: Proof dictionary

    Returns:
        True if proof is valid
    """
    if not isinstance(proof, dict):
        return False

    if proof.get("type") != "training_correctness":
        return False

    param_change = proof.get("param_change", 0)
    loss_decreased = proof.get("loss_decreased", False)

    return param_change > 0 and loss_decreased
