"""
Zero-Knowledge Proofs for Verifiable Federated Learning

This module implements ZK proof systems for verifying FL client updates:

Proofs:
- Gradient Norm Bound: Prove gradient norm ≤ bound (prevents scaling)
- Participation: Prove trained on ≥ min_samples (prevents free-riding)
- Training Correctness: Prove training executed correctly (prevents tampering)

Important:
- ZK proofs prevent gradient scaling and free-riding
- ZK proofs CANNOT prevent label flips or backdoors
- Use with Byzantine aggregation for complete defense
"""

from .proof_generator import ZKProofGenerator
from .proof_verifier import ZKProofVerifier
from .norm_bound_proof import GradientNormProof

__all__ = [
    "ZKProofGenerator",
    "ZKProofVerifier",
    "GradientNormProof",
]
