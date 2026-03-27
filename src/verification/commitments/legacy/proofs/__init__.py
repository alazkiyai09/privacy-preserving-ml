"""
Proof Modules

Zero-knowledge proof generation and verification for federated learning.
"""

from .proof_aggregator import ProofAggregator
from .gradient_proofs import generate_gradient_norm_proof, verify_gradient_norm_proof
from .training_proofs import generate_training_correctness_proof, verify_training_correctness_proof
from .participation_proofs import generate_participation_proof, verify_participation_proof

__all__ = [
    "ProofAggregator",
    "generate_gradient_norm_proof",
    "verify_gradient_norm_proof",
    "generate_training_correctness_proof",
    "verify_training_correctness_proof",
    "generate_participation_proof",
    "verify_participation_proof"
]
