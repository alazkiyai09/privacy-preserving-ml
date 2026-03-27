"""
Federated Learning Proofs

Zero-knowledge proofs specifically designed for federated learning verification.
"""

from .gradient_bounds import GradientBoundProof
from .data_validity import DataValidityProof
from .computation import ComputationProof

__all__ = ["GradientBoundProof", "DataValidityProof", "ComputationProof"]
