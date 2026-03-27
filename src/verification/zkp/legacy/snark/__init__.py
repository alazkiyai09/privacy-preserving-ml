"""
ZK-SNARK Primitives

This module provides the building blocks for zero-knowledge succinct
non-interactive arguments of knowledge.
"""

from .circuits import ArithmeticCircuit
from .r1cs import R1CS, QAP

__all__ = ["ArithmeticCircuit", "R1CS", "QAP"]
