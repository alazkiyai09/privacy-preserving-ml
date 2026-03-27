"""
Commitment Schemes

Pedersen commitment for gradient binding.
"""

import numpy as np
from typing import List, Tuple
import random


def commit_to_gradient(
    gradient: List[np.ndarray],
    randomness: int = None
) -> Tuple[int, int]:
    """
    Commit to gradient using simplified commitment scheme.

    Args:
        gradient: Gradient array
        randomness: Optional randomness value

    Returns:
        (commitment, randomness)

    Note:
        This is a simplified commitment. In production, would use
        Pedersen commitment from Day 9 ZK library.
    """
    if randomness is None:
        randomness = random.randint(1, 2**32)

    # Compute gradient hash
    grad_hash = hash(gradient.tobytes())

    # Simple commitment: C = hash(gradient, randomness)
    commitment = hash((grad_hash, randomness))

    return commitment, randomness


def open_gradient_commitment(
    commitment: int,
    gradient: List[np.ndarray],
    randomness: int
) -> bool:
    """
    Verify commitment opens to gradient.

    Args:
        commitment: Commitment value
        gradient: Gradient array
        randomness: Randomness used

    Returns:
        True if valid
    """
    grad_hash = hash(gradient.tobytes())
    expected = hash((grad_hash, randomness))

    return commitment == expected
