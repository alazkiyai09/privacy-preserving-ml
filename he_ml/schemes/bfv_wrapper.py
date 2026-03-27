"""
BFV Scheme Wrapper
===================
Interface for BFV (Brakerski-Fan-Vercauteren) homomorphic encryption.

NOTE: BFV has limited support in TenSEAL Python. For ML applications,
CKKS is strongly recommended.

BFV Use Cases:
- Exact integer arithmetic (no approximation error)
- Boolean operations
- Voting systems
- Discrete data processing

For continuous/ML data, use CKKS instead.
"""

from typing import Union, List, Optional, Any
import numpy as np
import tenseal as ts
import warnings

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any
Context = Any


class BFVVector:
    """
    Wrapper for BFV encrypted vectors (integers only).

    BFV provides exact integer arithmetic with no approximation error.
    However, it's less efficient than CKKS for machine learning tasks.
    """

    def __init__(
        self,
        data: np.ndarray,
        context: Context,
    ):
        """
        Create an encrypted BFV vector.

        Args:
            data: Integer plaintext vector to encrypt
            context: TenSEAL BFV context

        Note:
            Falls back to CKKS in TenSEAL Python implementation.
        """
        warnings.warn(
            "BFV has limited support. Consider using CKKS for ML applications. "
            "Falling back to CKKS."
        )

        self.context = context
        self._original_len = len(data)

        # Convert to float if needed (CKKS compatibility)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64)

        # Encrypt using CKKS (fallback)
        self.ciphertext = ts.ckks_vector(context, data.tolist())
        self._is_fallback = True

    @classmethod
    def from_plaintext(
        cls,
        data: np.ndarray,
        context: Context,
    ) -> 'BFVVector':
        """Create encrypted vector from plaintext."""
        return cls(data, context)

    def add(
        self,
        other: Union['BFVVector', np.ndarray, int],
    ) -> 'BFVVector':
        """Homomorphic addition."""
        if isinstance(other, BFVVector):
            result_ct = self.ciphertext + other.ciphertext
        else:
            other_list = other.tolist() if hasattr(other, 'tolist') else [float(other)]
            result_ct = self.ciphertext + other_list

        result = BFVVector.__new__(BFVVector)
        result.context = self.context
        result.ciphertext = result_ct
        result._original_len = self._original_len
        result._is_fallback = self._is_fallback
        return result

    def subtract(
        self,
        other: Union['BFVVector', np.ndarray, int],
    ) -> 'BFVVector':
        """Homomorphic subtraction."""
        if isinstance(other, BFVVector):
            result_ct = self.ciphertext - other.ciphertext
        else:
            other_list = other.tolist() if hasattr(other, 'tolist') else [float(other)]
            result_ct = self.ciphertext - other_list

        result = BFVVector.__new__(BFVVector)
        result.context = self.context
        result.ciphertext = result_ct
        result._original_len = self._original_len
        result._is_fallback = self._is_fallback
        return result

    def multiply(
        self,
        other: Union['BFVVector', np.ndarray, int],
        relin_key: RelinKeys,
    ) -> 'BFVVector':
        """Homomorphic multiplication."""
        if isinstance(other, BFVVector):
            result_ct = self.ciphertext * other.ciphertext
        else:
            other_list = other.tolist() if hasattr(other, 'tolist') else [float(other)]
            result_ct = self.ciphertext * other_list

        # Note: relinearization is not directly exposed in TenSEAL Python

        result = BFVVector.__new__(BFVVector)
        result.context = self.context
        result.ciphertext = result_ct
        result._original_len = self._original_len
        result._is_fallback = self._is_fallback
        return result

    def negate(self) -> 'BFVVector':
        """Homomorphic negation."""
        result_ct = -self.ciphertext

        result = BFVVector.__new__(BFVVector)
        result.context = self.context
        result.ciphertext = result_ct
        result._original_len = self._original_len
        result._is_fallback = self._is_fallback
        return result

    def decrypt(self, secret_key: SecretKey) -> np.ndarray:
        """Decrypt the vector."""
        decrypted = np.array(self.ciphertext.decrypt(secret_key))

        # Round to integers if we're doing "BFV" operations
        if self._is_fallback:
            # For fallback mode, return floats (actual CKKS behavior)
            return decrypted[:self._original_len]
        else:
            # True BFV would return integers
            return np.round(decrypted[:self._original_len]).astype(np.int64)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            result_ct = other - self.ciphertext
        else:
            result_ct = other - self.ciphertext

        result = BFVVector.__new__(BFVVector)
        result.context = self.context
        result.ciphertext = result_ct
        result._original_len = self._original_len
        result._is_fallback = self._is_fallback
        return result

    def __mul__(self, other):
        raise TypeError("Use .multiply(other, relin_key) for multiplication")

    def __rmul__(self, other):
        raise TypeError("Use .multiply(other, relin_key) for multiplication")

    def __neg__(self):
        return self.negate()

    def size_bytes(self) -> int:
        """Get ciphertext size in bytes."""
        return len(self.ciphertext.serialize())


def encrypted_and(
    a: BFVVector,
    b: BFVVector,
) -> BFVVector:
    """
    Homomorphic AND operation for binary data.

    In BFV, AND can be implemented as multiplication: a & b = a * b

    Args:
        a: Binary ciphertext (0s and 1s)
        b: Binary ciphertext (0s and 1s)

    Returns:
        Ciphertext with a & b

    Example:
        >>> a = BFVVector.from_plaintext(np.array([1, 0, 1]), ctx)
        >>> b = BFVVector.from_plaintext(np.array([1, 1, 0]), ctx)
        >>> result = encrypted_and(a, b)
        >>> # Decrypt to get [1, 0, 0]
    """
    # AND is just multiplication for binary values
    raise NotImplementedError("Use .multiply() for AND operation")


def encrypted_or(
    a: BFVVector,
    b: BFVVector,
) -> BFVVector:
    """
    Homomorphic OR operation for binary data.

    OR can be implemented as: a | b = a + b - a*b

    Args:
        a: Binary ciphertext (0s and 1s)
        b: Binary ciphertext (0s and 1s)

    Returns:
        Ciphertext with a | b
    """
    raise NotImplementedError("Implement using add and multiply")


def encrypted_xor(
    a: BFVVector,
    b: BFVVector,
) -> BFVVector:
    """
    Homomorphic XOR operation for binary data.

    XOR can be implemented as: a ^ b = a + b - 2*a*b

    Args:
        a: Binary ciphertext (0s and 1s)
        b: Binary ciphertext (0s and 1s)

    Returns:
        Ciphertext with a ^ b
    """
    raise NotImplementedError("Implement using add and multiply")


def encrypted_not(
    a: BFVVector,
) -> BFVVector:
    """
    Homomorphic NOT operation for binary data.

    NOT can be implemented as: ~a = 1 - a

    Args:
        a: Binary ciphertext (0s and 1s)

    Returns:
        Ciphertext with ~a
    """
    # NOT = 1 - a
    raise NotImplementedError("Use subtract from 1")
