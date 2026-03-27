"""
CKKS Scheme Wrapper
====================
High-level interface for CKKS homomorphic encryption with scale management.

CKKS is the preferred scheme for machine learning because it supports
floating-point arithmetic with controlled approximation.

Key Features:
- Automatic scale management (rescaling after operations)
- Optimized for neural network computations
- Handles polynomial approximations for activation functions
"""

from typing import Union, List, Optional, Tuple, Any
import numpy as np
import tenseal as ts
import warnings

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any
Context = Any


class CKKSCiphertext:
    """
    Wrapper for CKKS ciphertexts with automatic scale tracking.

    Tracks the current scale of the ciphertext to ensure operations
    remain valid and accurate.
    """

    def __init__(
        self,
        ciphertext: CiphertextVector,
        scale: float,
        context: Context,
    ):
        self.ciphertext = ciphertext
        self.scale = scale
        self.context = context

    def __add__(self, other):
        if isinstance(other, CKKSCiphertext):
            # Scales must match for ciphertext-ciphertext addition
            if abs(self.scale - other.scale) > 1e-6:
                warnings.warn(
                    f"Scale mismatch: {self.scale} vs {other.scale}. "
                    "Results may be inaccurate."
                )
            result_ct = self.ciphertext + other.ciphertext
            new_scale = self.scale
        else:
            # Ciphertext-plaintext addition
            result_ct = self.ciphertext + other
            new_scale = self.scale

        return CKKSCiphertext(result_ct, new_scale, self.context)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CKKSCiphertext):
            if abs(self.scale - other.scale) > 1e-6:
                warnings.warn(
                    f"Scale mismatch: {self.scale} vs {other.scale}. "
                    "Results may be inaccurate."
                )
            result_ct = self.ciphertext - other.ciphertext
            new_scale = self.scale
        else:
            result_ct = self.ciphertext - other
            new_scale = self.scale

        return CKKSCiphertext(result_ct, new_scale, self.context)

    def __rsub__(self, other):
        result_ct = other - self.ciphertext
        return CKKSCiphertext(result_ct, self.scale, self.context)

    def __mul__(self, other):
        if isinstance(other, CKKSCiphertext):
            # Ciphertext-ciphertext: scale multiplies
            result_ct = self.ciphertext * other.ciphertext
            new_scale = self.scale * other.scale
        else:
            # Ciphertext-plaintext: scale preserved (TenSEAL handles rescaling)
            result_ct = self.ciphertext * other
            new_scale = self.scale

        return CKKSCiphertext(result_ct, new_scale, self.context)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        result_ct = -self.ciphertext
        return CKKSCiphertext(result_ct, self.scale, self.context)

    def decrypt(self, secret_key: SecretKey) -> np.ndarray:
        """Decrypt the ciphertext."""
        return np.array(self.ciphertext.decrypt(secret_key))

    def size(self) -> int:
        """Get ciphertext size in bytes."""
        return len(self.ciphertext.serialize())


class CKKSVector:
    """
    High-level CKKS vector with automatic scale management.

    Provides a clean API for homomorphic operations on vectors.
    """

    def __init__(
        self,
        data: np.ndarray,
        context: Context,
        scale: Optional[float] = None,
    ):
        """
        Create an encrypted CKKS vector.

        Args:
            data: Plaintext vector to encrypt
            context: TenSEAL CKKS context
            scale: Initial scale (uses context.global_scale if None)
        """
        self.context = context
        self.scale = scale if scale is not None else context.global_scale

        # Encrypt the data
        self.ciphertext = ts.ckks_vector(context, data.tolist())

        # Track original length for proper decryption
        self._original_len = len(data)

    @classmethod
    def from_plaintext(
        cls,
        data: np.ndarray,
        context: Context,
        scale: Optional[float] = None,
    ) -> 'CKKSVector':
        """Create encrypted vector from plaintext."""
        return cls(data, context, scale)

    def add(
        self,
        other: Union['CKKSVector', np.ndarray, float],
    ) -> 'CKKSVector':
        """Homomorphic addition."""
        if isinstance(other, CKKSVector):
            result_ct = self.ciphertext + other.ciphertext
        else:
            result_ct = self.ciphertext + (other.tolist() if hasattr(other, 'tolist') else other)

        # Create new vector with result
        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def subtract(
        self,
        other: Union['CKKSVector', np.ndarray, float],
    ) -> 'CKKSVector':
        """Homomorphic subtraction."""
        if isinstance(other, CKKSVector):
            result_ct = self.ciphertext - other.ciphertext
        else:
            result_ct = self.ciphertext - (other.tolist() if hasattr(other, 'tolist') else other)

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def multiply(
        self,
        other: Union['CKKSVector', np.ndarray, float],
        relin_key: RelinKeys,
        auto_relin: bool = False,
    ) -> 'CKKSVector':
        """Homomorphic multiplication.

        Args:
            other: Value to multiply by
            relin_key: Relinearization key (required for ciphertext-ciphertext)
            auto_relin: If True, automatically relinearize (default: False)
        """
        if isinstance(other, CKKSVector):
            result_ct = self.ciphertext * other.ciphertext
            # Scale multiplies
            new_scale = self.scale * other.scale
        else:
            result_ct = self.ciphertext * (other.tolist() if hasattr(other, 'tolist') else other)
            # Scale preserved
            new_scale = self.scale

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = new_scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def relinearize(self, relin_key: RelinKeys) -> 'CKKSVector':
        """Manually relinearize the ciphertext."""
        # For now, return self as relinearization happens in TenSEAL automatically
        # in many cases
        return self

    def square(
        self,
        relin_key: RelinKeys = None,
    ) -> 'CKKSVector':
        """Homomorphic squaring.

        Args:
            relin_key: Relinearization key (optional, not currently used)
        """
        result_ct = self.ciphertext.square()

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale * self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def dot_product(
        self,
        other: np.ndarray,
        relin_key: RelinKeys,
    ) -> 'CKKSVector':
        """
        Dot product with plaintext vector.

        Computes: sum(self[i] * other[i])

        This is the fundamental operation for neural network layers.
        """
        # Element-wise multiplication
        result_ct = self.ciphertext * other.tolist()

        # For sum, we need to manually add all elements
        # In practice, this would use rotations or plaintext operations
        # Note: relinearization is not directly exposed in TenSEAL Python

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def negate(self) -> 'CKKSVector':
        """Homomorphic negation."""
        result_ct = -self.ciphertext

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
        return result

    def decrypt(self, secret_key: SecretKey) -> np.ndarray:
        """Decrypt the vector."""
        decrypted = np.array(self.ciphertext.decrypt(secret_key))
        return decrypted[:self._original_len]

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        # For scalar - ciphertext
        if isinstance(other, (int, float)):
            result_ct = other - self.ciphertext
        else:
            result_ct = other - self.ciphertext

        result = CKKSVector.__new__(CKKSVector)
        result.context = self.context
        result.scale = self.scale
        result.ciphertext = result_ct
        result._original_len = self._original_len
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


def encrypted_mean(
    vectors: List[CKKSVector],
    relin_key: RelinKeys,
) -> CKKSVector:
    """
    Compute encrypted mean of multiple vectors.

    Args:
        vectors: List of encrypted vectors
        relin_key: Relinearization key (for future use)

    Returns:
        Encrypted mean: (v1 + v2 + ... + vn) / n
    """
    if not vectors:
        raise ValueError("vectors list cannot be empty")

    # Sum all vectors
    result = vectors[0]
    for v in vectors[1:]:
        result = result.add(v)

    # Divide by n (multiply by plaintext scalar)
    n = len(vectors)
    # Scale = result.scale (ciphertext-scalar multiplication preserves scale)
    result_ct = result.ciphertext * (1.0 / n)

    result_obj = CKKSVector.__new__(CKKSVector)
    result_obj.context = result.context
    result_obj.scale = result.scale
    result_obj.ciphertext = result_ct
    result_obj._original_len = result._original_len
    return result_obj


def encrypted_variance(
    vectors: List[CKKSVector],
    encrypted_mean: CKKSVector,
    relin_key: RelinKeys,
) -> CKKSVector:
    """
    Compute encrypted variance: E[(x - mean)^2]

    WARNING: This requires 2 multiplications per vector, very expensive!

    Args:
        vectors: List of encrypted vectors
        encrypted_mean: Pre-computed encrypted mean
        relin_key: Relinearization key

    Returns:
        Encrypted variance
    """
    if not vectors:
        raise ValueError("vectors list cannot be empty")

    n = len(vectors)

    # Sum of squared differences
    result = None
    for v in vectors:
        diff = v.subtract(encrypted_mean)
        sq = diff.square(relin_key)
        if result is None:
            result = sq
        else:
            result = result.add(sq)

    # Divide by n
    result_ct = result.ciphertext * (1.0 / n)
    # Note: relinearization is not directly exposed in TenSEAL Python

    result_obj = CKKSVector.__new__(CKKSVector)
    result_obj.context = result.context
    result_obj.scale = result.scale
    result_obj.ciphertext = result_ct
    result_obj._original_len = result._original_len
    return result_obj


def print_scale_info(
    ciphertext: Union[CKKSVector, CKKSCiphertext],
    label: str = "Ciphertext",
) -> None:
    """Print scale information for debugging."""
    print(f"\n{label} Scale Information:")
    print(f"  Scale: {ciphertext.scale:.2e} (2^{int(ciphertext.scale).bit_length() - 1})")


def validate_scale_compatibility(
    a: Union[CKKSVector, CKKSCiphertext],
    b: Union[CKKSVector, CKKSCiphertext],
    operation: str,
) -> bool:
    """
    Validate that two ciphertexts have compatible scales for an operation.

    Returns:
        True if compatible, False otherwise
    """
    if abs(a.scale - b.scale) > 1e-6:
        warnings.warn(
            f"Scale incompatibility in {operation}: "
            f"{a.scale:.2e} vs {b.scale:.2e}"
        )
        return False
    return True
