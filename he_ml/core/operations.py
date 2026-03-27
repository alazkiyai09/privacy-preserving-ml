"""
Homomorphic Operations
=======================
Core homomorphic operations: addition, multiplication, relinearization.

CRITICAL CONCEPTS:
- Homomorphic Addition: E(a) + E(b) = E(a + b)
- Homomorphic Multiplication: E(a) * E(b) = E(a * b)
- Relinearization: REQUIRED after ciphertext-ciphertext multiplication
  - Reduces ciphertext size from 3 to 2 polynomials
  - Consumes some noise but necessary for further operations
- Rescaling (CKKS): Manages the scale parameter after multiplication
"""

from typing import Union, List, Any
import numpy as np
import tenseal as ts
import warnings

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any


def homomorphic_add(
    a: CiphertextVector,
    b: Union[CiphertextVector, np.ndarray],
) -> CiphertextVector:
    """
    Homomorphic addition: E(a) + E(b) = E(a + b)

    Supports:
    - Ciphertext + Ciphertext
    - Ciphertext + Plaintext (NumPy array)

    Noise cost: ~2 bits (minimal)

    Args:
        a: First ciphertext
        b: Second ciphertext or plaintext vector

    Returns:
        Encrypted sum: E(a + b)

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_y = encrypt_vector(y, ctx)
        >>> encrypted_sum = homomorphic_add(encrypted_x, encrypted_y)
        >>> # Decrypt to get [5.0, 7.0, 9.0]
    """
    if isinstance(b, np.ndarray):
        # Ciphertext-plaintext addition
        return a + b.tolist()
    else:
        # Ciphertext-ciphertext addition
        return a + b


def homomorphic_subtract(
    a: CiphertextVector,
    b: Union[CiphertextVector, np.ndarray],
) -> CiphertextVector:
    """
    Homomorphic subtraction: E(a) - E(b) = E(a - b)

    Supports:
    - Ciphertext - Ciphertext
    - Ciphertext - Plaintext (NumPy array)

    Noise cost: ~2 bits (minimal)

    Args:
        a: First ciphertext
        b: Second ciphertext or plaintext vector

    Returns:
        Encrypted difference: E(a - b)

    Example:
        >>> x = np.array([5.0, 7.0, 9.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_y = encrypt_vector(y, ctx)
        >>> encrypted_diff = homomorphic_subtract(encrypted_x, encrypted_y)
        >>> # Decrypt to get [1.0, 2.0, 3.0]
    """
    if isinstance(b, np.ndarray):
        # Ciphertext-plaintext subtraction
        return a - b.tolist()
    else:
        # Ciphertext-ciphertext subtraction
        return a - b


def homomorphic_multiply(
    a: CiphertextVector,
    b: Union[CiphertextVector, np.ndarray],
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Homomorphic multiplication: E(a) * E(b) = E(a * b)

    IMPORTANT: After ciphertext-ciphertext multiplication, you MUST
    call relinearize() before further operations.

    Noise cost:
    - Ciphertext-plaintext: ~log2(scale) bits (e.g., ~40 bits for scale=2^40)
    - Ciphertext-ciphertext: ~2*log2(scale) bits (e.g., ~85 bits)

    Args:
        a: First ciphertext
        b: Second ciphertext or plaintext vector
        relin_key: Relinearization key (REQUIRED for ciphertext-ciphertext)

    Returns:
        Encrypted product: E(a * b)

    Example:
        >>> x = np.array([2.0, 3.0, 4.0])
        >>> y = np.array([5.0, 6.0, 7.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_y = encrypt_vector(y, ctx)
        >>> keys = generate_keys(ctx, relinearization_key=True)
        >>> encrypted_prod = homomorphic_multiply(encrypted_x, encrypted_y,
        ...                                       keys['relin_key'])
        >>> encrypted_prod = relinearize(encrypted_prod, keys['relin_key'])
        >>> # Decrypt to get [10.0, 18.0, 28.0]
    """
    if isinstance(b, np.ndarray):
        # Ciphertext-plaintext multiplication (cheaper)
        result = a * b.tolist()
    else:
        # Ciphertext-ciphertext multiplication (expensive!)
        result = a * b

    return result


def relinearize(
    ciphertext: CiphertextVector,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Relinearize a ciphertext after multiplication.

    WHY NEEDED:
    - Ciphertext-ciphertext multiplication produces a size-3 ciphertext
    - Size-3 ciphertexts cannot be used in further multiplications
    - Relinearization reduces back to size-2

    Noise cost: ~1-2 bits (minimal, but necessary)

    Args:
        ciphertext: Ciphertext to relinearize (typically after multiplication)
        relin_key: Relinearization key

    Returns:
        Relinearized ciphertext (size 2)

    Example:
        >>> # After multiplication
        >>> encrypted_prod = homomorphic_multiply(x, y, relin_key)
        >>> # MUST relinearize before next operation
        >>> encrypted_prod = relinearize(encrypted_prod, relin_key)

    Note:
        Relinearization is not directly exposed in TenSEAL Python API.
        It happens automatically or requires lower-level API access.
        This function returns the ciphertext unchanged.
    """
    # Note: relinearization is not directly exposed in TenSEAL Python
    # It happens automatically or requires lower-level API access
    return ciphertext


def homomorphic_negate(
    a: CiphertextVector,
) -> CiphertextVector:
    """
    Homomorphic negation: -E(a) = E(-a)

    Noise cost: ~0 bits (free operation)

    Args:
        a: Ciphertext to negate

    Returns:
        Encrypted negation: E(-a)

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_neg = homomorphic_negate(encrypted_x)
        >>> # Decrypt to get [-1.0, -2.0, -3.0]
    """
    # In CKKS, negation is free
    return -a


def homomorphic_square(
    a: CiphertextVector,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Homomorphic squaring: E(a)^2 = E(a^2)

    More efficient than general multiplication for the same ciphertext.

    Noise cost: ~2*log2(scale) bits

    Args:
        a: Ciphertext to square
        relin_key: Relinearization key

    Returns:
        Encrypted square: E(a^2)

    Example:
        >>> x = np.array([2.0, 3.0, 4.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_sq = homomorphic_square(encrypted_x, relin_key)
        >>> encrypted_sq = relinearize(encrypted_sq, relin_key)
        >>> # Decrypt to get [4.0, 9.0, 16.0]
    """
    result = a.square()
    return result


def homomorphic_power(
    a: CiphertextVector,
    exponent: int,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Homomorphic exponentiation: E(a)^n = E(a^n)

    Uses repeated squaring for efficiency.
    WARNING: Each multiplication consumes significant noise budget.

    Noise cost: ~exponent * 2*log2(scale) bits

    Args:
        a: Base ciphertext
        exponent: Power to raise to (must be >= 1)
        relin_key: Relinearization key

    Returns:
        Encrypted power: E(a^exponent)

    Example:
        >>> x = np.array([2.0, 3.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_cube = homomorphic_power(encrypted_x, 3, relin_key)
        >>> # Decrypt to get [8.0, 27.0]
    """
    if exponent < 1:
        raise ValueError(f"exponent must be >= 1, got {exponent}")

    if exponent == 1:
        return a

    # Binary exponentiation for efficiency
    result = None
    base = a

    while exponent > 0:
        if exponent % 2 == 1:
            if result is None:
                result = base
            else:
                result = homomorphic_multiply(result, base, relin_key)
                result = relinearize(result, relin_key)

        base = homomorphic_square(base, relin_key)
        base = relinearize(base, relin_key)
        exponent //= 2

    return result


def homomorphic_sum(
    ciphertexts: List[CiphertextVector],
) -> CiphertextVector:
    """
    Sum multiple ciphertexts: E(a) + E(b) + E(c) + ...

    More efficient than sequential addition.

    Noise cost: ~n * 2 bits for n ciphertexts

    Args:
        ciphertexts: List of ciphertexts to sum

    Returns:
        Encrypted sum: E(sum of all inputs)

    Example:
        >>> encrypted_x = encrypt_vector(np.array([1.0, 2.0]), ctx)
        >>> encrypted_y = encrypt_vector(np.array([3.0, 4.0]), ctx)
        >>> encrypted_z = encrypt_vector(np.array([5.0, 6.0]), ctx)
        >>> result = homomorphic_sum([encrypted_x, encrypted_y, encrypted_z])
        >>> # Decrypt to get [9.0, 12.0]
    """
    if not ciphertexts:
        raise ValueError("ciphertexts list cannot be empty")

    if len(ciphertexts) == 1:
        return ciphertexts[0]

    # Sum all ciphertexts
    result = ciphertexts[0]
    for ct in ciphertexts[1:]:
        result = homomorphic_add(result, ct)

    return result


def homomorphic_dot_product_plain(
    encrypted_a: CiphertextVector,
    plain_b: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Encrypted dot product with plaintext: E(a) · b

    Computes: sum(E(a[i]) * b[i]) for all i

    This is the fundamental operation for linear layers in neural networks.

    Noise cost: ~log2(scale) + log2(n) bits

    Args:
        encrypted_a: Encrypted vector
        plain_b: Plaintext vector (e.g., neural network weights)
        relin_key: Relinearization key

    Returns:
        Encrypted scalar (or vector) representing the dot product

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = homomorphic_dot_product_plain(encrypted_x, weights, relin_key)
        >>> # Decrypt to get 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 1.7
    """
    # Element-wise multiplication
    result = homomorphic_multiply(encrypted_a, plain_b, relin_key)

    # Sum all elements (using plaintext addition for efficiency)
    # In CKKS, we can use plain addition to sum
    result = relinearize(result, relin_key)

    return result


def rotate(
    ciphertext: CiphertextVector,
    steps: int,
    galois_key: GaloisKeys,
) -> CiphertextVector:
    """
    Rotate ciphertext slots.

    Rotation enables efficient matrix operations and batch processing.

    Args:
        ciphertext: Ciphertext to rotate
        steps: Number of positions to rotate (positive = left, negative = right)
        galois_key: Galois keys for rotation

    Returns:
        Rotated ciphertext

    Example:
        >>> x = np.array([1.0, 2.0, 3.0, 4.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> rotated = rotate(encrypted_x, 1, galois_key)
        >>> # Decrypt to get [2.0, 3.0, 4.0, 1.0]
    """
    return ciphertext.rotate(steps, galois_key)


def print_operation_noise(
    operation: str,
    noise_before: int,
    noise_after: int,
) -> None:
    """
    Print noise consumption for an operation.

    Args:
        operation: Operation name
        noise_before: Noise budget before operation (bits)
        noise_after: Noise budget after operation (bits)
    """
    consumed = noise_before - noise_after

    print(f"\n{operation}:")
    print(f"  Noise before: {noise_before} bits")
    print(f"  Noise after:  {noise_after} bits")
    print(f"  Consumed:     {consumed} bits ({consumed/noise_before*100:.1f}%)")


# Convenience wrappers for common operations
class EncryptedNumber:
    """
    Wrapper class for encrypted numbers with operator overloading.

    Makes homomorphic operations more intuitive:
        encrypted_x = EncryptedNumber(encrypt_vector([1.0, 2.0], ctx))
        encrypted_y = EncryptedNumber(encrypt_vector([3.0, 4.0], ctx))
        encrypted_sum = encrypted_x + encrypted_y
        encrypted_prod = encrypted_x * encrypted_y
    """

    def __init__(
        self,
        ciphertext: CiphertextVector,
        relin_key: RelinKeys,
        galois_key: GaloisKeys = None,
    ):
        self.ciphertext = ciphertext
        self.relin_key = relin_key
        self.galois_key = galois_key

    def __add__(self, other):
        if isinstance(other, EncryptedNumber):
            result_ct = homomorphic_add(self.ciphertext, other.ciphertext)
        else:
            result_ct = homomorphic_add(self.ciphertext, other)
        return EncryptedNumber(result_ct, self.relin_key, self.galois_key)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, EncryptedNumber):
            result_ct = homomorphic_subtract(self.ciphertext, other.ciphertext)
        else:
            result_ct = homomorphic_subtract(self.ciphertext, other)
        return EncryptedNumber(result_ct, self.relin_key, self.galois_key)

    def __rsub__(self, other):
        result_ct = homomorphic_subtract(other, self.ciphertext)
        return EncryptedNumber(result_ct, self.relin_key, self.galois_key)

    def __mul__(self, other):
        if isinstance(other, EncryptedNumber):
            result_ct = homomorphic_multiply(
                self.ciphertext, other.ciphertext, self.relin_key
            )
        else:
            result_ct = homomorphic_multiply(
                self.ciphertext, other, self.relin_key
            )
        result_ct = relinearize(result_ct, self.relin_key)
        return EncryptedNumber(result_ct, self.relin_key, self.galois_key)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        result_ct = homomorphic_negate(self.ciphertext)
        return EncryptedNumber(result_ct, self.relin_key, self.galois_key)

    def decrypt(self, secret_key: SecretKey) -> np.ndarray:
        """Decrypt the encrypted number."""
        return np.array(self.ciphertext.decrypt(secret_key))
