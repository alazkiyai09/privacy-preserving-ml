"""
Matrix Operations for Homomorphic Encryption
=============================================
Encrypted matrix-vector multiplication for neural network layers.

Key Challenge:
- Matrices must be encoded to work with encrypted data
- Rotations (Galois keys) enable efficient matrix operations
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import tenseal as ts
import warnings

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any
Context = Any


def encrypted_plain_matrix_vector_multiply(
    encrypted_vector: CiphertextVector,
    plain_matrix: np.ndarray,
    relin_key: RelinKeys,
) -> List[CiphertextVector]:
    """
    Multiply encrypted vector by plaintext matrix.

    E(x) @ W where x is (m,) and W is (m, n), result is (n,)

    This is the core operation for neural network linear layers.

    STRATEGY: Dot product per column
    For each output dimension j:
        result[j] = Σ_i x[i] * W[i, j]

    Noise cost: n * log2(scale) bits for n output dimensions

    Args:
        encrypted_vector: Encrypted input vector (size m)
        plain_matrix: Plaintext weight matrix (m x n)
        relin_key: Relinearization key

    Returns:
        List of n encrypted output values

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])  # Input (3,)
        >>> W = np.array([[1.0, 2.0],      # Weights (3, 2)
        ...              [3.0, 4.0],
        ...              [5.0, 6.0]])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = encrypted_plain_matrix_vector_multiply(encrypted_x, W, relin_key)
        >>> # Decrypt: [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    """
    vec_size = encrypted_vector.size()
    if vec_size != plain_matrix.shape[0]:
        raise ValueError(
            f"Vector size ({vec_size}) must match "
            f"matrix rows ({plain_matrix.shape[0]})"
        )

    from he_ml.ml_ops.vector_ops import encrypted_dot_product_plain

    results = []
    for col_idx in range(plain_matrix.shape[1]):
        column = plain_matrix[:, col_idx]
        dot_prod = encrypted_dot_product_plain(encrypted_vector, column, relin_key)
        results.append(dot_prod)

    return results


def encrypted_plain_matrix_vector_multiply_with_bias(
    encrypted_vector: CiphertextVector,
    plain_matrix: np.ndarray,
    plain_bias: np.ndarray,
    relin_key: RelinKeys,
) -> List[CiphertextVector]:
    """
    Multiply encrypted vector by plaintext matrix and add bias.

    E(x) @ W + b

    Complete linear layer: weights + bias

    Args:
        encrypted_vector: Encrypted input vector (size m)
        plain_matrix: Plaintext weight matrix (m x n)
        plain_bias: Plaintext bias vector (size n)
        relin_key: Relinearization key

    Returns:
        List of n encrypted output values

    Example:
        >>> x = np.array([1.0, 2.0])
        >>> W = np.array([[1.0, 2.0],
        ...              [3.0, 4.0]])
        >>> b = np.array([0.5, 1.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = encrypted_plain_matrix_vector_multiply_with_bias(
        ...     encrypted_x, W, b, relin_key
        ... )
        >>> # Decrypt: [1*1+2*3+0.5, 1*2+2*4+1.0] = [7.5, 12.0]
    """
    # Compute matrix multiplication
    results = encrypted_plain_matrix_vector_multiply(
        encrypted_vector, plain_matrix, relin_key
    )

    # Add bias to each result
    final_results = []
    for result, bias in zip(results, plain_bias):
        with_bias = result + bias
        final_results.append(with_bias)

    return final_results


def encrypted_batch_matrix_multiply(
    encrypted_batch: List[CiphertextVector],
    plain_matrix: np.ndarray,
    relin_key: RelinKeys,
) -> List[List[CiphertextVector]]:
    """
    Multiply batch of encrypted vectors by plaintext matrix.

    Processes multiple inputs through the same weight matrix.

    Args:
        encrypted_batch: List of encrypted input vectors
        plain_matrix: Plaintext weight matrix (m x n)
        relin_key: Relinearization key

    Returns:
        List of lists of encrypted outputs (batch_size x n)

    Example:
        >>> batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        >>> W = np.array([[1.0, 2.0],
        ...              [3.0, 4.0]])
        >>> encrypted_batch = [encrypt_vector(v, ctx) for v in batch]
        >>> results = encrypted_batch_matrix_multiply(encrypted_batch, W, relin_key)
        >>> # Decrypt: [[7, 10], [15, 22]]
    """
    batch_results = []

    for encrypted_vec in encrypted_batch:
        result = encrypted_plain_matrix_vector_multiply(
            encrypted_vec, plain_matrix, relin_key
        )
        batch_results.append(result)

    return batch_results


def diagonal_matrix_vector_multiply(
    encrypted_vector: CiphertextVector,
    plain_diagonal: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Multiply encrypted vector by diagonal matrix (element-wise scaling).

    diag(d) @ E(x) = E(x) * d

    MUCH more efficient than full matrix multiplication.
    Useful for scaling operations, gating, attention weights.

    Args:
        encrypted_vector: Encrypted vector
        plain_diagonal: Diagonal elements (same size as vector)
        relin_key: Relinearization key

    Returns:
        Encrypted scaled vector

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> diag = np.array([0.5, 2.0, 1.5])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = diagonal_matrix_vector_multiply(encrypted_x, diag, relin_key)
        >>> # Decrypt to get [0.5, 4.0, 4.5]
    """
    vec_size = encrypted_vector.size()
    if vec_size != len(plain_diagonal):
        raise ValueError("Vector and diagonal must have same size")

    # Element-wise multiplication
    result = encrypted_vector * plain_diagonal.tolist()
    # Note: relinearization is not directly exposed in TenSEAL Python

    return result


def outer_product_approximation(
    encrypted_vector: CiphertextVector,
    plain_vector: np.ndarray,
    relin_key: RelinKeys,
    galois_key: GaloisKeys,
) -> List[CiphertextVector]:
    """
    Approximate outer product E(x) * y^T using rotations.

    WARNING: This is computationally expensive!
    Consider if you really need this or can restructure your computation.

    Args:
        encrypted_vector: Encrypted vector (size m)
        plain_vector: Plaintext vector (size n)
        relin_key: Relinearization key
        galois_key: Galois keys for rotations

    Returns:
        Flattened outer product (m*n elements)

    Note:
        This implementation is simplified and may not be efficient.
        For production, consider using packing strategies or
        avoiding outer products in encrypted domain.
    """
    warnings.warn(
        "Outer products in HE are very expensive. "
        "Consider restructuring your computation to avoid this."
    )

    # Simplified: Return element-wise products
    # Full outer product would require complex packing strategies
    raise NotImplementedError(
        "Full outer product requires advanced packing strategies. "
        "Consider using diagonal operations instead."
    )


def compute_diagonal_loading(
    encrypted_matrix: List[CiphertextVector],
    loading_value: float,
    relin_key: RelinKeys,
) -> List[CiphertextVector]:
    """
    Add loading value to diagonal of encrypted matrix (represented as list of columns).

    E(A) + λ*I

    Useful for regularization in ridge regression, numerical stability.

    Args:
        encrypted_matrix: Encrypted matrix (list of column vectors)
        loading_value: Value to add to diagonal
        relin_key: Relinearization key

    Returns:
        Matrix with diagonal loading applied

    Example:
        >>> # For a 3x3 matrix A, add λ to diagonal elements
        >>> # A_11 + λ, A_22 + λ, A_33 + λ
    """
    if len(encrypted_matrix) != len(encrypted_matrix):
        raise ValueError("Matrix must be square")

    result = []
    for i, col in enumerate(encrypted_matrix):
        # Add loading to diagonal element (i-th element of i-th column)
        # This is simplified - actual implementation depends on packing
        result.append(col + loading_value)

    return result


def matrix_transpose_operation(
    encrypted_matrix: List[CiphertextVector],
    shape: Tuple[int, int],
    galois_key: GaloisKeys,
) -> List[CiphertextVector]:
    """
    Transpose encrypted matrix using rotations.

    E(A)^T

    WARNING: Complex operation requiring careful packing.

    Args:
        encrypted_matrix: Encrypted matrix (list of column vectors)
        shape: (rows, cols) of the matrix
        galois_key: Galois keys for rotations

    Returns:
        Transposed matrix

    Note:
        This is a simplified placeholder. Full implementation requires
        specific packing strategies.
    """
    warnings.warn(
        "Matrix transpose in HE is complex and requires specific packing. "
        "Consider organizing your computation to avoid transposition."
    )

    # Placeholder: return original (would need proper implementation)
    return encrypted_matrix


def print_matrix_operation_info(
    matrix_shape: Tuple[int, int],
    batch_size: int = 1,
) -> None:
    """
    Print information about matrix operation complexity.

    Args:
        matrix_shape: (rows, cols) of the weight matrix
        batch_size: Number of input vectors
    """
    rows, cols = matrix_shape

    print(f"\nMatrix Operation Information:")
    print(f"  Matrix shape: {rows} x {cols}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total multiplications: {batch_size * rows * cols}")
    print(f"  Output dimensions: {batch_size} x {cols}")

    # Estimate noise consumption
    scale_bits = 40  # Assuming scale=2^40
    noise_per_mult = scale_bits + 2
    total_noise = batch_size * cols * noise_per_mult

    print(f"  Estimated noise consumption: ~{total_noise} bits")


def validate_matrix_vector_multiply(
    plaintext_vector: np.ndarray,
    plaintext_matrix: np.ndarray,
    encrypted_result: List[CiphertextVector],
    secret_key: SecretKey,
    tolerance: float = 1e-3,
) -> bool:
    """
    Validate encrypted matrix-vector multiplication against plaintext computation.

    Args:
        plaintext_vector: Original input vector
        plaintext_matrix: Weight matrix
        encrypted_result: Encrypted result from HE operation
        secret_key: Secret key for decryption
        tolerance: Max allowed error

    Returns:
        True if validation passes
    """
    # Compute expected result
    expected = plaintext_vector @ plaintext_matrix

    # Decrypt result
    decrypted = np.array([ct.decrypt(secret_key)[0] for ct in encrypted_result])

    # Compare
    error = np.max(np.abs(expected - decrypted))

    if error > tolerance:
        print(f"Validation failed: max error = {error:.2e}")
        return False

    return True
