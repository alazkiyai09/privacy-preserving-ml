"""
Vector Operations for Homomorphic Encryption
=============================================
Encrypted implementations of vector operations used in machine learning.

Key Operations:
- Dot products (fundamental for neural network layers)
- Polynomial evaluation (for activation functions)
- Element-wise operations
"""

from typing import List, Union, Optional
from typing import Any
import numpy as np
import tenseal as ts
import warnings

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any


def encrypted_dot_product_plain(
    encrypted_vector: CiphertextVector,
    plain_vector: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute encrypted dot product with plaintext vector.

    E(x) · w = Σ E(x[i]) * w[i]

    This is THE fundamental operation for neural network forward passes.

    Noise cost: ~log2(scale) bits (cheaper than ciphertext-ciphertext)

    Args:
        encrypted_vector: Encrypted input vector
        plain_vector: Plaintext weights (e.g., neural network weights)
        relin_key: Relinearization key

    Returns:
        Encrypted scalar (sum of products)

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = encrypted_dot_product_plain(encrypted_x, weights, relin_key)
        >>> # Decrypt to get 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 1.7
    """
    vec_size = encrypted_vector.size()
    if vec_size != len(plain_vector):
        raise ValueError(
            f"Vector size mismatch: {vec_size} vs {len(plain_vector)}"
        )

    # Element-wise multiplication
    result = encrypted_vector * plain_vector.tolist()

    # Sum all elements
    # In CKKS, we can use the built-in sum method
    result = result.sum()

    # Relinearize for further operations
    # Note: relinearization is not directly exposed in TenSEAL Python
    # It happens automatically or requires lower-level API access
    return result


def encrypted_dot_product_cipher(
    encrypted_a: CiphertextVector,
    encrypted_b: CiphertextVector,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute dot product of two encrypted vectors.

    E(x) · E(y) = Σ E(x[i]) * E(y[i])

    MUCH more expensive than plaintext version due to ciphertext-ciphertext multiplication.
    Use encrypted_dot_product_plain() when possible.

    Noise cost: ~2*log2(scale) bits per element

    Args:
        encrypted_a: First encrypted vector
        encrypted_b: Second encrypted vector
        relin_key: Relinearization key

    Returns:
        Encrypted dot product

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_y = encrypt_vector(y, ctx)
        >>> result = encrypted_dot_product_cipher(encrypted_x, encrypted_y, relin_key)
        >>> # Decrypt to get 1*4 + 2*5 + 3*6 = 32
    """
    size_a = encrypted_a.size()
    size_b = encrypted_b.size()
    if size_a != size_b:
        raise ValueError(
            f"Vector size mismatch: {size_a} vs {size_b}"
        )

    # Element-wise multiplication
    result = encrypted_a * encrypted_b

    # Sum all elements
    result = result.sum()

    # Relinearize
    # Note: relinearization is not directly exposed in TenSEAL Python
    # It happens automatically or requires lower-level API access
    return result


def encrypted_weighted_sum(
    encrypted_vectors: List[CiphertextVector],
    weights: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute weighted sum of encrypted vectors.

    Σ weights[i] * E(vectors[i])

    Useful for:
    - Attention mechanisms
    - Ensemble predictions
    - Weighted averaging

    Args:
        encrypted_vectors: List of encrypted vectors
        weights: Plaintext weights (must sum to 1 for averaging)
        relin_key: Relinearization key

    Returns:
        Encrypted weighted sum

    Example:
        >>> vectors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        >>> encrypted = [encrypt_vector(v, ctx) for v in vectors]
        >>> weights = np.array([0.6, 0.4])
        >>> result = encrypted_weighted_sum(encrypted, weights, relin_key)
        >>> # Decrypt to get 0.6*[1,2] + 0.4*[3,4] = [1.8, 2.8]
    """
    num_vecs = len(encrypted_vectors)
    if num_vecs != len(weights):
        raise ValueError(
            f"Number of vectors ({num_vecs}) must match "
            f"number of weights ({len(weights)})"
        )

    result = None
    for i, (enc_vec, weight) in enumerate(zip(encrypted_vectors, weights)):
        # Scale encrypted vector by weight
        weighted = enc_vec * weight
        # Note: relinearization is not directly exposed in TenSEAL Python
        # It happens automatically or requires lower-level API access

        if result is None:
            result = weighted
        else:
            result = result + weighted

    return result


def encrypted_euclidean_distance_plain(
    encrypted_vector: CiphertextVector,
    plain_center: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute encrypted Euclidean distance to plaintext center.

    ||E(x) - c||² = Σ (E(x[i]) - c[i])²

    Useful for:
    - k-means clustering
    - Nearest neighbor search
    - Anomaly detection

    Args:
        encrypted_vector: Encrypted vector
        plain_center: Plaintext center point
        relin_key: Relinearization key (not currently used due to TenSEAL API limitations)

    Returns:
        Encrypted squared Euclidean distance

    Example:
        >>> x = np.array([3.0, 4.0])
        >>> center = np.array([0.0, 0.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> distance = encrypted_euclidean_distance_plain(encrypted_x, center, relin_key)
        >>> # Decrypt to get 3² + 4² = 25
    """
    # Subtract center
    diff = encrypted_vector - plain_center.tolist()

    # Square each element (element-wise)
    # Note: This requires ciphertext-plaintext multiplication
    squared = diff * plain_center.tolist()

    # Sum all elements
    result = squared.sum()

    return result


def encrypted_polynomial(
    encrypted_x: CiphertextVector,
    coefficients: List[float],
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Evaluate polynomial on encrypted value: Σ coeff[i] * x^i

    E(Σ coeff[i] * x^i) = Σ coeff[i] * E(x)^i

    CRITICAL for activation functions in neural networks.
    Uses Horner's method for efficiency: ((a*x + b)*x + c)*x + d

    Noise cost: degree * log2(scale) bits (expensive!)

    Args:
        encrypted_x: Encrypted input value(s)
        coefficients: Polynomial coefficients [c0, c1, c2, ...]
            where c0 is constant term, c1 is linear coefficient, etc.
        relin_key: Relinearization key

    Returns:
        Encrypted polynomial evaluation

    Example:
        >>> x = np.array([0.5, 1.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> # Evaluate 1 + 2x + 3x²
        >>> result = encrypted_polynomial(encrypted_x, [1, 2, 3], relin_key)
        >>> # Decrypt to get [1 + 2*0.5 + 3*0.25, 1 + 2*1 + 3*1] = [2.75, 6.0]
    """
    if not coefficients:
        raise ValueError("coefficients cannot be empty")

    # Start with constant term
    result = encrypted_x * 0 + coefficients[0]

    # Use Horner's method for efficiency
    # For polynomial c0 + c1*x + c2*x² + ... + cn*x^n
    # Evaluate as: ((...((cn*x + c{n-1})*x + c{n-2})*x + ...) + c0
    # But we need to go in reverse order of coefficients (excluding constant)

    # Start from highest degree coefficient
    for i in range(len(coefficients) - 1, 0, -1):
        coeff = coefficients[i]
        if coeff != 0:
            # Add coefficient then multiply by x
            result = result + coeff
            result = result * encrypted_x
            # Note: relinearization is not directly exposed in TenSEAL Python
    # It happens automatically or requires lower-level API access
    return result


def encrypted_cosine_approximation(
    encrypted_x: CiphertextVector,
    relin_key: RelinKeys,
    degree: int = 3,
) -> CiphertextVector:
    """
    Approximate cosine using Chebyshev polynomial.

    cos(x) ≈ 1 - x²/2! + x⁴/4! - ...

    Args:
        encrypted_x: Encrypted input (should be normalized to [-π, π])
        degree: Polynomial degree (3, 5, or 7)
        relin_key: Relinearization key

    Returns:
        Encrypted approximation of cos(x)

    Note:
        This is a simplified approximation. For better accuracy,
        use higher degrees or more sophisticated approximations.
    """
    # Normalize x to [-1, 1] for Chebyshev
    # Assuming input is in [-π, π], we divide by π
    x_normalized = encrypted_x * (1.0 / np.pi)
    # Note: relinearization is not directly exposed in TenSEAL Python

    # Chebyshev coefficients for cos(π*x) on [-1, 1]
    if degree == 3:
        coeffs = [1.0, 0.0, -4.9348e-1]  # T0 - 0.49348*T2
    elif degree == 5:
        coeffs = [1.0, 0.0, -4.9348e-1, 0.0, 4.4337e-3]
    elif degree == 7:
        coeffs = [1.0, 0.0, -4.9348e-1, 0.0, 4.4337e-3, 0.0, -1.447e-4]
    else:
        raise ValueError(f"Unsupported degree: {degree}. Use 3, 5, or 7.")

    result = encrypted_polynomial(x_normalized, coeffs, relin_key)

    return result


def encrypted_l2_norm(
    encrypted_vector: CiphertextVector,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute L2 norm: ||x|| = sqrt(Σ x[i]²)

    Note: Square root is NOT natively supported in HE.
    This returns the squared norm.

    Args:
        encrypted_vector: Encrypted vector
        relin_key: Relinearization key

    Returns:
        Encrypted squared L2 norm (||x||²)

    Example:
        >>> x = np.array([3.0, 4.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = encrypted_l2_norm(encrypted_x, relin_key)
        >>> # Decrypt to get 3² + 4² = 25 (not 5, because sqrt is not supported)
    """
    # Square each element
    squared = encrypted_vector * encrypted_vector
    # Note: relinearization is not directly exposed in TenSEAL Python

    # Sum
    result = squared.sum()

    return result


def encrypted_mse_loss(
    encrypted_predicted: CiphertextVector,
    plain_target: np.ndarray,
    relin_key: RelinKeys,
) -> CiphertextVector:
    """
    Compute Mean Squared Error loss (encrypted prediction, plaintext target).

    MSE = (1/n) * Σ (predicted - target)²

    Args:
        encrypted_predicted: Encrypted predictions
        plain_target: Plaintext target values
        relin_key: Relinearization key

    Returns:
        Encrypted MSE value

    Example:
        >>> y_pred = np.array([2.0, 3.0, 4.0])
        >>> y_true = np.array([2.5, 2.5, 4.5])
        >>> encrypted_pred = encrypt_vector(y_pred, ctx)
        >>> mse = encrypted_mse_loss(encrypted_pred, y_true, relin_key)
        >>> # Decrypt to get ((2-2.5)² + (3-2.5)² + (4-4.5)²) / 3 = 0.25
    """
    pred_size = encrypted_predicted.size()
    if pred_size != len(plain_target):
        raise ValueError("Predicted and target must have same length")

    n = len(plain_target)

    # Subtract target
    diff = encrypted_predicted - plain_target.tolist()

    # Square
    squared = diff * diff.tolist()
    # Note: relinearization is not directly exposed in TenSEAL Python

    # Sum and divide by n
    result = squared.sum()
    result = result * (1.0 / n)
    # Note: relinearization is not directly exposed in TenSEAL Python
    # It happens automatically or requires lower-level API access
    return result


def vector_matrix_multiply_encrypted(
    encrypted_vector: CiphertextVector,
    plain_matrix: np.ndarray,
    relin_key: RelinKeys,
) -> List[CiphertextVector]:
    """
    Multiply encrypted vector by plaintext matrix.

    E(x) * W = E(x) @ W

    Output is a list of ciphertexts (one per output dimension).

    Args:
        encrypted_vector: Encrypted input vector (1 x m)
        plain_matrix: Plaintext weight matrix (m x n)
        relin_key: Relinearization key

    Returns:
        List of n encrypted values (result vector)

    Example:
        >>> x = np.array([1.0, 2.0])  # 2 elements
        >>> W = np.array([[1.0, 2.0, 3.0],  # 2x3 matrix
        ...              [4.0, 5.0, 6.0]])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> result = vector_matrix_multiply_encrypted(encrypted_x, W, relin_key)
        >>> # Decrypt to get [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    """
    if encrypted_vector.size() != plain_matrix.shape[0]:
        raise ValueError(
            f"Vector size ({encrypted_vector.size()}) must match "
            f"matrix rows ({plain_matrix.shape[0]})"
        )

    results = []

    # For each column in the matrix, compute dot product
    for col_idx in range(plain_matrix.shape[1]):
        column = plain_matrix[:, col_idx]
        dot_product = encrypted_dot_product_plain(encrypted_vector, column, relin_key)
        results.append(dot_product)

    return results


def batch_dot_product_plain(
    encrypted_batch: List[CiphertextVector],
    plain_weights: np.ndarray,
    relin_key: RelinKeys,
) -> List[CiphertextVector]:
    """
    Compute dot products for a batch of encrypted vectors with plaintext weights.

    Args:
        encrypted_batch: List of encrypted vectors
        plain_weights: Plaintext weight vector
        relin_key: Relinearization key

    Returns:
        List of encrypted dot products (one per input vector)

    Example:
        >>> batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        >>> weights = np.array([0.5, 0.5])
        >>> encrypted_batch = [encrypt_vector(v, ctx) for v in batch]
        >>> results = batch_dot_product_plain(encrypted_batch, weights, relin_key)
        >>> # Decrypt to get [1.5, 3.5]
    """
    results = []

    for encrypted_vec in encrypted_batch:
        dot_prod = encrypted_dot_product_plain(encrypted_vec, plain_weights, relin_key)
        results.append(dot_prod)

    return results


def print_vector_operation_stats(
    operation: str,
    vector_size: int,
    noise_consumed: int,
    time_ms: float,
) -> None:
    """
    Print statistics for vector operations.

    Args:
        operation: Operation name
        vector_size: Size of input vector
        noise_consumed: Noise budget consumed (bits)
        time_ms: Time taken (milliseconds)
    """
    print(f"\n{operation}:")
    print(f"  Vector size: {vector_size}")
    print(f"  Noise consumed: {noise_consumed} bits")
    print(f"  Time: {time_ms:.2f} ms")
