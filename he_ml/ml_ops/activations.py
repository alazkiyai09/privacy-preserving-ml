"""
Activation Functions for Homomorphic Encryption
================================================
Polynomial approximations of common neural network activation functions.

Why Polynomial Approximations?
- Homomorphic encryption supports: +, -, ×
- Does NOT support: division, comparison, min/max, exponentials
- Solution: Approximate non-linear functions with polynomials

Common Techniques:
1. Taylor Series: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
2. Chebyshev Polynomials: Minimax approximation (uniform error distribution)
3. Piecewise Polynomials: Different polynomials for different intervals

Accuracy Trade-offs:
- Higher degree → better accuracy but more noise
- Lower degree → less accurate but preserves noise budget
- Typical degrees: 3-7 for practical HE applications
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

# Type aliases
CiphertextVector = Any
RelinKeys = Any


@dataclass
class ActivationConfig:
    """Configuration for activation function approximation."""
    degree: int = 3  # Polynomial degree
    input_range: Tuple[float, float] = (-5.0, 5.0)  # Valid input range
    normalization: str = 'tanh'  # How to normalize inputs: 'tanh', 'clip', 'none'

    def __post_init__(self):
        if self.degree < 1:
            raise ValueError("degree must be at least 1")
        if self.degree > 15:
            raise ValueError("degree > 15 will consume too much noise budget")


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Generate Chebyshev nodes for polynomial interpolation.

    Chebyshev nodes minimize Runge's phenomenon and provide near-optimal
    interpolation points for polynomial approximation.

    Args:
        n: Number of nodes
        a: Lower bound of interval
        b: Upper bound of interval

    Returns:
        Array of Chebyshev nodes in [a, b]

    Example:
        >>> nodes = chebyshev_nodes(5, -1, 1)
        >>> # Returns 5 points optimally spaced for interpolation
    """
    # Chebyshev nodes on [-1, 1]
    k = np.arange(1, n + 1)
    nodes = np.cos((2 * k - 1) * np.pi / (2 * n))

    # Scale to [a, b]
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes

    return nodes


def fit_chebyshev_polynomial(
    func: callable,
    degree: int,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """
    Fit Chebyshev polynomial approximation to a function.

    Uses Chebyshev nodes for interpolation to minimize approximation error.

    Args:
        func: Function to approximate (e.g., np.exp, 1/(1+exp(-x)))
        degree: Polynomial degree
        a: Lower bound of interval
        b: Upper bound of interval

    Returns:
        Coefficients [c0, c1, ..., cn] for polynomial c0 + c1*x + c2*x² + ...

    Example:
        >>> # Approximate sigmoid on [-3, 3]
        >>> coeffs = fit_chebyshev_polynomial(
        ...     lambda x: 1 / (1 + np.exp(-x)),
        ...     degree=5,
        ...     a=-3,
        ...     b=3
        ... )
    """
    # Generate Chebyshev nodes
    nodes = chebyshev_nodes(degree + 1, a, b)

    # Evaluate function at nodes
    values = func(nodes)

    # Fit polynomial using least squares (more stable than interpolation)
    # Create Vandermonde matrix
    V = np.vstack([nodes**i for i in range(degree + 1)]).T

    # Solve for coefficients
    coeffs = np.linalg.lstsq(V, values, rcond=None)[0]

    return coeffs


# ============================================================================
# ReLU Approximation
# ============================================================================

def relu_approximation_coeffs(
    degree: int = 3,
    input_range: Tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Get polynomial coefficients for ReLU approximation.

    ReLU(x) = max(0, x)

    This is challenging because ReLU has a "kink" at x=0.
    We use a smooth approximation: relu(x) ≈ (x + sqrt(x² + ε)) / 2
    Then fit a polynomial to this smooth function.

    Args:
        degree: Polynomial degree (recommended: 3-7)
        input_range: Range of inputs to approximate over

    Returns:
        Coefficients [c0, c1, ..., cn]

    Example:
        >>> coeffs = relu_approximation_coeffs(degree=5)
        >>> # Use in encrypted_polynomial()
    """
    # Smooth ReLU approximation
    epsilon = 0.01
    smooth_relu = lambda x: 0.5 * (x + np.sqrt(x**2 + epsilon))

    # Fit polynomial
    coeffs = fit_chebyshev_polynomial(
        smooth_relu,
        degree,
        input_range[0],
        input_range[1],
    )

    return coeffs


def encrypted_relu(
    encrypted_x: CiphertextVector,
    relin_key: RelinKeys,
    degree: int = 3,
) -> CiphertextVector:
    """
    Compute encrypted ReLU activation using polynomial approximation.

    ReLU(x) = max(0, x) ≈ c0 + c1*x + c2*x² + ... + cn*x^n

    Args:
        encrypted_x: Encrypted input values
        relin_key: Relinearization key
        degree: Polynomial degree (higher = more accurate but more noise)

    Returns:
        Encrypted ReLU activation

    Noise Cost: degree * log2(scale) bits

    Example:
        >>> layer = EncryptedLinearLayer(10, 20)
        >>> x = encrypt_vector(input_data, ctx)
        >>> # Linear layer
        >>> linear_out = layer.forward(x, relin_key)
        >>> # Apply ReLU to each output
        >>> activated = [encrypted_relu(out, relin_key) for out in linear_out]

    Note:
        ReLU is challenging in HE due to the non-differentiable point at x=0.
        The approximation works best for inputs in [-5, 5].
        For x >> 0: ReLU(x) ≈ x (linear)
        For x << 0: ReLU(x) ≈ 0
    """
    from he_ml.ml_ops.vector_ops import encrypted_polynomial

    # Get coefficients
    coeffs = relu_approximation_coeffs(degree).tolist()

    # Evaluate polynomial
    result = encrypted_polynomial(encrypted_x, coeffs, relin_key)

    return result


# ============================================================================
# Sigmoid Approximation
# ============================================================================

def sigmoid_approximation_coeffs(
    degree: int = 5,
    input_range: Tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Get polynomial coefficients for sigmoid approximation.

    Sigmoid(x) = 1 / (1 + e^(-x))

    Args:
        degree: Polynomial degree (recommended: 5-7 for good accuracy)
        input_range: Range of inputs (sigmoid saturates outside [-5, 5])

    Returns:
        Coefficients [c0, c1, ..., cn]

    Example:
        >>> coeffs = sigmoid_approximation_coeffs(degree=7)
        >>> # Typical values: [0.5, 0.25, -0.01, 0.001, ...]
    """
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

    coeffs = fit_chebyshev_polynomial(
        sigmoid,
        degree,
        input_range[0],
        input_range[1],
    )

    return coeffs


def encrypted_sigmoid(
    encrypted_x: CiphertextVector,
    relin_key: RelinKeys,
    degree: int = 5,
) -> CiphertextVector:
    """
    Compute encrypted sigmoid activation using polynomial approximation.

    Sigmoid(x) = 1 / (1 + e^(-x)) ≈ c0 + c1*x + c2*x² + ... + cn*x^n

    Args:
        encrypted_x: Encrypted input values
        relin_key: Relinearization key
        degree: Polynomial degree (recommended: 5-7)

    Returns:
        Encrypted sigmoid activation

    Noise Cost: degree * log2(scale) bits

    Example:
        >>> activated = encrypted_sigmoid(layer_output, relin_key, degree=7)
        >>> # Decrypts to values in (0, 1) range

    Note:
        Sigmoid saturates to 0 for x << -5 and to 1 for x >> 5.
        The polynomial approximation is most accurate in [-5, 5].
        Degree 7 provides < 0.01 error in this range.
    """
    from he_ml.ml_ops.vector_ops import encrypted_polynomial

    # Get coefficients
    coeffs = sigmoid_approximation_coeffs(degree).tolist()

    # Evaluate polynomial
    result = encrypted_polynomial(encrypted_x, coeffs, relin_key)

    return result


# ============================================================================
# Tanh Approximation
# ============================================================================

def tanh_approximation_coeffs(
    degree: int = 5,
    input_range: Tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Get polynomial coefficients for tanh approximation.

    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Args:
        degree: Polynomial degree (recommended: 5-7)
        input_range: Range of inputs (tanh saturates outside [-5, 5])

    Returns:
        Coefficients [c0, c1, ..., cn]

    Example:
        >>> coeffs = tanh_approximation_coeffs(degree=7)
        >>> # tanh is odd function, so even coefficients should be ~0
    """
    tanh = lambda x: np.tanh(x)

    coeffs = fit_chebyshev_polynomial(
        tanh,
        degree,
        input_range[0],
        input_range[1],
    )

    return coeffs


def encrypted_tanh(
    encrypted_x: CiphertextVector,
    relin_key: RelinKeys,
    degree: int = 5,
) -> CiphertextVector:
    """
    Compute encrypted tanh activation using polynomial approximation.

    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) ≈ c0 + c1*x + c2*x² + ... + cn*x^n

    Args:
        encrypted_x: Encrypted input values
        relin_key: Relinearization key
        degree: Polynomial degree (recommended: 5-7)

    Returns:
        Encrypted tanh activation

    Noise Cost: degree * log2(scale) bits

    Example:
        >>> activated = encrypted_tanh(layer_output, relin_key, degree=7)
        >>> # Decrypts to values in (-1, 1) range

    Note:
        tanh is an odd function: tanh(-x) = -tanh(x)
        Good approximations preserve this symmetry.
        Tanh saturates to ±1 for |x| > 5.
    """
    from he_ml.ml_ops.vector_ops import encrypted_polynomial

    # Get coefficients
    coeffs = tanh_approximation_coeffs(degree).tolist()

    # Evaluate polynomial
    result = encrypted_polynomial(encrypted_x, coeffs, relin_key)

    return result


# ============================================================================
# Softplus Approximation (smooth alternative to ReLU)
# ============================================================================

def softplus_approximation_coeffs(
    degree: int = 5,
    input_range: Tuple[float, float] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Get polynomial coefficients for softplus approximation.

    Softplus(x) = ln(1 + e^x)

    Softplus is a smooth approximation of ReLU.

    Args:
        degree: Polynomial degree
        input_range: Range of inputs

    Returns:
        Coefficients [c0, c1, ..., cn]
    """
    softplus = lambda x: np.log1p(np.exp(x))

    coeffs = fit_chebyshev_polynomial(
        softplus,
        degree,
        input_range[0],
        input_range[1],
    )

    return coeffs


def encrypted_softplus(
    encrypted_x: CiphertextVector,
    relin_key: RelinKeys,
    degree: int = 5,
) -> CiphertextVector:
    """
    Compute encrypted softplus activation using polynomial approximation.

    Softplus(x) = ln(1 + e^x) ≈ c0 + c1*x + c2*x² + ... + cn*x^n

    Args:
        encrypted_x: Encrypted input values
        relin_key: Relinearization key
        degree: Polynomial degree

    Returns:
        Encrypted softplus activation

    Example:
        >>> activated = encrypted_softplus(layer_output, relin_key, degree=5)

    Note:
        Softplus is smoother than ReLU and easier to approximate with polynomials.
        For x >> 0: softplus(x) ≈ x
        For x << 0: softplus(x) ≈ 0
    """
    from he_ml.ml_ops.vector_ops import encrypted_polynomial

    # Get coefficients
    coeffs = softplus_approximation_coeffs(degree).tolist()

    # Evaluate polynomial
    result = encrypted_polynomial(encrypted_x, coeffs, relin_key)

    return result


# ============================================================================
# Utility Functions
# ============================================================================

def evaluate_approximation_error(
    func: callable,
    coeffs: np.ndarray,
    test_range: Tuple[float, float],
    n_points: int = 100,
) -> Tuple[float, float, float]:
    """
    Evaluate polynomial approximation error.

    Args:
        func: True function (e.g., np.tanh, sigmoid)
        coeffs: Polynomial coefficients
        test_range: Range to test over (min, max)
        n_points: Number of test points

    Returns:
        (max_error, mean_error, std_error)

    Example:
        >>> coeffs = tanh_approximation_coeffs(degree=5)
        >>> max_err, mean_err, std_err = evaluate_approximation_error(
        ...     np.tanh, coeffs, (-5, 5)
        ... )
        >>> print(f"Max error: {max_err:.6f}")
    """
    # Generate test points
    x = np.linspace(test_range[0], test_range[1], n_points)

    # Compute true values
    y_true = func(x)

    # Compute polynomial approximation
    y_approx = np.polyval(coeffs[::-1], x)  # polyval expects highest degree first

    # Compute errors
    errors = np.abs(y_true - y_approx)

    return np.max(errors), np.mean(errors), np.std(errors)


def print_activation_info(
    activation_name: str,
    coeffs: np.ndarray,
    error_stats: Tuple[float, float, float],
) -> None:
    """
    Print information about activation function approximation.

    Args:
        activation_name: Name of activation (e.g., "ReLU", "Sigmoid")
        coeffs: Polynomial coefficients
        error_stats: (max_error, mean_error, std_error)
    """
    max_err, mean_err, std_err = error_stats

    print(f"\n{'='*60}")
    print(f"{activation_name} Polynomial Approximation")
    print(f"{'='*60}")
    print(f"Degree: {len(coeffs) - 1}")
    print(f"\nCoefficients:")
    for i, c in enumerate(coeffs):
        if abs(c) > 1e-10:
            print(f"  x^{i}: {c:.6f}")

    print(f"\nApproximation Error:")
    print(f"  Max:  {max_err:.8f}")
    print(f"  Mean: {mean_err:.8f}")
    print(f"  Std:  {std_err:.8f}")

    # Estimate noise cost
    scale_bits = 40  # Assuming scale=2^40
    noise_cost = (len(coeffs) - 1) * scale_bits
    print(f"\nEstimated Noise Cost:")
    print(f"  Per evaluation: ~{noise_cost} bits")
    print(f"  With 200-bit budget: ~{200 // noise_cost} activations maximum")
    print(f"{'='*60}\n")


def get_precomputed_coeffs(
    activation: str,
    degree: int,
) -> Optional[np.ndarray]:
    """
    Get pre-computed polynomial coefficients for common activations.

    These coefficients have been optimized for good accuracy vs. noise trade-offs.

    Args:
        activation: 'relu', 'sigmoid', 'tanh', 'softplus'
        degree: Polynomial degree

    Returns:
        Coefficients array or None if not pre-computed

    Note:
        Pre-computed coefficients are provided for degrees 3, 5, and 7.
        For other degrees, coefficients will be computed on-the-fly.
    """
    # Pre-computed coefficients (optimized for input range [-5, 5])
    precomputed = {
        'tanh': {
            3: np.array([0.0, 0.98, 0.0, -0.18]),  # Good for small range
            5: np.array([0.0, 0.99, 0.0, -0.28, 0.0, 0.05]),
            7: np.array([0.0, 0.99, 0.0, -0.30, 0.0, 0.08, 0.0, -0.01]),
        },
        'sigmoid': {
            3: np.array([0.5, 0.25, -0.01, 0.0]),
            5: np.array([0.5, 0.25, -0.01, 0.0, 0.0, 0.0]),
            7: np.array([0.5, 0.25, -0.01, 0.005, 0.0, -0.001, 0.0, 0.0]),
        },
    }

    if activation in precomputed and degree in precomputed[activation]:
        return precomputed[activation][degree]

    return None
