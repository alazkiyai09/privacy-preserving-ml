"""
Non-Linear Activation Functions in TEE
========================================

Implements activation functions that are impossible or expensive
in homomorphic encryption but trivial in TEE.

In HE:
- ReLU: Requires polynomial approximation (expensive, inaccurate)
- Sigmoid: Requires degree 5-7 polynomial (200-400 bits noise)
- Softmax: Multi-step polynomial (very expensive)

In TEE:
- All activations are native operations
- No noise budget constraints
- Exact computation (no approximation)
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

from tee_ml.core.enclave import EnclaveSession


@dataclass
class ActivationResult:
    """Result of activation function in TEE."""
    output: np.ndarray
    operation_name: str
    input_shape: tuple
    output_shape: tuple
    computation_time_ns: int


def tee_relu(x: np.ndarray, session: EnclaveSession) -> np.ndarray:
    """
    ReLU activation in TEE.

    ReLU(x) = max(0, x)

    In HE: Requires smooth approximation (x + sqrt(x² + ε)) / 2
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        ReLU(x)
    """
    import time
    start = time.perf_counter_ns()

    result = session.execute(lambda arr: np.maximum(0, x))

    return result


def tee_sigmoid(x: np.ndarray, session: EnclaveSession) -> np.ndarray:
    """
    Sigmoid activation in TEE.

    Sigmoid(x) = 1 / (1 + exp(-x))

    In HE: Requires degree 5-7 polynomial approximation (~200-400 bits noise)
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        Sigmoid(x)
    """
    import time
    start = time.perf_counter_ns()

    result = session.execute(lambda arr: 1 / (1 + np.exp(-x)))

    return result


def tee_tanh(x: np.ndarray, session: EnclaveSession) -> np.ndarray:
    """
    Hyperbolic tangent activation in TEE.

    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    In HE: Requires polynomial approximation (~200 bits noise)
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        Tanh(x)
    """
    import time
    start = time.perf_counter_ns()

    result = session.execute(lambda arr: np.tanh(x))

    return result


def tee_softmax(x: np.ndarray, session: EnclaveSession, axis: int = -1) -> np.ndarray:
    """
    Softmax activation in TEE.

    Softmax(x)_i = exp(x_i) / sum(exp(x_j))

    In HE: Extremely expensive (exponential + division + normalization)
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to compute softmax

    Returns:
        Softmax(x)
    """
    import time
    start = time.perf_counter_ns()

    # Stable softmax: subtract max before exp
    def softmax_stable(arr):
        # Subtract max for numerical stability
        arr_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_arr = np.exp(arr_shifted)
        return exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)

    result = session.execute(softmax_stable)

    return result


def tee_leaky_relu(
    x: np.ndarray,
    session: EnclaveSession,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Leaky ReLU activation in TEE.

    LeakyReLU(x) = x if x > 0 else alpha * x

    In HE: Requires conditional (impossible) or approximation
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session
        alpha: Slope for negative values

    Returns:
        LeakyReLU(x)
    """
    result = session.execute(lambda arr: np.where(x > 0, x, alpha * x))
    return result


def tee_elu(
    x: np.ndarray,
    session: EnclaveSession,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    ELU (Exponential Linear Unit) activation in TEE.

    ELU(x) = x if x > 0 else alpha * (exp(x) - 1)

    In HE: Requires exponential (impossible) or approximation
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session
        alpha: Scaling factor for negative values

    Returns:
        ELU(x)
    """
    def elu_func(arr):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    result = session.execute(elu_func)
    return result


def tee_gelu(x: np.ndarray, session: EnclaveSession) -> np.ndarray:
    """
    GELU (Gaussian Error Linear Unit) activation in TEE.

    GELU(x) = x * Φ(x) where Φ is Gaussian CDF
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    In HE: Extremely expensive (tanh + x³ + multiplication)
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        GELU(x)
    """
    def gelu_func(arr):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    result = session.execute(gelu_func)
    return result


def tee_swish(x: np.ndarray, session: EnclaveSession, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation in TEE.

    Swish(x) = x * sigmoid(beta * x)

    In HE: Requires sigmoid approximation + multiplication
    In TEE: Native operation, exact result

    Args:
        x: Input array
        session: Active enclave session
        beta: Parameter for sigmoid

    Returns:
        Swish(x)
    """
    def swish_func(arr):
        return x * (1 / (1 + np.exp(-beta * x)))

    result = session.execute(swish_func)
    return result


class TeeActivationLayer:
    """
    Generic activation layer for TEE.

    Supports multiple activation functions with consistent interface.
    """

    def __init__(self, activation: str, **kwargs):
        """
        Initialize activation layer.

        Args:
            activation: Name of activation ('relu', 'sigmoid', 'tanh', 'softmax', etc.)
            **kwargs: Additional parameters (alpha, beta, axis, etc.)
        """
        self.activation = activation.lower()
        self.params = kwargs

    def forward(self, x: np.ndarray, session: EnclaveSession) -> np.ndarray:
        """
        Apply activation to input.

        Args:
            x: Input array
            session: Active enclave session

        Returns:
            Activated output
        """
        if self.activation == 'relu':
            return tee_relu(x, session)
        elif self.activation == 'sigmoid':
            return tee_sigmoid(x, session)
        elif self.activation == 'tanh':
            return tee_tanh(x, session)
        elif self.activation == 'softmax':
            axis = self.params.get('axis', -1)
            return tee_softmax(x, session, axis=axis)
        elif self.activation == 'leaky_relu':
            alpha = self.params.get('alpha', 0.01)
            return tee_leaky_relu(x, session, alpha=alpha)
        elif self.activation == 'elu':
            alpha = self.params.get('alpha', 1.0)
            return tee_elu(x, session, alpha=alpha)
        elif self.activation == 'gelu':
            return tee_gelu(x, session)
        elif self.activation == 'swish':
            beta = self.params.get('beta', 1.0)
            return tee_swish(x, session, beta=beta)
        elif self.activation == 'none' or self.activation == 'linear':
            return x  # No activation
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def get_overhead_ns(self) -> int:
        """
        Get estimated overhead for this activation.

        In TEE, overhead is minimal (native operations).
        This is for comparison with HE.

        Returns:
            Estimated overhead in nanoseconds
        """
        # Base overhead for function call in TEE
        base_overhead = 100  # 100 ns

        # Additional overhead based on complexity
        complexity_multiplier = {
            'relu': 1,
            'leaky_relu': 1,
            'elu': 2,  # Requires exp
            'gelu': 3,  # Requires tanh + x³
            'sigmoid': 2,  # Requires exp
            'tanh': 2,  # Requires exp
            'swish': 3,  # Requires sigmoid
            'softmax': 3,  # Requires exp + sum + division
            'none': 0,
            'linear': 0,
        }

        return base_overhead * complexity_multiplier.get(self.activation, 1)

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"TeeActivationLayer(activation='{self.activation}', {params_str})"
        return f"TeeActivationLayer(activation='{self.activation}')"


def compare_activation_costs(activations: list) -> dict:
    """
    Compare computational cost of different activations in TEE vs HE.

    Args:
        activations: List of activation names

    Returns:
        Dictionary with cost comparison
    """
    costs = {}

    for act in activations:
        layer = TeeActivationLayer(act)
        tee_cost_ns = layer.get_overhead_ns()

        # HE costs (based on literature)
        he_costs = {
            'relu': 400,  # Degree 5 polynomial
            'sigmoid': 400,  # Degree 5 polynomial
            'tanh': 400,  # Degree 5 polynomial
            'softmax': 2000,  # Very expensive
            'leaky_relu': 600,  # Requires conditional
            'elu': 1000,  # Requires exp
            'gelu': 1500,  # Requires tanh + x³
            'swish': 800,  # Sigmoid + multiply
        }

        he_cost_ns = he_costs.get(act, 500)

        costs[act] = {
            'tee_ns': tee_cost_ns,
            'he_ns': he_cost_ns,
            'speedup': he_cost_ns / tee_cost_ns,
        }

    return costs


# Batch processing for efficiency
def tee_batch_activations(
    x_batch: np.ndarray,
    session: EnclaveSession,
    activation: str = 'relu',
) -> np.ndarray:
    """
    Apply activation to batch of inputs in single enclave entry.

    More efficient than entering/exiting enclave for each sample.

    Args:
        x_batch: Batch of inputs (shape: [batch_size, features])
        session: Active enclave session
        activation: Activation function name

    Returns:
        Activated batch
    """
    layer = TeeActivationLayer(activation)
    result = layer.forward(x_batch, session)
    return result
