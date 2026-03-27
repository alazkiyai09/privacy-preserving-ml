"""
Arithmetic Operations in TEE
==============================

Implements arithmetic operations that are DIFFICULT or EXPENSIVE in
homomorphic encryption but trivial in TEE.

Key limitation of HE:
- Division is very expensive (requires polynomial approximation)
- Non-linear operations are costly
- Normalization requires multiple expensive operations

TEE advantages:
- Native arithmetic operations
- No noise budget concerns
- Exact results
"""

from typing import Optional, Tuple
import numpy as np

from tee_ml.core.enclave import EnclaveSession


def tee_divide(
    x: np.ndarray,
    session: EnclaveSession,
    divisor: float,
) -> np.ndarray:
    """
    Divide array by scalar in TEE.

    In HE: VERY EXPENSIVE (requires polynomial approximation of 1/x)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        divisor: Divisor

    Returns:
        x / divisor
    """
    result = session.execute(lambda arr: x / divisor)
    return result


def tee_reciprocal(
    x: np.ndarray,
    session: EnclaveSession,
) -> np.ndarray:
    """
    Compute element-wise reciprocal in TEE.

    In HE: VERY EXPENSIVE (requires polynomial approximation)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        1 / x
    """
    result = session.execute(lambda arr: 1.0 / x)
    return result


def tee_power(
    x: np.ndarray,
    session: EnclaveSession,
    exponent: float,
) -> np.ndarray:
    """
    Compute element-wise power in TEE.

    In HE: EXPENSIVE for non-integer exponents
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        exponent: Exponent

    Returns:
        x ** exponent
    """
    result = session.execute(lambda arr: np.power(x, exponent))
    return result


def tee_sqrt(
    x: np.ndarray,
    session: EnclaveSession,
) -> np.ndarray:
    """
    Compute element-wise square root in TEE.

    In HE: EXPENSIVE (requires polynomial approximation)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        sqrt(x)
    """
    result = session.execute(lambda arr: np.sqrt(x))
    return result


def tee_normalize(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize array along axis (L2 normalization) in TEE.

    x_normalized = x / ||x||_2

    In HE: VERY EXPENSIVE (sqrt + division)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to normalize
        eps: Small value for numerical stability

    Returns:
        Normalized array
    """
    def normalize_func(arr):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    result = session.execute(normalize_func)
    return result


def tee_layer_normalization(
    x: np.ndarray,
    session: EnclaveSession,
    gamma: np.ndarray,
    beta: np.ndarray,
    axis: int = -1,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Layer normalization in TEE.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    In HE: EXTREMELY EXPENSIVE (mean + variance + sqrt + division)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        gamma: Scale parameter
        beta: Shift parameter
        axis: Axis to normalize over
        eps: Small value for numerical stability

    Returns:
        Normalized and scaled array
    """
    def layer_norm_func(arr):
        # Calculate mean and variance
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)

        # Normalize
        normalized = (x - mean) / np.sqrt(var + eps)

        # Scale and shift
        return gamma * normalized + beta

    result = session.execute(layer_norm_func)
    return result


def tee_batch_normalization(
    x: np.ndarray,
    session: EnclaveSession,
    mean: np.ndarray,
    var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    axis: int = 0,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Batch normalization (inference mode) in TEE.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    In HE: EXTREMELY EXPENSIVE (division + sqrt + multiplication)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        mean: Running mean (from training)
        var: Running variance (from training)
        gamma: Scale parameter
        beta: Shift parameter
        axis: Axis to normalize over
        eps: Small value for numerical stability

    Returns:
        Normalized and scaled array
    """
    def batch_norm_func(arr):
        # Normalize using running statistics
        normalized = (x - mean) / np.sqrt(var + eps)

        # Scale and shift
        return gamma * normalized + beta

    result = session.execute(batch_norm_func)
    return result


def tee_standardize(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Standardize array (z-score normalization) in TEE.

    z = (x - mean) / std

    In HE: VERY EXPENSIVE (mean + std + division)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to standardize
        eps: Small value for numerical stability

    Returns:
        Standardized array
    """
    def standardize_func(arr):
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        return (x - mean) / (std + eps)

    result = session.execute(standardize_func)
    return result


def tee_min_max_scale(
    x: np.ndarray,
    session: EnclaveSession,
    feature_range: Tuple[float, float] = (0, 1),
    axis: int = -1,
) -> np.ndarray:
    """
    Min-max scaling in TEE.

    x_scaled = (x - min) / (max - min) * (new_max - new_min) + new_min

    In HE: EXPENSIVE (min/max + division + multiplication)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        feature_range: Target range (min, max)
        axis: Axis along which to scale

    Returns:
        Scaled array
    """
    def min_max_func(arr):
        min_val = np.min(x, axis=axis, keepdims=True)
        max_val = np.max(x, axis=axis, keepdims=True)

        # Scale to [0, 1]
        scaled = (x - min_val) / (max_val - min_val + 1e-8)

        # Scale to target range
        new_min, new_max = feature_range
        scaled = scaled * (new_max - new_min) + new_min

        return scaled

    result = session.execute(min_max_func)
    return result


def tee_log(
    x: np.ndarray,
    session: EnclaveSession,
) -> np.ndarray:
    """
    Compute natural logarithm in TEE.

    In HE: VERY EXPENSIVE (requires polynomial approximation)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        log(x)
    """
    result = session.execute(lambda arr: np.log(x))
    return result


def tee_exp(
    x: np.ndarray,
    session: EnclaveSession,
) -> np.ndarray:
    """
    Compute exponential in TEE.

    In HE: VERY EXPENSIVE (requires polynomial approximation)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session

    Returns:
        exp(x)
    """
    result = session.execute(lambda arr: np.exp(x))
    return result


def tee_log_softmax(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    Log-softmax in TEE.

    LogSoftmax(x) = x - log(sum(exp(x)))

    More numerically stable than softmax + log.

    In HE: EXTREMELY EXPENSIVE
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to compute

    Returns:
        Log-softmax output
    """
    def log_softmax_func(arr):
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))

    result = session.execute(log_softmax_func)
    return result


def tee_l2_normalize(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    L2 normalization (same as normalize) in TEE.

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to normalize

    Returns:
        L2-normalized array
    """
    return tee_normalize(x, session, axis=axis)


def tee_l1_normalize(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    L1 normalization in TEE.

    x_normalized = x / ||x||_1

    In HE: EXPENSIVE (absolute sum + division)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to normalize
        eps: Small value for numerical stability

    Returns:
        L1-normalized array
    """
    def l1_normalize_func(arr):
        norm = np.sum(np.abs(x), axis=axis, keepdims=True)
        return x / (norm + eps)

    result = session.execute(l1_normalize_func)
    return result


def tee_clip_and_normalize(
    x: np.ndarray,
    session: EnclaveSession,
    min_val: float = -5.0,
    max_val: float = 5.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Clip values then normalize in TEE.

    Two-step operation that's very expensive in HE.

    Args:
        x: Input array
        session: Active enclave session
        min_val: Minimum value for clipping
        max_val: Maximum value for clipping
        axis: Axis along which to normalize

    Returns:
        Clipped and normalized array
    """
    def clip_normalize_func(arr):
        # Clip
        clipped = np.clip(x, min_val, max_val)
        # Normalize
        norm = np.linalg.norm(clipped, axis=axis, keepdims=True)
        return clipped / (norm + 1e-8)

    result = session.execute(clip_normalize_func)
    return result


class TeeArithmeticLayer:
    """
    Generic arithmetic layer for TEE.

    Supports various arithmetic and normalization operations.
    """

    def __init__(self, operation: str, **kwargs):
        """
        Initialize arithmetic layer.

        Args:
            operation: Operation name ('divide', 'normalize', 'layer_norm', etc.)
            **kwargs: Additional parameters
        """
        self.operation = operation.lower()
        self.params = kwargs

    def forward(self, x: np.ndarray, session: EnclaveSession) -> np.ndarray:
        """
        Apply arithmetic operation.

        Args:
            x: Input array
            session: Active enclave session

        Returns:
            Operation result
        """
        if self.operation == 'divide':
            divisor = self.params.get('divisor', 1.0)
            return tee_divide(x, session, divisor=divisor)
        elif self.operation == 'normalize':
            axis = self.params.get('axis', -1)
            return tee_normalize(x, session, axis=axis)
        elif self.operation == 'layer_norm':
            gamma = self.params.get('gamma', np.ones(x.shape[-1]))
            beta = self.params.get('beta', np.zeros(x.shape[-1]))
            axis = self.params.get('axis', -1)
            return tee_layer_normalization(x, session, gamma, beta, axis=axis)
        elif self.operation == 'batch_norm':
            mean = self.params['mean']
            var = self.params['var']
            gamma = self.params.get('gamma', np.ones_like(mean))
            beta = self.params.get('beta', np.zeros_like(mean))
            return tee_batch_normalization(x, session, mean, var, gamma, beta)
        elif self.operation == 'standardize':
            axis = self.params.get('axis', -1)
            return tee_standardize(x, session, axis=axis)
        elif self.operation == 'min_max_scale':
            feature_range = self.params.get('feature_range', (0, 1))
            axis = self.params.get('axis', -1)
            return tee_min_max_scale(x, session, feature_range=feature_range, axis=axis)
        elif self.operation == 'log':
            return tee_log(x, session)
        elif self.operation == 'exp':
            return tee_exp(x, session)
        elif self.operation == 'sqrt':
            return tee_sqrt(x, session)
        elif self.operation == 'power':
            exponent = self.params.get('exponent', 2.0)
            return tee_power(x, session, exponent=exponent)
        else:
            raise ValueError(f"Unknown arithmetic operation: {self.operation}")

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"TeeArithmeticLayer(operation='{self.operation}', {params_str})"
        return f"TeeArithmeticLayer(operation='{self.operation}')"
