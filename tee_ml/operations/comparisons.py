"""
Comparison Operations in TEE
==============================

Implements comparison operations that are IMPOSSIBLE in homomorphic
encryption but trivial in TEE.

Key limitation of HE:
- Cannot compare encrypted values
- Cannot compute argmax on encrypted data
- Cannot threshold encrypted data
- Order-preserving encryption is not secure

TEE advantages:
- Native comparison operations
- No approximations needed
- Exact results
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

from tee_ml.core.enclave import EnclaveSession


@dataclass
class ComparisonResult:
    """Result of comparison operation in TEE."""
    output: np.ndarray
    operation_name: str
    input_shape: tuple
    output_shape: tuple


def tee_argmax(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    Find argmax along axis in TEE.

    Returns index of maximum value.

    In HE: IMPOSSIBLE (requires comparisons on encrypted data)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to find argmax

    Returns:
        Indices of maximum values
    """
    result = session.execute(lambda arr: np.argmax(x, axis=axis))
    return result


def tee_argmin(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    Find argmin along axis in TEE.

    Returns index of minimum value.

    In HE: IMPOSSIBLE (requires comparisons on encrypted data)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to find argmin

    Returns:
        Indices of minimum values
    """
    result = session.execute(lambda arr: np.argmin(x, axis=axis))
    return result


def tee_threshold(
    x: np.ndarray,
    session: EnclaveSession,
    threshold: float,
) -> np.ndarray:
    """
    Apply threshold to array in TEE.

    Returns binary mask where x > threshold.

    In HE: IMPOSSIBLE (requires comparisons on encrypted data)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        threshold: Threshold value

    Returns:
        Binary mask (True where x > threshold)
    """
    result = session.execute(lambda arr: x > threshold)
    return result


def tee_equal(
    x: np.ndarray,
    session: EnclaveSession,
    value: float,
) -> np.ndarray:
    """
    Check equality with value in TEE.

    Returns binary mask where x == value.

    In HE: IMPOSSIBLE (exact comparison on encrypted data)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        value: Value to compare

    Returns:
        Binary mask (True where x == value)
    """
    result = session.execute(lambda arr: x == value)
    return result


def tee_top_k(
    x: np.ndarray,
    session: EnclaveSession,
    k: int,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top k values and indices in TEE.

    Returns (values, indices) of top k elements.

    In HE: IMPOSSIBLE (requires sorting/comparisons on encrypted data)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        k: Number of top elements to return
        axis: Axis along which to find top k

    Returns:
        Tuple of (top_values, top_indices)
    """
    def top_k_func(arr):
        # For 1D array, simplify
        if x.ndim == 1:
            # Partition to get top k
            indices = np.argpartition(x, -k)[-k:]
            # Sort just the top k
            indices_sorted = indices[np.argsort(x[indices])[::-1]]
            values = x[indices_sorted]
            return values, indices_sorted

        # For multi-dimensional arrays
        if axis is None:
            # Flatten
            x_flat = x.flatten()
            indices = np.argpartition(x_flat, -k)[-k:]
            indices_sorted = indices[np.argsort(x_flat[indices])[::-1]]
            values = x_flat[indices_sorted]
            return values, indices_sorted
        else:
            # Along specified axis
            if axis == -1:
                axis = len(x.shape) - 1

            # Apply partition along axis
            indices = np.argpartition(x, -k, axis=axis)
            # Get top k indices using proper indexing
            top_indices = np.take(indices, range(indices.shape[axis] - k, indices.shape[axis]), axis=axis)
            # Get values at those indices
            top_values = np.take_along_axis(x, top_indices, axis=axis)
            # Sort top k values
            sort_indices = np.argsort(top_values, axis=axis)[..., ::-1]
            top_values_sorted = np.take_along_axis(top_values, sort_indices, axis=axis)
            top_indices_sorted = np.take_along_axis(top_indices, sort_indices, axis=axis)

            return top_values_sorted, top_indices_sorted

    values, indices = session.execute(top_k_func)
    return values, indices


def tee_where(
    condition: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    session: EnclaveSession,
) -> np.ndarray:
    """
    Element-wise selection based on condition in TEE.

    Returns x where condition is True, y otherwise.

    In HE: IMPOSSIBLE (requires conditional on encrypted data)
    In TEE: Native operation

    Args:
        condition: Boolean array
        x: Value when condition is True
        y: Value when condition is False
        session: Active enclave session

    Returns:
        Selected values
    """
    def where_func(arr):
        return np.where(condition, x, y)

    result = session.execute(where_func)
    return result


def tee_clip(
    x: np.ndarray,
    session: EnclaveSession,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """
    Clip values to range [min_val, max_val] in TEE.

    In HE: IMPOSSIBLE (requires comparisons)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped array
    """
    result = session.execute(lambda arr: np.clip(x, min_val, max_val))
    return result


def tee_maximum(
    x: np.ndarray,
    session: EnclaveSession,
    other: float,
) -> np.ndarray:
    """
    Element-wise maximum with scalar in TEE.

    In HE: IMPOSSIBLE (requires comparison)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        other: Scalar to compare

    Returns:
        Element-wise maximum
    """
    result = session.execute(lambda arr: np.maximum(x, other))
    return result


def tee_minimum(
    x: np.ndarray,
    session: EnclaveSession,
    other: float,
) -> np.ndarray:
    """
    Element-wise minimum with scalar in TEE.

    In HE: IMPOSSIBLE (requires comparison)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        other: Scalar to compare

    Returns:
        Element-wise minimum
    """
    result = session.execute(lambda arr: np.minimum(x, other))
    return result


def tee_compare(
    x: np.ndarray,
    session: EnclaveSession,
    operator: str,
    value: float,
) -> np.ndarray:
    """
    Generic comparison operation in TEE.

    Args:
        x: Input array
        session: Active enclave session
        operator: Comparison operator ('>', '>=', '<', '<=', '==', '!=')
        value: Value to compare

    Returns:
        Boolean result array

    Raises:
        ValueError: If operator is not recognized
    """
    ops = {
        '>': lambda arr: x > value,
        '>=': lambda arr: x >= value,
        '<': lambda arr: x < value,
        '<=': lambda arr: x <= value,
        '==': lambda arr: x == value,
        '!=': lambda arr: x != value,
    }

    if operator not in ops:
        raise ValueError(f"Unknown operator: {operator}")

    result = session.execute(ops[operator])
    return result


def tee_sort(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    Sort array along axis in TEE.

    In HE: IMPOSSIBLE (requires comparisons)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to sort

    Returns:
        Sorted array
    """
    result = session.execute(lambda arr: np.sort(x, axis=axis))
    return result


def tee_argsort(
    x: np.ndarray,
    session: EnclaveSession,
    axis: int = -1,
) -> np.ndarray:
    """
    Get indices that would sort array in TEE.

    In HE: IMPOSSIBLE (requires comparisons)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to sort

    Returns:
        Indices for sorting
    """
    result = session.execute(lambda arr: np.argsort(x, axis=axis))
    return result


def tee_searchsorted(
    x: np.ndarray,
    session: EnclaveSession,
    values: np.ndarray,
    side: str = 'left',
) -> np.ndarray:
    """
    Find insertion points for values in sorted array in TEE.

    In HE: IMPOSSIBLE (requires binary search with comparisons)
    In TEE: Native operation

    Args:
        x: Sorted input array
        session: Active enclave session
        values: Values to find insertion points for
        side: 'left' or 'right'

    Returns:
        Insertion indices
    """
    def searchsorted_func():
        return np.searchsorted(x, values, side=side)

    result = session.execute(searchsorted_func)
    return result


def tee_allclose(
    x: np.ndarray,
    session: EnclaveSession,
    y: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Check if two arrays are element-wise equal within tolerance in TEE.

    In HE: IMPOSSIBLE (requires comparisons)
    In TEE: Native operation

    Args:
        x: First array
        session: Active enclave session
        y: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if arrays are close
    """
    def allclose_func(arr):
        return np.allclose(x, y, rtol=rtol, atol=atol)

    result = session.execute(allclose_func)
    return result


def tee_count_nonzero(
    x: np.ndarray,
    session: EnclaveSession,
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Count non-zero elements in TEE.

    In HE: IMPOSSIBLE (requires comparisons)
    In TEE: Native operation

    Args:
        x: Input array
        session: Active enclave session
        axis: Axis along which to count

    Returns:
        Count of non-zero elements
    """
    result = session.execute(lambda arr: np.count_nonzero(x, axis=axis))
    return result


class TeeComparisonLayer:
    """
    Generic comparison layer for TEE.

    Supports various comparison operations.
    """

    def __init__(self, operation: str, **kwargs):
        """
        Initialize comparison layer.

        Args:
            operation: Operation name ('argmax', 'threshold', 'top_k', etc.)
            **kwargs: Additional parameters (k, axis, threshold, etc.)
        """
        self.operation = operation.lower()
        self.params = kwargs

    def forward(self, x: np.ndarray, session: EnclaveSession) -> np.ndarray:
        """
        Apply comparison operation.

        Args:
            x: Input array
            session: Active enclave session

        Returns:
            Comparison result
        """
        if self.operation == 'argmax':
            axis = self.params.get('axis', -1)
            return tee_argmax(x, session, axis=axis)
        elif self.operation == 'argmin':
            axis = self.params.get('axis', -1)
            return tee_argmin(x, session, axis=axis)
        elif self.operation == 'threshold':
            threshold = self.params.get('threshold', 0.0)
            return tee_threshold(x, session, threshold=threshold)
        elif self.operation == 'clip':
            min_val = self.params.get('min', 0.0)
            max_val = self.params.get('max', 1.0)
            return tee_clip(x, session, min_val=min_val, max_val=max_val)
        elif self.operation == 'sort':
            axis = self.params.get('axis', -1)
            return tee_sort(x, session, axis=axis)
        elif self.operation == 'top_k':
            k = self.params.get('k', 5)
            axis = self.params.get('axis', -1)
            values, indices = tee_top_k(x, session, k=k, axis=axis)
            return values, indices
        else:
            raise ValueError(f"Unknown comparison operation: {self.operation}")

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if params_str:
            return f"TeeComparisonLayer(operation='{self.operation}', {params_str})"
        return f"TeeComparisonLayer(operation='{self.operation}')"
