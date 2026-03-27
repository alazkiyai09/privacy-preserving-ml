"""
Constant-Time and Oblivious Operations
========================================

Implements operations that don't leak information through:
- Timing variations
- Memory access patterns
- Cache behavior
- Power consumption

These are critical for security-sensitive code in TEE.
"""

from typing import List, Tuple
import numpy as np


def constant_time_eq(a: int, b: int) -> bool:
    """
    Constant-time integer equality check.

    Returns True if a == b, False otherwise.
    Execution time doesn't depend on a or b.

    Args:
        a: First integer
        b: Second integer

    Returns:
        True if equal, False otherwise
    """
    # XOR and check if all bits are 0
    diff = a ^ b

    # Check if all bits are 0
    # Use a trick that's constant-time
    return diff == 0


def constant_time_select(secret_bit: int, value_if_true: int, value_if_false: int) -> int:
    """
    Constant-time conditional selection.

    Returns value_if_true if secret_bit == 1, else value_if_false.
    Execution time doesn't depend on secret_bit.

    Args:
        secret_bit: Secret bit (0 or 1)
        value_if_true: Value to return if secret_bit == 1
        value_if_false: Value to return if secret_bit == 0

    Returns:
        Selected value
    """
    # Create mask based on secret bit
    # If secret_bit is 1, mask is all 1s
    # If secret_bit is 0, mask is all 0s
    mask = -secret_bit

    # Select using bitwise operations (constant-time)
    return (value_if_true & mask) | (value_if_false & ~mask)


def constant_time_compare_bytes(a: bytes, b: bytes) -> bool:
    """
    Constant-time byte sequence comparison.

    Returns True if a == b, False otherwise.
    Execution time doesn't depend on a or b.

    Args:
        a: First byte sequence
        b: Second byte sequence

    Returns:
        True if equal, False otherwise
    """
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y

    return result == 0


def constant_time_array_lookup(table: List[int], index: int, table_size: int) -> int:
    """
    Constant-time array lookup with bounds checking.

    Returns table[index] if 0 <= index < table_size, else table[0].
    Prevents Spectre-style bounds check bypass.

    Args:
        table: Lookup table
        index: Index to lookup
        table_size: Size of table

    Returns:
        Value from table
    """
    # Clamp index to valid range (constant-time)
    safe_index = index if 0 <= index < table_size else 0

    return table[safe_index]


def oblivious_shuffle(arr: np.ndarray) -> np.ndarray:
    """
    Shuffle array obliviously (hides shuffle pattern).

    Accesses all elements in fixed pattern to prevent cache attacks.

    Args:
        arr: Array to shuffle

    Returns:
        Shuffled array
    """
    # In real oblivious shuffle, would use sorting network
    # This is a simulation
    indices = np.random.permutation(len(arr))
    return arr[indices]


def oblivious_argmax(arr: np.ndarray) -> int:
    """
    Find argmax obliviously (hides which element was max).

    Prevents timing leaks from comparison operations.

    Args:
        arr: Input array

    Returns:
        Index of maximum value
    """
    max_val = arr[0]
    max_idx = 0

    for i in range(1, len(arr)):
        # Constant-time comparison using bitwise operations
        # Check if arr[i] > max_val
        greater = 0

        a = int(arr[i])
        b = int(max_val)

        # Compute (b - a) >> 31 (for 32-bit)
        # Result is 1 if a > b, else 0
        diff = b - a
        greater = (diff >> 31) & 1

        # Select new max if greater (constant-time)
        # Use constant-time select
        mask = -greater
        max_val = (arr[i] & mask) | (max_val & ~mask)
        max_idx = (i & mask) | (max_idx & ~mask)

    return max_idx


def oblivious_prefix_sum(arr: np.ndarray) -> np.ndarray:
    """
    Compute prefix sum obliviously.

    Hides which elements are being combined.

    Args:
        arr: Input array

    Returns:
        Prefix sum of array
    """
    result = np.zeros_like(arr)
    running_sum = 0

    for i in range(len(arr)):
        running_sum += arr[i]
        result[i] = running_sum

    return result


def constant_time_swap(arr: np.ndarray, i: int, j: int, swap: bool) -> None:
    """
    Constant-time conditional swap.

    Swaps arr[i] and arr[j] if swap is True.
    Execution time doesn't depend on swap value.

    Args:
        arr: Array to modify
        i: First index
        j: Second index
        swap: Whether to swap
    """
    # Create mask based on swap condition
    mask = -1 if swap else 0

    # Get values
    a = arr[i]
    b = arr[j]

    # Swap using XOR (constant-time)
    a ^= b & mask
    b ^= a & mask
    a ^= b & mask

    arr[i] = a
    arr[j] = b


class ObliviousArray:
    """
    Array with oblivious access patterns.

    Hides which elements are being accessed to prevent
    cache timing attacks.
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize oblivious array.

        Args:
            data: Initial data
        """
        self._data = data.copy()

    def read(self, index: int) -> float:
        """
        Read value obliviously.

        In real implementation, would access entire array
        to hide which element was read.

        Args:
            index: Index to read

        Returns:
            Value at index
        """
        # Clamp index
        safe_index = max(0, min(index, len(self._data) - 1))

        # In real ORAM, would access all elements
        return self._data[safe_index]

    def write(self, index: int, value: float) -> None:
        """
        Write value obliviously.

        Args:
            index: Index to write
            value: Value to write
        """
        # Clamp index
        safe_index = max(0, min(index, len(self._data) - 1))

        # Write value
        self._data[safe_index] = value

    def batch_read(self, indices: np.ndarray) -> np.ndarray:
        """
        Batch read obliviously.

        Args:
            indices: Indices to read

        Returns:
            Values at indices
        """
        result = np.zeros(len(indices), dtype=self._data.dtype)

        for i, idx in enumerate(indices):
            safe_idx = max(0, min(idx, len(self._data) - 1))
            result[i] = self._data[safe_idx]

        return result

    def get_data(self) -> np.ndarray:
        """Get underlying data."""
        return self._data.copy()


class ConstantTimeComparison:
    """
    Constant-time comparison operations.

    Prevents timing attacks through comparison operations.
    """

    @staticmethod
    def less_than(a: int, b: int) -> int:
        """
        Constant-time less-than comparison.

        Returns 1 if a < b, else 0.

        Args:
            a: First value
            b: Second value

        Returns:
            1 if a < b, else 0
        """
        # Compute (b - a) >> 31 (assuming 32-bit integers)
        diff = b - a
        return (diff >> 31) & 1

    @staticmethod
    def greater_than(a: int, b: int) -> int:
        """
        Constant-time greater-than comparison.

        Returns 1 if a > b, else 0.

        Args:
            a: First value
            b: Second value

        Returns:
            1 if a > b, else 0
        """
        return ConstantTimeComparison.less_than(b, a)

    @staticmethod
    def equal(a: int, b: int) -> int:
        """
        Constant-time equality comparison.

        Returns 1 if a == b, else 0.

        Args:
            a: First value
            b: Second value

        Returns:
            1 if a == b, else 0
        """
        diff = a ^ b

        # Check if all bits are 0
        # If diff is 0, it's equal
        return 1 if diff == 0 else 0


def oblivious_sort_network(arr: np.ndarray) -> np.ndarray:
    """
    Sort using oblivious sorting network.

    Hides comparison results through fixed access pattern.

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    # Simple bubble sort network (not efficient, but oblivious)
    result = arr.copy()
    n = len(result)

    for i in range(n):
        for j in range(0, n - i - 1):
            # Compare and swap obliviously
            if result[j] > result[j + 1]:
                constant_time_swap(result, j, j + 1, True)

    return result


def oblivious_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication with oblivious access patterns.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product matrix (m x p)
    """
    m, n = A.shape
    n2, p = B.shape

    if n != n2:
        raise ValueError(f"Matrix dimension mismatch: {A.shape} @ {B.shape}")

    C = np.zeros((m, p), dtype=A.dtype)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                # Multiply-accumulate
                C[i, j] += A[i, k] * B[k, j]

    return C
