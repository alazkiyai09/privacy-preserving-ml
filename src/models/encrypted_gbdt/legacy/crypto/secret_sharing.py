"""
Additive Secret Sharing primitives for secure computation.

Implements (t, n)-threshold secret sharing where a secret is split into n shares,
and any t shares can reconstruct the secret, but fewer than t shares reveal nothing.

For privacy-preserving GBDT, we use additive secret sharing where shares sum to
the secret in a finite field.
"""

import numpy as np
from typing import List, Tuple, Optional
import random


class SecretSharing:
    """
    Additive secret sharing over finite fields.

    A secret s is split into n shares: [s1, s2, ..., sn]
    such that: s = (s1 + s2 + ... + sn) mod p

    where p is a large prime (the field modulus).
    """

    def __init__(self, field_modulus: Optional[int] = None):
        """
        Initialize secret sharing scheme.

        Args:
            field_modulus: Prime modulus for the field. If None, uses default large prime.
                          Default: 2^31 - 1 (a Mersenne prime that fits in float64)
        """
        if field_modulus is None:
            # Use 2^31 - 1 = 2147483647, a Mersenne prime that fits in float64 exactly
            self.field_modulus = (1 << 31) - 1
        else:
            # Ensure modulus is prime (basic check)
            if not self._is_prime(field_modulus):
                raise ValueError(f"Field modulus must be prime, got {field_modulus}")
            self.field_modulus = field_modulus

    def _is_prime(self, n: int) -> bool:
        """Simple primality test."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def share(self, value: int, n_parties: int) -> np.ndarray:
        """
        Split a value into n additive shares.

        The secret s is split as:
        s = s1 + s2 + ... + sn-1 + sn (mod p)

        where s1, ..., sn-1 are random and sn = s - (s1 + ... + sn-1) mod p.

        Args:
            value: Secret value to share (can be negative, will be converted to field element)
            n_parties: Number of parties to share among

        Returns:
            Array of n shares that sum to value (mod field_modulus)

        Raises:
            ValueError: If n_parties < 2
        """
        if n_parties < 2:
            raise ValueError("Need at least 2 parties for secret sharing")

        # Convert value to field element (handle negative values)
        field_value = value % self.field_modulus

        # Generate n-1 random shares
        shares = np.zeros(n_parties, dtype=np.int64)
        for i in range(n_parties - 1):
            shares[i] = random.randint(0, self.field_modulus - 1)

        # Last share is determined by the secret
        sum_random = np.sum(shares[:-1]) % self.field_modulus
        shares[-1] = (field_value - sum_random) % self.field_modulus

        return shares

    def share_array(self, values: np.ndarray, n_parties: int) -> np.ndarray:
        """
        Split an array of values into n shares.

        Each value is independently shared.

        Args:
            values: Array of secrets (n_values,)
            n_parties: Number of parties to share among

        Returns:
            Array of shares (n_parties, n_values)
        """
        n_values = len(values)
        shares = np.zeros((n_parties, n_values), dtype=np.int64)

        for i, value in enumerate(values):
            shares[:, i] = self.share(value, n_parties)

        return shares

    def reconstruct(self, shares: np.ndarray) -> int:
        """
        Reconstruct secret from shares.

        Args:
            shares: Array of shares that sum to the secret

        Returns:
            Reconstructed secret value

        Raises:
            ValueError: If shares is empty
        """
        if len(shares) == 0:
            raise ValueError("Cannot reconstruct from empty shares")

        # Sum shares modulo field modulus
        secret = np.sum(shares) % self.field_modulus

        # Convert back to signed integer (handle values that might be negative)
        if secret > self.field_modulus // 2:
            secret -= self.field_modulus

        return int(secret)

    def reconstruct_array(self, shares: np.ndarray) -> np.ndarray:
        """
        Reconstruct array of secrets from shares.

        Args:
            shares: Array of shares (n_parties, n_values)

        Returns:
            Array of reconstructed secrets (n_values,)
        """
        n_values = shares.shape[1]
        secrets = np.zeros(n_values, dtype=np.int64)

        for i in range(n_values):
            secrets[i] = self.reconstruct(shares[:, i])

        return secrets

    def add_shared(self,
                   shares_a: np.ndarray,
                   shares_b: np.ndarray) -> np.ndarray:
        """
        Add two shared values without revealing them.

        If [a] and [b] are shares of a and b, returns shares of (a + b).

        Args:
            shares_a: Shares of value a
            shares_b: Shares of value b

        Returns:
            Shares of (a + b)
        """
        if len(shares_a) != len(shares_b):
            raise ValueError("Shares must have same number of parties")

        # Element-wise addition modulo field
        result = (shares_a + shares_b) % self.field_modulus
        return result

    def subtract_shared(self,
                       shares_a: np.ndarray,
                       shares_b: np.ndarray) -> np.ndarray:
        """
        Subtract two shared values without revealing them.

        If [a] and [b] are shares of a and b, returns shares of (a - b).

        Args:
            shares_a: Shares of value a
            shares_b: Shares of value b

        Returns:
            Shares of (a - b)
        """
        if len(shares_a) != len(shares_b):
            raise ValueError("Shares must have same number of parties")

        # Element-wise subtraction modulo field
        result = (shares_a - shares_b) % self.field_modulus
        return result

    def multiply_by_constant(self,
                            shares: np.ndarray,
                            constant: int) -> np.ndarray:
        """
        Multiply a shared value by a public constant.

        If [a] are shares of a, returns shares of (a * constant).

        Args:
            shares: Shares of value a
            constant: Public constant to multiply by

        Returns:
            Shares of (a * constant)
        """
        # Multiply each share by constant modulo field
        constant_mod = constant % self.field_modulus
        result = (shares * constant_mod) % self.field_modulus
        return result

    def share_float(self, value: float, n_parties: int, scale: int = 1000000) -> np.ndarray:
        """
        Share a floating-point value by scaling to integer.

        Args:
            value: Floating-point value to share
            n_parties: Number of parties
            scale: Scaling factor (default: 1,000,000 for 6 decimal places)

        Returns:
            Array of shares
        """
        scaled_value = int(value * scale)
        return self.share(scaled_value, n_parties)

    def reconstruct_float(self, shares: np.ndarray, scale: int = 1000000) -> float:
        """
        Reconstruct floating-point value from shares.

        Args:
            shares: Array of shares
            scale: Scaling factor used during sharing

        Returns:
            Reconstructed float value
        """
        scaled_value = self.reconstruct(shares)
        return scaled_value / scale


# Convenience functions for standalone use
def share(value: int, n_parties: int, field_modulus: Optional[int] = None) -> np.ndarray:
    """Share a value into n parties."""
    ss = SecretSharing(field_modulus)
    return ss.share(value, n_parties)


def reconstruct(shares: np.ndarray, field_modulus: Optional[int] = None) -> int:
    """Reconstruct a value from shares."""
    ss = SecretSharing(field_modulus)
    return ss.reconstruct(shares)


def add_shares(shares_a: np.ndarray,
               shares_b: np.ndarray,
               field_modulus: Optional[int] = None) -> np.ndarray:
    """Add two shared values."""
    ss = SecretSharing(field_modulus)
    return ss.add_shared(shares_a, shares_b)
