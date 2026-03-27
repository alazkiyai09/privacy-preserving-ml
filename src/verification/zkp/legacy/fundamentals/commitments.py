"""
Pedersen Commitment Scheme

A Pedersen commitment is a cryptographic primitive that allows you to commit to a value
while keeping it hidden, with the ability to reveal it later.

Properties:
- Binding: Cannot commit to one value and open as another
- Hiding: Commitment reveals no information about the value
- Perfectly hiding under the discrete log assumption

Mathematical Form:
C = g^v * h^r mod p
where:
- v is the value being committed to
- r is random blinding factor
- g, h are generators of the group
- p is the group order

Security Assumptions:
1. Discrete Logarithm Problem (DLP) is hard in the chosen group
2. g and h are randomly chosen independent generators
3. The blinding factor r is truly random and never reused
"""

import hashlib
import random
from typing import List, Tuple, Optional
import yaml


class PedersenCommitment:
    """
    Pedersen commitment scheme for hiding values with later revelation.

    SECURITY ASSUMPTIONS:
    - Discrete logarithm problem is hard in curve25519
    - Generators are chosen from a trusted setup
    - Randomness is never reused

    WARNINGS:
    - Randomness MUST be uniformly random and NEVER reused
    - Value + randomness pair reveals everything - protect the randomness!
    """

    # Curve25519 parameters (using simplified arithmetic for demonstration)
    # In production, use actual elliptic curve library (e.g., ecdsa, cryptography.io)
    CURVE_ORDER = 2**252 + 27742317777372353535851937790883648493

    # Default generators (in production, these should come from trusted setup)
    DEFAULT_G = 2  # Simplified: using multiplicative group modulo p
    DEFAULT_H = 3  # Different generator for binding property

    def __init__(
        self,
        group_order: int = None,
        generators: Optional[List[int]] = None,
        config_path: str = None,
    ):
        """
        Initialize Pedersen commitment scheme.

        Args:
            group_order: Order of the group (prime)
            generators: List of generators [g, h]
            config_path: Optional path to config file

        Security Note:
            Generators MUST be unpredictable. In production, obtain from
            multi-party computation (MPC) ceremony.
        """
        if config_path:
            self._load_config(config_path)
        else:
            self.group_order = group_order or self.CURVE_ORDER
            if generators:
                self.g, self.h = generators[0], generators[1]
            else:
                # WARNING: These are placeholder values!
                # Production: Use proper elliptic curve points
                self.g = self.DEFAULT_G
                self.h = self.DEFAULT_H

        # Verify g and h are in the group
        assert self.g < self.group_order, "Generator g not in group"
        assert self.h < self.group_order, "Generator h not in group"
        assert self.g != self.h, "Generators must be different for binding property"

        # Logarithm of g base h should be unknown (binding property)
        # In production, this is ensured by MPC ceremony

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.group_order = int(config["security"]["curve"]["order"])

        # Load generators from config if available
        if config["pedersen_commitment"]["generator_g"]:
            self.g = int(config["pedersen_commitment"]["generator_g"], 16)
        else:
            self.g = self.DEFAULT_G

        if config["pedersen_commitment"]["generator_h"]:
            self.h = int(config["pedersen_commitment"]["generator_h"], 16)
        else:
            self.h = self.DEFAULT_H

    def commit(self, value: int, randomness: int = None) -> Tuple[int, int]:
        """
        Create a commitment to a value.

        Mathematical Form:
        C = g^v * h^r mod p

        Args:
            value: The value to commit to (must be non-negative)
            randomness: Optional blinding factor (randomly generated if not provided)

        Returns:
            (commitment, randomness) - The commitment value and the randomness used

        Security Requirements:
            - Randomness MUST be uniformly random in [0, group_order)
            - Randomness MUST NEVER be reused for different commitments
            - Value must be in valid range

        Example:
            >>> scheme = PedersenCommitment()
            >>> commitment, r = scheme.commit(42)
            >>> # commitment is now binding to value 42
            >>> # To reveal: scheme.verify(commitment, 42, r)
        """
        if value < 0:
            raise ValueError("Value must be non-negative")

        if value >= self.group_order:
            raise ValueError(f"Value must be less than group order ({self.group_order})")

        # Generate randomness if not provided
        if randomness is None:
            # CRITICAL: Use cryptographically secure random number generator
            randomness = random.SystemRandom().randint(1, self.group_order - 1)

        if randomness <= 0 or randomness >= self.group_order:
            raise ValueError(f"Randomness must be in (0, {self.group_order})")

        # Compute commitment: C = g^v * h^r mod p
        # NOTE: This is simplified modular arithmetic
        # Production: Use actual elliptic curve point multiplication
        commitment = (pow(self.g, value, self.group_order) *
                     pow(self.h, randomness, self.group_order)) % self.group_order

        return commitment, randomness

    def verify(
        self,
        commitment: int,
        value: int,
        randomness: int
    ) -> bool:
        """
        Verify that a commitment opens to a specific value.

        Mathematical Form:
        Check: C == g^v * h^r mod p

        Args:
            commitment: The commitment value to verify
            value: The claimed value
            randomness: The randomness used to create the commitment

        Returns:
            True if the commitment is valid for this value/randomness pair

        Security Note:
            - This only proves the commitment is consistent
            - Does NOT prove the value is in a specific range
            - Use range proofs for that

        Example:
            >>> scheme = PedersenCommitment()
            >>> commitment, r = scheme.commit(42)
            >>> assert scheme.verify(commitment, 42, r)
            >>> assert not scheme.verify(commitment, 43, r)  # Wrong value
        """
        if value < 0 or value >= self.group_order:
            return False

        if randomness <= 0 or randomness >= self.group_order:
            return False

        # Recompute commitment and check if it matches
        expected = (pow(self.g, value, self.group_order) *
                   pow(self.h, randomness, self.group_order)) % self.group_order

        return commitment == expected

    def open(self, commitment: int, randomness: int) -> int:
        """
        Open a commitment to reveal the committed value.

        WARNING: This is a simplified implementation!
        In the actual Pedersen scheme, you cannot recover the value
        without knowing it beforehand (this is the hiding property).

        This method is provided for testing purposes where we store
        the value separately. In production, use verify() instead.

        Args:
            commitment: The commitment to open
            randomness: The randomness used to create the commitment

        Returns:
            The committed value (if available)

        Example:
            >>> scheme = PedersenCommitment()
            >>> commitment, r = scheme.commit(42)
            >>> # In real scenario, you must already know the value
            >>> # This is just for testing
            >>> assert scheme.verify(commitment, 42, r)
        """
        raise NotImplementedError(
            "Pedersen commitments are perfectly hiding - cannot recover value without knowing it. "
            "Use verify() with a known value instead."
        )

    def add_commitments(
        self,
        commitment1: int,
        commitment2: int
    ) -> int:
        """
        Add two commitments homomorphically.

        Property: C(v1, r1) * C(v2, r2) = C(v1+v2, r1+r2)

        Args:
            commitment1: First commitment
            commitment2: Second commitment

        Returns:
            A new commitment to the sum of values

        Use Case:
            In federated learning, clients can commit to individual gradients,
            and server can sum the commitments to get commitment to aggregated gradient.

        Example:
            >>> scheme = PedersenCommitment()
            >>> c1, r1 = scheme.commit(10)
            >>> c2, r2 = scheme.commit(20)
            >>> c_sum = scheme.add_commitments(c1, c2)
            >>> # c_sum is now a commitment to 30 with randomness r1+r2
        """
        return (commitment1 * commitment2) % self.group_order

    def scalar_multiply_commitment(
        self,
        commitment: int,
        scalar: int
    ) -> int:
        """
        Multiply a commitment by a scalar.

        Property: C(v, r)^k = C(k*v, k*r)

        Args:
            commitment: The commitment to multiply
            scalar: The scalar value

        Returns:
            A new commitment to the scaled value

        Use Case:
            In federated learning, scaling gradients or model updates.
        """
        return pow(commitment, scalar, self.group_order)

    def commit_vector(
        self,
        values: List[int],
        randomness: int = None
    ) -> Tuple[int, int]:
        """
        Commit to a vector of values using a single randomness.

        Uses homomorphic property:
        C(v, r) = product(C(v_i, r_i)) where sum(r_i) = r

        Args:
            values: List of values to commit to
            randomness: Optional blinding factor

        Returns:
            (commitment, randomness)

        Example:
            >>> scheme = PedersenCommitment()
            >>> commitment, r = scheme.commit_vector([1, 2, 3, 4, 5])
            >>> # commitment binds to the entire vector
        """
        if randomness is None:
            randomness = random.SystemRandom().randint(1, self.group_order - 1)

        # For simplicity, commit to hash of vector
        # In production, use proper vector commitment scheme
        value_hash = int(hashlib.sha256(str(values).encode()).hexdigest(), 16) % self.group_order

        return self.commit(value_hash, randomness)


class CommitmentEqualityProof:
    """
    Proof that two commitments commit to the same value.

    This is a simple sigma protocol proving:
        C1 = g^v * h^r1
        C2 = g^v * h^r2
    (same value v, different randomness)

    Uses Chaum-Pedersen protocol (variant of Schnorr).
    """

    def __init__(self, commitment_scheme: PedersenCommitment):
        self.scheme = commitment_scheme

    def generate_proof(
        self,
        value: int,
        randomness1: int,
        randomness2: int,
        commitment1: int,
        commitment2: int
    ) -> Tuple[int, int]:
        """
        Generate proof that commitment1 and commitment2 commit to same value.

        Args:
            value: The committed value
            randomness1: Randomness for first commitment
            randomness2: Randomness for second commitment
            commitment1: First commitment
            commitment2: Second commitment

        Returns:
            (challenge, response) - The proof
        """
        # Verify commitments are valid
        assert self.scheme.verify(commitment1, value, randomness1)
        assert self.scheme.verify(commitment2, value, randomness2)

        # Generate random witness
        w = random.SystemRandom().randint(1, self.scheme.group_order - 1)

        # Compute t1 = g^w * h^0 (for commitment1)
        # Compute t2 = g^w * h^0 (for commitment2)
        t1 = pow(self.scheme.g, w, self.scheme.group_order)
        t2 = t1  # Same since we're proving same value

        # Generate challenge (in real ZK, use Fiat-Shamir with hash)
        challenge = int(hashlib.sha256(f"{t1}{t2}{commitment1}{commitment2}".encode()).hexdigest(), 16)

        # Compute response
        response = (w - challenge * value) % (self.scheme.group_order - 1)

        return challenge, response

    def verify(
        self,
        commitment1: int,
        commitment2: int,
        proof: Tuple[int, int]
    ) -> bool:
        """
        Verify proof that commitments commit to same value.

        Args:
            commitment1: First commitment
            commitment2: Second commitment
            proof: (challenge, response)

        Returns:
            True if proof is valid
        """
        challenge, response = proof

        # Recompute t values
        t1 = (pow(self.scheme.g, response, self.scheme.group_order) *
              pow(commitment1, challenge, self.scheme.group_order)) % self.scheme.group_order

        t2 = (pow(self.scheme.g, response, self.scheme.group_order) *
              pow(commitment2, challenge, self.scheme.group_order)) % self.scheme.group_order

        # Verify challenge
        expected_challenge = int(hashlib.sha256(f"{t1}{t2}{commitment1}{commitment2}".encode()).hexdigest(), 16)

        return challenge == expected_challenge
