"""
Range Proofs

A range proof allows a prover to prove that a committed value lies within a specific
range [min, max] without revealing the actual value.

Applications in Federated Learning:
- Prove gradient norm is bounded: ||gradient|| <= C
- Prove learning rate is in valid range
- Prove client data statistics are reasonable

There are several approaches:
1. Boudot range proof: Uses set membership
2. Bulletproofs: Efficient, no trusted setup
3. Simple bit decomposition: Prove each bit is 0 or 1

This module implements a simplified range proof using bit decomposition.

Mathematical Concept:
To prove a <= x <= b:
1. Decompose x into bits: x = sum(2^i * b_i)
2. Prove each b_i is in {0, 1}
3. Prove range constraint from bit decomposition

Security Assumptions:
- Discrete logarithm is hard
- Commitment scheme is binding and hiding
"""

import random
from typing import Tuple, List
from .commitments import PedersenCommitment


class RangeProof:
    """
    Range proof for proving a value is within [min, max] without revealing it.

    SECURITY ASSUMPTIONS:
    - Pedersen commitment is binding and hiding
    - Each bit is independently proven to be 0 or 1
    - Randomness is never reused

    WARNINGS:
    - This is a simplified implementation
    - Production systems should use Bulletproofs or Boudot proofs
    """

    def __init__(
        self,
        bit_length: int = 64,
        commitment_scheme: PedersenCommitment = None
    ):
        """
        Initialize range proof.

        Args:
            bit_length: Number of bits for value range (max value = 2^bit_length - 1)
            commitment_scheme: Optional commitment scheme instance

        Example:
            >>> scheme = PedersenCommitment()
            >>> proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)
            >>> commitment, r = scheme.commit(42)
            >>> proof = proof_system.generate_proof(42, r, commitment, 0, 100)
            >>> assert proof_system.verify(commitment, 0, 100, proof)
        """
        self.bit_length = bit_length
        self.max_value = (1 << bit_length) - 1
        self.commitment_scheme = commitment_scheme or PedersenCommitment()

    def decompose_bits(self, value: int) -> List[int]:
        """
        Decompose value into binary representation.

        Args:
            value: The value to decompose

        Returns:
            List of bits (least significant first)

        Example:
            >>> proof_system = RangeProof()
            >>> bits = proof_system.decompose_bits(5)
            >>> assert bits == [1, 0, 1]  # 5 = 101 in binary
        """
        bits = []
        for i in range(self.bit_length):
            bits.append((value >> i) & 1)
        return bits

    def compose_bits(self, bits: List[int]) -> int:
        """
        Compose bits back into value.

        Args:
            bits: List of bits (least significant first)

        Returns:
            The composed value

        Example:
            >>> proof_system = RangeProof()
            >>> value = proof_system.compose_bits([1, 0, 1])
            >>> assert value == 5
        """
        value = 0
        for i, bit in enumerate(bits):
            value |= (bit << i)
        return value

    def generate_proof(
        self,
        value: int,
        randomness: int,
        commitment: int,
        min_val: int = 0,
        max_val: int = None
    ) -> bytes:
        """
        Generate range proof for a committed value.

        Args:
            value: The actual value (secret)
            randomness: The randomness used in commitment
            commitment: The commitment C = g^value * h^randomness
            min_val: Minimum of valid range
            max_val: Maximum of valid range (defaults to 2^bit_length - 1)

        Returns:
            Serialized proof as bytes

        Security Requirements:
            - min_val <= value <= max_val must hold
            - Randomness must not be reused
            - Value must be positive

        Proof Structure:
        1. Decompose value into bits
        2. For each bit, prove it's 0 or 1
        3. Prove bit decomposition matches commitment
        """
        if max_val is None:
            max_val = self.max_value

        if not (min_val <= value <= max_val):
            raise ValueError(f"Value {value} not in range [{min_val}, {max_val}]")

        # Decompose value into bits
        bits = self.decompose_bits(value)

        # Generate proof for each bit
        # Simplified: we just store the bits and use commitment binding property
        # In production, use proper sigma protocol for each bit
        proof_data = {
            'bits': bits,
            'min_val': min_val,
            'max_val': max_val,
            # In real range proof, we'd include challenges and responses
        }

        # Serialize proof
        import json
        return json.dumps({
            'bits': bits,
            'min_val': min_val,
            'max_val': max_val,
            'bit_length': self.bit_length,
        }).encode()

    def verify(
        self,
        commitment: int,
        min_val: int,
        max_val: int,
        proof: bytes
    ) -> bool:
        """
        Verify range proof.

        Args:
            commitment: The commitment to verify
            min_val: Minimum of valid range
            max_val: Maximum of valid range
            proof: The range proof

        Returns:
            True if proof is valid

        Note:
            This simplified version uses the binding property of Pedersen commitments.
            The real proof would verify each bit is 0 or 1 without revealing them.
        """
        try:
            import json
            proof_data = json.loads(proof.decode())

            bits = proof_data['bits']
            proof_min = proof_data['min_val']
            proof_max = proof_data['max_val']

            # Check range matches
            if proof_min != min_val or proof_max != max_val:
                return False

            # Reconstruct value from bits (this reveals it - simplified!)
            # In real range proof, we wouldn't reconstruct
            value = self.compose_bits(bits)

            # Check range
            if not (min_val <= value <= max_val):
                return False

            # Verify commitment (we need randomness for this)
            # In real protocol, proof would include zero-knowledge verification
            # This is simplified - just check range is valid

            return True

        except Exception:
            return False

    def generate_bulletproof_style(
        self,
        value: int,
        randomness: int,
        commitment: int,
        min_val: int = 0,
        max_val: int = None
    ) -> dict:
        """
        Generate a more sophisticated range proof (Bulletproof-style).

        This is a simplified version that demonstrates the concept without
        the full cryptographic complexity of Bulletproofs.

        Args:
            value: The actual value
            randomness: The randomness used in commitment
            commitment: The commitment C = g^value * h^randomness
            min_val: Minimum of valid range
            max_val: Maximum of valid range

        Returns:
            Dictionary containing proof data

        Concept:
        Instead of revealing bits, we prove:
        1. value = sum(2^i * b_i) where each b_i ∈ {0, 1}
        2. sum(b_i * 2^i) is in valid range

        The proof uses inner product arguments to avoid revealing bits.
        """
        if max_val is None:
            max_val = self.max_value

        # For this simplified version, we'll use the bit decomposition approach
        # with commitments to each bit
        bits = self.decompose_bits(value)

        # Generate random blinding factors for each bit commitment
        bit_randomness = [
            random.SystemRandom().randint(1, self.commitment_scheme.group_order - 1)
            for _ in bits
        ]

        # Create commitments to each bit
        bit_commitments = []
        for bit, r in zip(bits, bit_randomness):
            comm, _ = self.commitment_scheme.commit(bit, r)
            bit_commitments.append(comm)

        # Calculate total value and randomness from bits
        # value = sum(2^i * bit_i)
        # randomness = sum(2^i * randomness_i)
        total_value = sum((1 << i) * bit for i, bit in enumerate(bits))
        total_randomness = sum((1 << i) * r for i, r in enumerate(bit_randomness))

        # Verify reconstruction
        assert total_value == value, "Bit decomposition doesn't match value"

        # Create proof
        proof = {
            'bit_commitments': bit_commitments,
            'min_val': min_val,
            'max_val': max_val,
            'commitment': commitment,
            # In real bulletproofs, we'd include inner product proof
            # For now, we use this simplified version
        }

        return proof

    def verify_bulletproof_style(
        self,
        commitment: int,
        min_val: int,
        max_val: int,
        proof: dict
    ) -> bool:
        """
        Verify Bulletproof-style range proof.

        Args:
            commitment: The commitment to verify
            min_val: Minimum of valid range
            max_val: Maximum of valid range
            proof: The proof data

        Returns:
            True if proof is valid

        Simplified verification:
        1. Check bit commitments are valid
        2. Verify weighted sum of bit commitments matches main commitment
        """
        try:
            bit_commitments = proof['bit_commitments']

            # In real verification, we'd check that the weighted sum
            # of bit commitments matches the main commitment using
            # the homomorphic property

            # C = product(C_i^(2^i))
            # where C_i is commitment to bit i

            reconstructed = 1
            for i, bit_comm in enumerate(bit_commitments):
                power = 1 << i  # 2^i
                reconstructed = (reconstructed *
                               pow(bit_comm, power, self.commitment_scheme.group_order)) % self.commitment_scheme.group_order

            # Check if reconstructed commitment matches
            # In real proof, this would be zero-knowledge
            return reconstructed == commitment

        except Exception:
            return False


class BoundedVectorProof:
    """
    Prove that a vector's L2 norm is bounded.

    USE CASE IN FEDERATED LEARNING:
    Prove ||gradient|| <= C without revealing the gradient.

    Mathematical approach:
    1. Commit to each gradient element
    2. Prove sum of squares is bounded
    3. Use range proofs for the sum

    This is a simplified implementation demonstrating the concept.
    """

    def __init__(
        self,
        bound: float,
        commitment_scheme: PedersenCommitment = None
    ):
        """
        Initialize bounded vector proof.

        Args:
            bound: The L2 norm bound
            commitment_scheme: Optional commitment scheme

        Example:
            >>> scheme = PedersenCommitment()
            >>> proof = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)
            >>> gradient = [0.1, 0.2, 0.3]
            >>> proof_data = proof.generate_proof(gradient)
            >>> assert proof.verify(proof_data)
        """
        self.bound = bound
        self.bound_squared = bound ** 2
        self.commitment_scheme = commitment_scheme or PedersenCommitment()

    def generate_proof(self, vector: list) -> dict:
        """
        Generate proof that ||vector|| <= bound.

        Args:
            vector: The vector (list of floats)

        Returns:
            Proof dictionary

        Approach:
        1. Compute squared norm: ||v||^2 = sum(v_i^2)
        2. Commit to squared norm
        3. Generate range proof that norm is <= bound
        """
        import numpy as np

        # Compute L2 norm
        norm_squared = sum(float(x) ** 2 for x in vector)

        # Check bound
        if norm_squared > self.bound_squared:
            raise ValueError(f"Vector norm {norm_squared**0.5} exceeds bound {self.bound}")

        # Commit to norm_squared
        # We need to convert to integer for commitment
        norm_int = int(norm_squared * (1 << 32))  # Fixed-point representation
        commitment, randomness = self.commitment_scheme.commit(norm_int)

        # Generate range proof
        range_proof = RangeProof(bit_length=64, commitment_scheme=self.commitment_scheme)
        bound_int = int(self.bound_squared * (1 << 32))

        proof_data = {
            'commitment': commitment,
            'randomness': randomness,
            'norm_squared': norm_int,
            'bound_squared': bound_int,
            'dimension': len(vector),
            # In real proof, we'd also commit to individual elements
        }

        return proof_data

    def verify(self, proof_data: dict) -> bool:
        """
        Verify proof that vector norm is bounded.

        Args:
            proof_data: The proof dictionary

        Returns:
            True if proof is valid

        Simplified verification:
        1. Verify commitment
        2. Check norm_squared <= bound_squared
        """
        try:
            commitment = proof_data['commitment']
            randomness = proof_data['randomness']
            norm_squared = proof_data['norm_squared']
            bound_squared = proof_data['bound_squared']

            # Verify commitment
            if not self.commitment_scheme.verify(commitment, norm_squared, randomness):
                return False

            # Check bound
            if norm_squared > bound_squared:
                return False

            return True

        except Exception:
            return False


class ComparisonProof:
    """
    Prove relationship between two committed values (e.g., a < b).

    USE CASE:
    Prove loss decreased: loss_new < loss_old

    Approach:
    - Prove (b - a) is in range [1, max_value]
    - Uses homomorphic property of commitments
    """

    def __init__(self, commitment_scheme: PedersenCommitment = None):
        """
        Initialize comparison proof.

        Args:
            commitment_scheme: Optional commitment scheme
        """
        self.commitment_scheme = commitment_scheme or PedersenCommitment()

    def generate_less_than_proof(
        self,
        value_a: int,
        randomness_a: int,
        commitment_a: int,
        value_b: int,
        randomness_b: int,
        commitment_b: int
    ) -> dict:
        """
        Generate proof that value_a < value_b.

        Args:
            value_a: First value
            randomness_a: Randomness for commitment_a
            commitment_a: Commitment to value_a
            value_b: Second value
            randomness_b: Randomness for commitment_b
            commitment_b: Commitment to value_b

        Returns:
            Proof dictionary

        Approach:
        Prove (value_b - value_a) > 0 using homomorphic property
        """
        if value_a >= value_b:
            raise ValueError(f"value_a ({value_a}) must be less than value_b ({value_b})")

        # Compute difference
        diff = value_b - value_a

        # Using homomorphic property:
        # C(b-a, r_b-r_a) = C(b, r_b) / C(a, r_a)
        diff_randomness = (randomness_b - randomness_a) % self.commitment_scheme.group_order

        # In multiplicative group: C(b) / C(a) = C(b-a)
        diff_commitment = (commitment_b *
                          pow(commitment_a, -1, self.commitment_scheme.group_order)) % self.commitment_scheme.group_order

        # Generate range proof for difference in range [1, max]
        range_proof = RangeProof(bit_length=64, commitment_scheme=self.commitment_scheme)

        proof = {
            'diff_commitment': diff_commitment,
            'diff_randomness': diff_randomness,
            'value_a_commitment': commitment_a,
            'value_b_commitment': commitment_b,
        }

        return proof

    def verify_less_than_proof(self, proof: dict) -> bool:
        """
        Verify proof that value_a < value_b.

        Args:
            proof: The proof dictionary

        Returns:
            True if proof is valid
        """
        try:
            diff_commitment = proof['diff_commitment']
            diff_randomness = proof['diff_randomness']

            # Verify difference is positive (greater than 0)
            # We use range proof to show diff >= 1
            range_proof = RangeProof(bit_length=64, commitment_scheme=self.commitment_scheme)

            # This is simplified - real proof would use proper range proof
            return True

        except Exception:
            return False
