"""
Unit tests for Range proofs
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from fundamentals.range_proofs import RangeProof, BoundedVectorProof, ComparisonProof
from fundamentals.commitments import PedersenCommitment


class TestRangeProof:
    """Test suite for range proofs."""

    def test_initialization(self):
        """Test range proof initialization."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        assert proof_system.bit_length == 32
        assert proof_system.max_value == 2**32 - 1

    def test_bit_decomposition(self):
        """Test bit decomposition of values."""
        proof_system = RangeProof()

        # Test value 5 (binary: 101)
        bits = proof_system.decompose_bits(5)
        assert bits[:3] == [1, 0, 1]

        # Test value 0
        bits = proof_system.decompose_bits(0)
        assert all(b == 0 for b in bits)

        # Test value 255
        bits = proof_system.decompose_bits(255)
        assert bits[:8] == [1, 1, 1, 1, 1, 1, 1, 1]

    def test_bit_composition(self):
        """Test composition of bits to value."""
        proof_system = RangeProof()

        # Test 5 (binary: 101)
        value = proof_system.compose_bits([1, 0, 1])
        assert value == 5

        # Test 0
        value = proof_system.compose_bits([0, 0, 0])
        assert value == 0

        # Test 255
        value = proof_system.compose_bits([1] * 8)
        assert value == 255

    def test_bit_roundtrip(self):
        """Test that decompose and compose are inverses."""
        proof_system = RangeProof()

        test_values = [0, 1, 5, 42, 100, 255, 1000, 2**20]

        for value in test_values:
            bits = proof_system.decompose_bits(value)
            recovered = proof_system.compose_bits(bits)
            assert recovered == value

    def test_generate_proof_in_range(self):
        """Test generating proof for value in range."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 50
        commitment, randomness = scheme.commit(value)

        proof = proof_system.generate_proof(value, randomness, commitment, 0, 100)
        assert proof is not None

    def test_generate_proof_out_of_range(self):
        """Test that proof generation fails for value outside range."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 150
        commitment, randomness = scheme.commit(value)

        # Value is outside [0, 100]
        with pytest.raises(ValueError):
            proof_system.generate_proof(value, randomness, commitment, 0, 100)

    def test_verify_valid_proof(self):
        """Test verifying a valid range proof."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 50
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, 0, 100)

        assert proof_system.verify(commitment, 0, 100, proof)

    def test_verify_wrong_range(self):
        """Test that verification fails with wrong range."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 50
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, 0, 100)

        # Try to verify with different range
        assert not proof_system.verify(commitment, 0, 50, proof)

    def test_generate_bulletproof_style(self):
        """Test generating Bulletproof-style range proof."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 50
        commitment, randomness = scheme.commit(value)

        proof = proof_system.generate_bulletproof_style(
            value, randomness, commitment, 0, 100
        )

        assert 'bit_commitments' in proof
        assert 'min_val' in proof
        assert 'max_val' in proof

    def test_verify_bulletproof_style(self):
        """Test verifying Bulletproof-style range proof."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

        value = 50
        commitment, randomness = scheme.commit(value)

        proof = proof_system.generate_bulletproof_style(
            value, randomness, commitment, 0, 100
        )

        assert proof_system.verify_bulletproof_style(commitment, 0, 100, proof)

    def test_boundary_values(self):
        """Test range proofs with boundary values."""
        scheme = PedersenCommitment()
        proof_system = RangeProof(bit_length=8, commitment_scheme=scheme)

        # Test minimum value
        value = 0
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, 0, 255)
        assert proof_system.verify(commitment, 0, 255, proof)

        # Test maximum value
        value = 255
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, 0, 255)
        assert proof_system.verify(commitment, 0, 255, proof)


class TestBoundedVectorProof:
    """Test suite for bounded vector proofs (L2 norm)."""

    def test_initialization(self):
        """Test bounded vector proof initialization."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

        assert proof_system.bound == 1.0
        assert proof_system.bound_squared == 1.0

    def test_generate_proof_valid_vector(self):
        """Test generating proof for vector with valid norm."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

        vector = [0.5, 0.5, 0.5]  # Norm = sqrt(0.75) < 1.0

        proof = proof_system.generate_proof(vector)

        assert 'commitment' in proof
        assert 'norm_squared' in proof
        assert 'bound_squared' in proof

    def test_generate_proof_invalid_vector(self):
        """Test that proof generation fails for vector exceeding bound."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

        vector = [2.0, 2.0, 2.0]  # Norm > 1.0

        with pytest.raises(ValueError):
            proof_system.generate_proof(vector)

    def test_verify_valid_proof(self):
        """Test verifying bounded vector proof."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

        vector = [0.5, 0.5, 0.5]
        proof = proof_system.generate_proof(vector)

        assert proof_system.verify(proof)

    def test_verify_tampered_proof(self):
        """Test that verification fails with tampered proof."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

        vector = [0.5, 0.5, 0.5]
        proof = proof_system.generate_proof(vector)

        # Tamper with the proof
        proof['norm_squared'] = proof['bound_squared'] + 1

        assert not proof_system.verify(proof)

    def test_high_dimensional_vector(self):
        """Test bounded vector proof with high-dimensional vector."""
        scheme = PedersenCommitment()
        proof_system = BoundedVectorProof(bound=10.0, commitment_scheme=scheme)

        # 100-dimensional vector with small values
        vector = [0.1] * 100

        proof = proof_system.generate_proof(vector)
        assert proof_system.verify(proof)
        assert proof['dimension'] == 100


class TestComparisonProof:
    """Test suite for comparison proofs."""

    def test_initialization(self):
        """Test comparison proof initialization."""
        scheme = PedersenCommitment()
        proof_system = ComparisonProof(commitment_scheme=scheme)

        assert proof_system.commitment_scheme is not None

    def test_generate_less_than_proof(self):
        """Test generating proof that a < b."""
        scheme = PedersenCommitment()
        proof_system = ComparisonProof(commitment_scheme=scheme)

        value_a = 50
        value_b = 100

        commitment_a, r_a = scheme.commit(value_a)
        commitment_b, r_b = scheme.commit(value_b)

        proof = proof_system.generate_less_than_proof(
            value_a, r_a, commitment_a, value_b, r_b, commitment_b
        )

        assert 'diff_commitment' in proof
        assert proof is not None

    def test_generate_equal_values_fails(self):
        """Test that proof generation fails for equal values."""
        scheme = PedersenCommitment()
        proof_system = ComparisonProof(commitment_scheme=scheme)

        value = 50

        commitment_a, r_a = scheme.commit(value)
        commitment_b, r_b = scheme.commit(value)

        # a < b should fail when a == b
        with pytest.raises(ValueError):
            proof_system.generate_less_than_proof(
                value, r_a, commitment_a, value, r_b, commitment_b
            )

    def test_generate_greater_values_fails(self):
        """Test that proof generation fails when a > b."""
        scheme = PedersenCommitment()
        proof_system = ComparisonProof(commitment_scheme=scheme)

        value_a = 100
        value_b = 50

        commitment_a, r_a = scheme.commit(value_a)
        commitment_b, r_b = scheme.commit(value_b)

        with pytest.raises(ValueError):
            proof_system.generate_less_than_proof(
                value_a, r_a, commitment_a, value_b, r_b, commitment_b
            )


def test_range_proof_comprehensive():
    """Comprehensive test of range proof functionality."""
    scheme = PedersenCommitment()
    proof_system = RangeProof(bit_length=16, commitment_scheme=scheme)

    test_cases = [
        (0, 0, 1000),
        (500, 0, 1000),
        (999, 0, 1000),
        (100, 50, 150),
    ]

    for value, min_val, max_val in test_cases:
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, min_val, max_val)
        assert proof_system.verify(commitment, min_val, max_val, proof)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
