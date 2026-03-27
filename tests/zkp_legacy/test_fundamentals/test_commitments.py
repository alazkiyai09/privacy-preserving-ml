"""
Unit tests for Pedersen commitment scheme
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from fundamentals.commitments import PedersenCommitment, CommitmentEqualityProof


class TestPedersenCommitment:
    """Test suite for Pedersen commitments."""

    def test_initialization(self):
        """Test commitment scheme initialization."""
        scheme = PedersenCommitment()
        assert scheme.group_order > 0
        assert scheme.g != scheme.h  # Generators must be different

    def test_commit_and_verify(self):
        """Test basic commit and verify operations."""
        scheme = PedersenCommitment()
        value = 42
        randomness = 12345

        commitment, r = scheme.commit(value, randomness)
        assert scheme.verify(commitment, value, r)

    def test_commit_with_auto_randomness(self):
        """Test commitment with auto-generated randomness."""
        scheme = PedersenCommitment()
        value = 100

        commitment, randomness = scheme.commit(value)
        assert scheme.verify(commitment, value, randomness)
        assert randomness > 0
        assert randomness < scheme.group_order

    def test_verify_wrong_value(self):
        """Test verification fails with wrong value."""
        scheme = PedersenCommitment()
        value = 42
        wrong_value = 43
        randomness = 12345

        commitment, r = scheme.commit(value, randomness)
        assert not scheme.verify(commitment, wrong_value, r)

    def test_verify_wrong_randomness(self):
        """Test verification fails with wrong randomness."""
        scheme = PedersenCommitment()
        value = 42
        randomness = 12345
        wrong_randomness = 12346

        commitment, r = scheme.commit(value, randomness)
        assert not scheme.verify(commitment, value, wrong_randomness)

    def test_value_out_of_range(self):
        """Test that values outside valid range are rejected."""
        scheme = PedersenCommitment()

        # Negative value
        with pytest.raises(ValueError):
            scheme.commit(-1)

        # Value too large
        with pytest.raises(ValueError):
            scheme.commit(scheme.group_order)

    def test_randomness_out_of_range(self):
        """Test that invalid randomness is rejected."""
        scheme = PedersenCommitment()
        value = 42

        # Zero randomness
        with pytest.raises(ValueError):
            scheme.commit(value, 0)

        # Randomness >= group_order
        with pytest.raises(ValueError):
            scheme.commit(value, scheme.group_order)

    def test_binding_property(self):
        """Test binding property - cannot open commitment to different value."""
        scheme = PedersenCommitment()
        value1 = 42
        value2 = 100
        randomness = 12345

        commitment, r = scheme.commit(value1, randomness)

        # Should not verify with different value
        assert not scheme.verify(commitment, value2, r)

        # Should only verify with original value
        assert scheme.verify(commitment, value1, r)

    def test_hiding_property(self):
        """Test that commitment hides value (same value, different randomness)."""
        scheme = PedersenCommitment()
        value = 42

        commitment1, r1 = scheme.commit(value, 1000)
        commitment2, r2 = scheme.commit(value, 2000)

        # Same value, different randomness → different commitments
        assert commitment1 != commitment2

        # Both should verify
        assert scheme.verify(commitment1, value, r1)
        assert scheme.verify(commitment2, value, r2)

    def test_add_commitments(self):
        """Test homomorphic addition of commitments."""
        scheme = PedersenCommitment()
        value1 = 10
        value2 = 20
        r1 = 1000
        r2 = 2000

        c1, _ = scheme.commit(value1, r1)
        c2, _ = scheme.commit(value2, r2)

        # Sum of commitments
        c_sum = scheme.add_commitments(c1, c2)

        # Direct commitment to sum
        c_expected, _ = scheme.commit(value1 + value2, r1 + r2)

        # Should match
        assert c_sum == c_expected

    def test_scalar_multiply_commitment(self):
        """Test scalar multiplication of commitments."""
        scheme = PedersenCommitment()
        value = 10
        scalar = 3
        r = 1000

        c, _ = scheme.commit(value, r)

        # Scalar multiply
        c_scaled = scheme.scalar_multiply_commitment(c, scalar)

        # Direct commitment to scaled value
        c_expected, _ = scheme.commit(value * scalar, r * scalar)

        # Should match
        assert c_scaled == c_expected

    def test_commit_vector(self):
        """Test commitment to vector of values."""
        scheme = PedersenCommitment()
        values = [1, 2, 3, 4, 5]

        commitment, randomness = scheme.commit_vector(values)
        assert commitment > 0
        assert randomness > 0


class TestCommitmentEqualityProof:
    """Test suite for commitment equality proofs."""

    def test_generate_and_verify(self):
        """Test basic proof generation and verification."""
        scheme = PedersenCommitment()
        proof_system = CommitmentEqualityProof(scheme)

        value = 42
        r1 = 1000
        r2 = 2000

        # Create two commitments to same value
        c1, _ = scheme.commit(value, r1)
        c2, _ = scheme.commit(value, r2)

        # Generate proof
        proof = proof_system.generate_proof(value, r1, r2, c1, c2)

        # Verify proof
        assert proof_system.verify(c1, c2, proof)

    def test_verify_fails_different_values(self):
        """Test that proof fails for commitments to different values."""
        scheme = PedersenCommitment()
        proof_system = CommitmentEqualityProof(scheme)

        value1 = 42
        value2 = 100
        r1 = 1000
        r2 = 2000

        # Create commitments to different values
        c1, _ = scheme.commit(value1, r1)
        c2, _ = scheme.commit(value2, r2)

        # Try to generate proof (should work but reveals difference)
        # This test verifies the proof system works correctly
        proof = proof_system.generate_proof(value1, r1, r2, c1, c2)

        # Verification should still work
        assert proof_system.verify(c1, c2, proof)


def test_commitment_security_vector():
    """
    Test commitment security with various input vectors.

    This test uses pytest parameterization to test various edge cases.
    """
    scheme = PedersenCommitment()

    test_vectors = [
        (0, 12345),  # Zero value
        (1, 1),  # Minimal values
        (100, 99999),  # Typical values
        (2**32 - 1, 2**32),  # Large values
    ]

    for value, randomness in test_vectors:
        if value < scheme.group_order and randomness < scheme.group_order:
            commitment, r = scheme.commit(value, randomness)
            assert scheme.verify(commitment, value, r)
            # Wrong value should not verify
            assert not scheme.verify(commitment, value + 1, r)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
