#!/usr/bin/env python3
"""
Range Proof Demonstration

Demonstrates proving a value is within a range without revealing it.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from fundamentals.range_proofs import RangeProof
from fundamentals.commitments import PedersenCommitment


def main():
    print("=" * 70)
    print("RANGE PROOF DEMONSTRATION")
    print("=" * 70)
    print()

    # Setup
    print("1. Setting up range proof system...")
    scheme = PedersenCommitment()
    proof_system = RangeProof(bit_length=32, commitment_scheme=scheme)

    min_val = 0
    max_val = 100

    print(f"   Range: [{min_val}, {max_val}]")
    print()

    # Example 1: Value in range
    print("2. Example 1: Value in range")
    value = 50
    commitment, randomness = scheme.commit(value)

    print(f"   Value: {value}")
    print(f"   Commitment: {commitment}")

    proof = proof_system.generate_proof(value, randomness, commitment, min_val, max_val)
    print(f"   Proof generated: {len(proof)} bytes")

    is_valid = proof_system.verify(commitment, min_val, max_val, proof)
    print(f"   Verification: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Example 2: Value out of range
    print("3. Example 2: Value out of range")
    value = 150
    commitment, randomness = scheme.commit(value)

    print(f"   Value: {value}")
    print(f"   Commitment: {commitment}")

    try:
        proof = proof_system.generate_proof(value, randomness, commitment, min_val, max_val)
        print(f"   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    print()

    # Example 3: Boundary values
    print("4. Example 3: Boundary values")
    for value in [0, 100]:
        commitment, randomness = scheme.commit(value)
        proof = proof_system.generate_proof(value, randomness, commitment, 0, 100)
        is_valid = proof_system.verify(commitment, 0, 100, proof)
        print(f"   Value {value}: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Example 4: Bounded vector
    print("5. Example 4: Bounded vector (gradient norm)")
    from fundamentals.range_proofs import BoundedVectorProof

    vector_system = BoundedVectorProof(bound=1.0, commitment_scheme=scheme)

    # Valid vector
    valid_vector = [0.5, 0.5, 0.5]
    print(f"   Vector: {valid_vector}")
    print(f"   L2 norm: {(sum(x**2 for x in valid_vector))**0.5:.4f}")

    proof = vector_system.generate_proof(valid_vector)
    is_valid = vector_system.verify(proof)
    print(f"   Verification: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Invalid vector
    invalid_vector = [2.0, 2.0, 2.0]
    print(f"   Vector: {invalid_vector}")
    print(f"   L2 norm: {(sum(x**2 for x in invalid_vector))**0.5:.4f}")

    try:
        proof = vector_system.generate_proof(invalid_vector)
        print(f"   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
