#!/usr/bin/env python3
"""
Simple Pedersen Commitment Example

Demonstrates basic commitment scheme usage.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from fundamentals.commitments import PedersenCommitment


def main():
    print("=" * 70)
    print("PEDERSEN COMMITMENT EXAMPLE")
    print("=" * 70)
    print()

    # Initialize commitment scheme
    print("1. Initializing Pedersen commitment scheme...")
    scheme = PedersenCommitment()
    print(f"   Group order: {scheme.group_order}")
    print(f"   Generator g: {scheme.g}")
    print(f"   Generator h: {scheme.h}")
    print()

    # Commit to a value
    print("2. Committing to secret value...")
    secret_value = 42
    randomness = 12345

    commitment, r = scheme.commit(secret_value, randomness)

    print(f"   Secret value: {secret_value}")
    print(f"   Randomness: {randomness}")
    print(f"   Commitment: {commitment}")
    print()

    # Verify commitment
    print("3. Verifying commitment...")
    is_valid = scheme.verify(commitment, secret_value, r)
    print(f"   Verification result: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Try wrong value
    print("4. Trying to open with wrong value...")
    wrong_value = 43
    is_valid = scheme.verify(commitment, wrong_value, r)
    print(f"   Try to open as value {wrong_value}: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Homomorphic property
    print("5. Demonstrating homomorphic property...")
    value1 = 10
    value2 = 20
    r1 = 1000
    r2 = 2000

    c1, _ = scheme.commit(value1, r1)
    c2, _ = scheme.commit(value2, r2)

    c_sum = scheme.add_commitments(c1, c2)
    c_expected, _ = scheme.commit(value1 + value2, r1 + r2)

    print(f"   C1 = commit({value1}, {r1}) = {c1}")
    print(f"   C2 = commit({value2}, {r2}) = {c2}")
    print(f"   C1 + C2 = {c_sum}")
    print(f"   commit({value1 + value2}, {r1 + r2}) = {c_expected}")
    print(f"   Match: {c_sum == c_expected} ✓")
    print()

    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
