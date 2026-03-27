#!/usr/bin/env python3
"""
Simple Enclave Example
======================

Demonstrates basic TEE functionality:
- Creating an enclave
- Entering with data
- Executing operations
- Remote attestation
- Sealed storage
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import create_enclave
from tee_ml.core.attestation import AttestationService, simulate_remote_attestation
from tee_ml.core.sealed_storage import create_sealed_storage


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def main():
    """Run simple enclave demonstration."""

    print_section("TEE ML - Simple Enclave Example")

    # ========================================================================
    # 1. Create Enclave
    # ========================================================================
    print("1. Creating Enclave")
    print("-" * 70)

    enclave = create_enclave(enclave_id="demo-enclave", memory_limit_mb=128)

    print(f"✓ Enclave created: {enclave.enclave_id}")
    print(f"  Memory limit: {enclave.memory_limit_bytes / (1024**2):.0f} MB")
    print(f"  Measurement: {enclave.get_measurement().hex()[:32]}...")
    print(f"  Is isolated: {enclave.is_isolated()}")

    # ========================================================================
    # 2. Enter Enclave with Data
    # ========================================================================
    print_section("2. Enter Enclave with Data")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Input data: {data}")

    session = enclave.enter(data)
    print(f"✓ Entered enclave")
    print(f"  Session ID: {session.session_id}")
    print(f"  Memory usage: {session.get_memory_usage():.4f} MB")

    # ========================================================================
    # 3. Execute Operations in Enclave
    # ========================================================================
    print_section("3. Execute Operations in Enclave")

    # Operation 1: Multiply by 2
    print("Operation 1: Multiply by 2")
    result1 = session.execute(lambda x: x * 2)
    print(f"  Result: {result1}")

    # Operation 2: Add 10
    print("\nOperation 2: Add 10")
    result2 = session.execute(lambda x: x + 10)
    print(f"  Result: {result2}")

    # Operation 3: Custom function
    print("\nOperation 3: Custom (x² + 3x + 1)")
    def custom_func(x):
        return x**2 + 3*x + 1

    result3 = session.execute(custom_func)
    print(f"  Result: {result3}")

    # ========================================================================
    # 4. Exit Enclave
    # ========================================================================
    print_section("4. Exit Enclave")

    final_result = enclave.exit(session)
    print(f"✓ Exited enclave")
    print(f"  Final result: {final_result}")
    print(f"  Session duration: {session.get_session_duration_ns() / 1000:.1f} μs")

    # Show statistics
    stats = enclave.get_statistics()
    print(f"\nEnclave Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total exits: {stats['total_exits']}")
    print(f"  Memory used: {stats['memory_used_mb']:.4f} MB")
    print(f"  Memory utilization: {stats['memory_utilization']*100:.1f}%")

    # ========================================================================
    # 5. Remote Attestation
    # ========================================================================
    print_section("5. Remote Attestation")

    print("Simulating remote attestation...")
    attestation_result = simulate_remote_attestation(
        enclave=enclave,
        challenger_id="remote_client"
    )

    print(f"✓ Attestation completed")
    print(f"  Challenger: {attestation_result['challenger_id']}")
    print(f"  Enclave ID: {attestation_result['enclave_id']}")
    print(f"  Nonce: {attestation_result['nonce'][:32]}...")
    print(f"  Valid: {attestation_result['result']['valid']}")
    print(f"  Reason: {attestation_result['result']['reason']}")

    # ========================================================================
    # 6. Sealed Storage
    # ========================================================================
    print_section("6. Sealed Storage")

    # Create storage
    storage = create_sealed_storage(storage_path="/tmp/tee_demo_storage")

    # Seal some data
    secret_data = b"This is a secret model weight or sensitive parameter"
    print(f"Original data: {secret_data.decode()}")

    storage.save_sealed(
        key="demo_secret",
        data=secret_data,
        enclave_id=enclave.enclave_id,
        measurement=enclave.get_measurement()
    )
    print("✓ Data sealed and saved")

    # List sealed keys
    keys = storage.list_sealed_keys()
    print(f"  Sealed keys: {keys}")

    # Unseal data
    unsealed = storage.load_sealed(
        key="demo_secret",
        enclave_id=enclave.enclave_id,
        measurement=enclave.get_measurement()
    )
    print(f"\n✓ Data unsealed")
    print(f"  Unsealed data: {unsealed.decode()}")

    # Verify integrity
    assert unsealed == secret_data, "Data integrity check failed!"
    print(f"  Integrity: ✓ Verified")

    # ========================================================================
    # 7. Overhead Demonstration
    # ========================================================================
    print_section("7. Overhead Demonstration")

    from tee_ml.simulation.overhead import OverheadModel, compare_tee_vs_he

    model = OverheadModel()

    # Estimate overhead for 5-layer inference
    print("Estimating overhead for 5-layer neural network:")
    overhead = model.calculate_overhead(
        operation_time_ns=10000,  # 10 μs computation
        data_size_mb=1.0,
        num_entries=5,
        num_exits=5,
    )

    print(f"  Computation: {overhead['computation_ns']:.0f} ns")
    print(f"  Entry overhead: {overhead['entry_overhead_ns']:.0f} ns")
    print(f"  Exit overhead: {overhead['exit_overhead_ns']:.0f} ns")
    print(f"  Memory encryption: {overhead['memory_encryption_ns']:.0f} ns")
    print(f"  Total overhead: {overhead['total_overhead_ns']:.0f} ns")
    print(f"  Slowdown factor: {overhead['slowdown_factor']:.2f}x")

    # Compare with HE
    print("\nComparing TEE vs. HE:")
    comparison = compare_tee_vs_he(
        plaintext_time_ns=10000,  # 10 μs
        he_time_ns=10000000,      # 10 ms (1000x slower)
        tee_overhead_ns=overhead['total_overhead_ns']
    )

    print(f"  Plaintext: {comparison['plaintext_time_ns']/1000:.1f} μs")
    print(f"  HE time: {comparison['he_time_ns']/1000:.1f} μs ({comparison['he_slowdown']:.1f}x)")
    print(f"  TEE time: {comparison['tee_time_ns']/1000:.1f} μs ({comparison['tee_slowdown']:.2f}x)")
    print(f"  TEE speedup vs HE: {comparison['tee_speedup_vs_he']:.1f}x")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")

    print("✓ Demonstrated all core TEE functionality:")
    print("  1. Enclave creation and management")
    print("  2. Secure data entry/exit")
    print("  3. Isolated execution")
    print("  4. Remote attestation")
    print("  5. Sealed storage")
    print("  6. Realistic overhead modeling")
    print("  7. TEE vs. HE comparison")

    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
