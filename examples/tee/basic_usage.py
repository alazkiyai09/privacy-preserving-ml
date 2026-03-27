"""
Basic TEE Usage Example
========================

Demonstrates basic TEE operations including:
- Creating an enclave
- Executing operations in TEE
- Using attestation
- Sealed storage
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave, create_enclave
from tee_ml.core.attestation import AttestationService, create_attestation_service
from tee_ml.core.sealed_storage import SealedStorage, create_sealed_storage


def basic_enclave_usage():
    """Demonstrate basic enclave usage."""
    print("=" * 70)
    print("Basic TEE Enclave Usage")
    print("=" * 70)
    print()

    # 1. Create an enclave
    print("1. Creating enclave...")
    enclave = create_enclave(
        enclave_id="example-enclave",
        memory_limit_mb=128,
    )
    print(f"   ✓ Created enclave: {enclave.enclave_id}")
    print(f"   ✓ Memory limit: {enclave.memory_limit_mb} MB")
    print()

    # 2. Enter enclave and execute operation
    print("2. Executing operation in enclave...")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    session = enclave.enter(data)
    print(f"   ✓ Entered enclave with data: {data}")

    # Execute a simple operation
    result = session.execute(lambda arr: arr * 2)
    print(f"   ✓ Operation result: {result}")

    # 3. Exit enclave
    enclave.exit(session)
    print(f"   ✓ Exited enclave")
    print()

    # 4. Get enclave statistics
    print("3. Enclave statistics...")
    stats = enclave.get_statistics()
    print(f"   ✓ Total sessions: {stats['total_sessions']}")
    print(f"   ✓ Memory usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"   ✓ Memory utilization: {stats['memory_utilization']:.2%}")
    print()


def attestation_example():
    """Demonstrate remote attestation."""
    print("=" * 70)
    print("Remote Attestation Example")
    print("=" * 70)
    print()

    # Create enclave and attestation service
    enclave = create_enclave(enclave_id="attestation-example")
    attestation_service = create_attestation_service(enclave)

    print("1. Generating attestation report...")
    report = attestation_service.generate_report(
        enclave=enclave,
        nonce=b"challenge-nonce-12345",
    )
    print(f"   ✓ Report generated for enclave: {report.enclave_id}")
    print(f"   ✓ Measurement: {report.measurement.hex()[:40]}...")
    print(f"   ✓ Nonce: {report.nonce.hex()}")
    print()

    print("2. Verifying attestation report...")
    verification = attestation_service.verify_report(report)
    print(f"   ✓ Verification result: {verification.is_valid}")
    print(f"   ✓ Verification message: {verification.message}")
    print()

    print("3. Performing remote attestation...")
    success = attestation_service.remote_attestation(
        enclave=enclave,
        challenger_id="client-app-123",
    )
    print(f"   ✓ Remote attestation success: {success}")
    print()


def sealed_storage_example():
    """Demonstrate sealed storage for persistent data."""
    print("=" * 70)
    print("Sealed Storage Example")
    print("=" * 70)
    print()

    # Create enclave and sealed storage
    enclave = create_enclave(enclave_id="sealed-storage-example")
    sealed_storage = create_sealed_storage(enclave)

    # Data to seal
    secret_key = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"1. Sealing secret key: {secret_key}")

    # Seal the data
    sealed_data = sealed_storage.seal(
        data=secret_key.tobytes(),
        enclave_id=enclave.enclave_id,
        measurement=enclave.get_measurement(),
    )
    print(f"   ✓ Data sealed")
    print(f"   ✓ Encrypted size: {len(sealed_data.encrypted_data)} bytes")
    print()

    # Save to file
    storage_path = Path("/tmp/sealed_key.bin")
    sealed_storage.save_sealed(sealed_data, storage_path)
    print(f"2. Saved sealed data to: {storage_path}")
    print()

    # Load from file
    print("3. Loading sealed data...")
    loaded_sealed = sealed_storage.load_sealed(storage_path)
    print(f"   ✓ Sealed data loaded")
    print()

    # Unseal the data
    print("4. Unsealing data...")
    unsealed_bytes = sealed_storage.unseal(
        sealed_data=loaded_sealed,
        enclave_id=enclave.enclave_id,
        measurement=enclave.get_measurement(),
    )
    unsealed_key = np.frombuffer(unsealed_bytes, dtype=np.float64)
    print(f"   ✓ Unsealed key: {unsealed_key}")
    print(f"   ✓ Original matches: {np.allclose(secret_key, unsealed_key)}")
    print()


def ml_operations_example():
    """Demonstrate ML operations in TEE."""
    print("=" * 70)
    print("ML Operations in TEE")
    print("=" * 70)
    print()

    from tee_ml.operations.activations import (
        tee_relu,
        tee_sigmoid,
        tee_softmax,
    )
    from tee_ml.operations.comparisons import (
        tee_argmax,
        tee_threshold,
    )
    from tee_ml.operations.arithmetic import (
        tee_normalize,
        tee_divide,
    )

    enclave = create_enclave(enclave_id="ml-operations-example")

    # Input data
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    print(f"Input: {x}")
    print()

    # Activations
    print("1. Activations:")
    session = enclave.enter(x)

    relu_result = tee_relu(x, session)
    print(f"   ReLU: {relu_result}")

    sigmoid_result = tee_sigmoid(x, session)
    print(f"   Sigmoid: {sigmoid_result}")

    softmax_input = np.array([2.0, 1.0, 0.1])
    softmax_result = tee_softmax(softmax_input, session)
    print(f"   Softmax: {softmax_result}")

    print()

    # Comparisons
    print("2. Comparisons:")
    argmax_result = tee_argmax(softmax_input, session)
    print(f"   Argmax: {argmax_result} (index of max value)")

    threshold_result = tee_threshold(x, 0.5, session)
    print(f"   Threshold (>0.5): {threshold_result}")

    print()

    # Arithmetic
    print("3. Arithmetic:")
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = tee_normalize(vector, session)
    print(f"   Normalize: {normalized}")
    print(f"   L2 norm: {np.linalg.norm(normalized):.4f}")

    divided = tee_divide(vector, 2.0, session)
    print(f"   Divide by 2: {divided}")

    enclave.exit(session)
    print()


def complete_workflow_example():
    """Demonstrate a complete TEE workflow."""
    print("=" * 70)
    print("Complete TEE Workflow: Privacy-Preserving Inference")
    print("=" * 70)
    print()

    # Setup
    enclave = create_enclave(enclave_id="inference-enclave")
    attestation_service = create_attestation_service(enclave)
    sealed_storage = create_sealed_storage(enclave)

    print("Scenario: Client wants to run inference on sensitive data")
    print()

    # 1. Verify enclave integrity
    print("1. Verifying enclave integrity...")
    report = attestation_service.generate_report(enclave, nonce=b"client-nonce")
    verification = attestation_service.verify_report(report)
    print(f"   ✓ Enclave verified: {verification.is_valid}")
    print()

    # 2. Load sealed model weights
    print("2. Loading sealed model weights...")
    # In real scenario, these would be actual model weights
    weights = np.random.randn(10, 5) * 0.1
    sealed_weights = sealed_storage.seal(
        weights.tobytes(),
        enclave.enclave_id,
        enclave.get_measurement(),
    )
    print(f"   ✓ Model weights loaded and sealed")
    print()

    # 3. Client sends sensitive data
    print("3. Processing client's sensitive data...")
    sensitive_data = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5])

    # Enter enclave with sensitive data
    session = enclave.enter(sensitive_data)

    # Perform inference (simplified)
    def inference_func(arr):
        # In real scenario, this would be actual neural network forward pass
        weights_matrix = np.frombuffer(sealed_weights.encrypted_data, dtype=np.float64)
        weights_matrix = weights_matrix.reshape(10, 5)
        return np.dot(arr, weights_matrix)

    result = session.execute(inference_func)
    print(f"   ✓ Inference result: {result}")
    print()

    # 4. Exit enclave
    enclave.exit(session)

    # 5. Statistics
    stats = enclave.get_statistics()
    print("5. Session statistics:")
    print(f"   ✓ Total operations: {stats['total_sessions']}")
    print(f"   ✓ Memory used: {stats['memory_usage_mb']:.2f} MB")
    print()

    print("=" * 70)
    print("Workflow Complete!")
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TEE ML Framework Examples" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    # Run examples
    try:
        basic_enclave_usage()
        attestation_example()
        sealed_storage_example()
        ml_operations_example()
        complete_workflow_example()

        print("\n")
        print("✓ All examples completed successfully!")
        print("\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
