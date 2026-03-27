"""
HT2ML Hybrid Workflow Example
==============================

Demonstrates the complete HT2ML hybrid system:
- HE encryption for input privacy
- TEE processing for performance
- Handoff protocol between HE and TEE
- Optimal split point analysis
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave, create_enclave
from tee_ml.protocol.handoff import (
    HEContext,
    HEData,
    HT2MLProtocol,
    create_handoff_protocol,
)
from tee_ml.protocol.split_optimizer import (
    LayerSpecification,
    SplitStrategy,
    SplitOptimizer,
    create_layer_specifications,
    estimate_optimal_split,
    visualize_split,
    analyze_tradeoffs,
)


def create_sample_network():
    """Create a sample neural network architecture."""
    print("=" * 70)
    print("Creating Sample Neural Network")
    print("=" * 70)
    print()

    # Define network architecture
    input_size = 20
    hidden_sizes = [10, 5]
    output_size = 2
    activations = ['relu', 'sigmoid', 'softmax']

    print("Network Architecture:")
    print(f"  Input:  {input_size} features")
    print(f"  Hidden: {hidden_sizes}")
    print(f"  Output: {output_size} classes")
    print(f"  Activations: {activations}")
    print()

    return input_size, hidden_sizes, output_size, activations


def analyze_optimal_split():
    """Analyze optimal HE/TEE split point."""
    print("=" * 70)
    print("Optimal Split Point Analysis")
    print("=" * 70)
    print()

    # Create network architecture
    input_size, hidden_sizes, output_size, activations = create_sample_network()

    # Estimate optimal split with different strategies
    print("Analyzing split points with different strategies...\n")

    strategies = [
        SplitStrategy.PRIVACY_MAX,
        SplitStrategy.PERFORMANCE_MAX,
        SplitStrategy.BALANCED,
    ]

    recommendations = {}

    for strategy in strategies:
        print(f"\n{strategy.value.upper()} Strategy:")
        print("-" * 70)

        # Note: Using small layers to fit in noise budget
        rec = estimate_optimal_split(
            input_size=input_size,
            hidden_sizes=[5, 2],  # Smaller layers
            output_size=output_size,
            activations=activations,
            noise_budget=200,
            strategy=strategy,
        )

        recommendations[strategy.value] = rec

        rec.print_summary()

    print()

    return recommendations


def simulate_ht2ml_workflow():
    """Simulate complete HT2ML workflow."""
    print("=" * 70)
    print("HT2ML Hybrid Workflow Simulation")
    print("=" * 70)
    print()

    # Step 1: Create network and find optimal split
    print("Step 1: Define network and find optimal split")
    print("-" * 70)

    layers = create_layer_specifications(
        input_size=20,
        hidden_sizes=[5, 2],  # Small layers to fit in HE budget
        output_size=2,
        activations=['relu', 'sigmoid', 'softmax'],
    )

    optimizer = SplitOptimizer(noise_budget=200)
    rec = optimizer.recommend_split(layers, SplitStrategy.BALANCED)

    print(f"Optimal split: {rec.he_layers} HE layers, {rec.tee_layers} TEE layers")
    print(f"Split after layer: {rec.split_point}")
    print()

    # Step 2: Create HE context and encrypt input
    print("Step 2: Encrypt input with HE")
    print("-" * 70)

    # Client input
    client_input = np.random.randn(20)
    print(f"Client input shape: {client_input.shape}")
    print(f"Client input (first 5): {client_input[:5]}")

    # Create HE context (simulated)
    he_context = HEContext(
        scheme='ckks',
        poly_modulus_degree=4096,
        scale=2**30,
        eval=1,
    )

    # Simulate HE encryption
    # In real system, this would use TenSEAL
    encrypted_data = HEData(
        encrypted_data=[],  # Simulated encrypted data
        shape=client_input.shape,
        scheme='ckks',
        scale=2**30,
    )

    print(f"✓ Input encrypted with HE (CKKS)")
    print(f"✓ Scheme: {he_context.scheme}")
    print(f"✓ Polynomial modulus degree: {he_context.poly_modulus_degree}")
    print()

    # Step 3: Create TEE enclave and protocol
    print("Step 3: Setup TEE enclave and handoff protocol")
    print("-" * 70)

    enclave = create_enclave(enclave_id="ht2ml-enclave")
    protocol = create_handoff_protocol(enclave)

    print(f"✓ Created TEE enclave: {enclave.enclave_id}")
    print(f"✓ Created HT2ML protocol")
    print()

    # Step 4: HE→TEE handoff
    print("Step 4: Perform HE→TEE handoff")
    print("-" * 70)

    # In real system, HE layers would be executed here
    # For simulation, we skip to handoff

    success, plaintext = protocol.handoff_he_to_tee(
        encrypted_data=encrypted_data,
        he_context=he_context,
        nonce=b"ht2ml-nonce-12345",
    )

    if success:
        print(f"✓ Handoff successful")
        print(f"✓ Data decrypted in TEE enclave")
        print()

        # Step 5: Process in TEE
        print("Step 5: Process remaining layers in TEE")
        print("-" * 70)

        # Enter enclave with data
        session = enclave.enter(plaintext)

        # Simulate neural network layers in TEE
        # Layer 1 (after handoff)
        def layer1(arr):
            weights = np.random.randn(5, 20) * 0.1
            bias = np.random.randn(5) * 0.01
            output = np.dot(weights, arr) + bias
            return np.maximum(0, output)  # ReLU

        result1 = session.execute(layer1)
        print(f"  Layer 1 (ReLU): input shape (20,) → output shape (5,)")

        # Layer 2
        def layer2(arr):
            weights = np.random.randn(2, 5) * 0.1
            bias = np.random.randn(2) * 0.01
            output = np.dot(weights, arr) + bias
            return 1 / (1 + np.exp(-output))  # Sigmoid

        result2 = session.execute(layer2)
        print(f"  Layer 2 (Sigmoid): input shape (5,) → output shape (2,)")

        # Layer 3 (output)
        def layer3(arr):
            exp_arr = np.exp(arr)
            return exp_arr / np.sum(exp_arr)  # Softmax

        result3 = session.execute(layer3)
        print(f"  Layer 3 (Softmax): input shape (2,) → output shape (2,)")
        print()

        # Exit enclave
        enclave.exit(session)

        # Step 6: Get prediction
        print("Step 6: Final Prediction")
        print("-" * 70)

        predicted_class = np.argmax(result3)
        confidence = result3[predicted_class]

        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {result3}")
        print()

    else:
        print("✗ Handoff failed")
        return

    # Step 7: Statistics
    print("Step 7: Handoff Statistics")
    print("-" * 70)

    stats = protocol.get_handoff_statistics()
    print(f"Total handoffs: {stats['total_handoffs']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average time: {stats['avg_time_ns'] / 1000:.2f} μs")
    print()

    print("=" * 70)
    print("HT2ML Workflow Complete!")
    print("=" * 70)
    print()

    # Visualize the split
    print("\nNetwork Visualization:")
    visualize_split(rec, layers)


def compare_architectures():
    """Compare different privacy-preserving architectures."""
    print("=" * 70)
    print("Architecture Comparison: HE vs TEE vs HT2ML")
    print("=" * 70)
    print()

    architectures = [
        ("Pure HE", {
            'privacy': 'Cryptographic (math)',
            'trust': 'Mathematics only',
            'performance': '100-1000x slower',
            'depth': '1-2 layers max',
            'nonlinear_ops': 'Very expensive',
            'use_case': 'Input privacy only',
        }),
        ("Pure TEE", {
            'privacy': 'Hardware-based',
            'trust': 'Hardware vendor',
            'performance': '1.1-2x slower',
            'depth': 'Unlimited',
            'nonlinear_ops': 'Efficient',
            'use_case': 'Computation privacy',
        }),
        ("HT2ML Hybrid", {
            'privacy': 'Cryptographic (HE) + Hardware (TEE)',
            'trust': 'Math OR hardware (flexible)',
            'performance': '10-100x slower (HE dominates)',
            'depth': 'Unlimited',
            'nonlinear_ops': 'Efficient (in TEE)',
            'use_case': 'Input + computation privacy',
        }),
    ]

    # Print comparison table
    print(f"{'Architecture':<20} {'Privacy':<25} {'Trust':<20} {'Performance':<20}")
    print("-" * 70)

    for name, props in architectures:
        print(f"{name:<20} {props['privacy']:<25} {props['trust']:<20} {props['performance']:<20}")

    print()
    print("Detailed Comparison:")
    print()

    for name, props in architectures:
        print(f"{name}:")
        print(f"  Depth: {props['depth']}")
        print(f"  Non-linear ops: {props['nonlinear_ops']}")
        print(f"  Best for: {props['use_case']}")
        print()


def analyze_tradeoffs_example():
    """Analyze trade-offs between different strategies."""
    print("=" * 70)
    print("Trade-off Analysis")
    print("=" * 70)
    print()

    # Create network architecture
    input_size = 20
    hidden_sizes = [5, 2]
    output_size = 2
    activations = ['relu', 'sigmoid', 'softmax']

    # Analyze trade-offs
    analysis = analyze_tradeoffs(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activations=activations,
        noise_budget=200,
    )

    print("Network Configuration:")
    print(f"  Layers: {analysis['num_layers']}")
    print(f"  Noise budget: {analysis['noise_budget']} bits")
    print(f"  Feasible splits: {analysis['feasible_splits']}")
    print()

    print("Strategy Comparison:")
    print()

    for strategy_name, rec_data in analysis['recommendations'].items():
        print(f"{strategy_name.upper()}:")
        print(f"  HE layers: {rec_data['he_layers']}")
        print(f"  TEE layers: {rec_data['tee_layers']}")
        print(f"  Privacy score: {rec_data['privacy_score']:.2f}/1.00")
        print(f"  Performance score: {rec_data['performance_score']:.2f}/1.00")
        print(f"  Estimated time: {rec_data['estimated_time_ms']:.2f} ms")
        print(f"  Feasible: {rec_data['feasible']}")
        print()


def security_analysis():
    """Analyze security properties of HT2ML."""
    print("=" * 70)
    print("HT2ML Security Analysis")
    print("=" * 70)
    print()

    print("Threat Model:")
    print()

    threats = [
        ("Malicious OS", {
            'can_read_input': 'No (HE encrypted)',
            'can_read_model': 'No (TEE protected)',
            'can_read_computation': 'Partial (side-channels)',
            'mitigation': 'Constant-time ops, cache randomization',
        }),
        ("Malicious Client", {
            'can_read_input': 'Yes (they own it)',
            'can_read_model': 'No',
            'can_read_computation': 'No',
            'mitigation': 'Input validation, rate limiting',
        }),
        ("Network Attacker", {
            'can_read_input': 'No (HE encrypted)',
            'can_read_model': 'No (TEE protected)',
            'can_read_computation': 'No',
            'mitigation': 'Secure channel, TLS',
        }),
    ]

    print(f"{'Threat':<20} {'Input':<12} {'Model':<12} {'Computation':<12} {'Mitigation':<30}")
    print("-" * 70)

    for threat, props in threats:
        print(f"{threat:<20} {props['can_read_input']:<12} {props['can_read_model']:<12} "
              f"{props['can_read_computation']:<12} {props['mitigation']:<30}")

    print()
    print("Security Properties:")
    print()
    print("✓ Input Privacy: Cryptographic (HE - CKKS)")
    print("✓ Model Privacy: Hardware (TEE - SGX)")
    print("✓ Computation Privacy: Hardware (TEE - SGX)")
    print("✓ Attestation: Remote attestation verifies TEE integrity")
    print("✓ Replay Protection: Nonces in handoff protocol")
    print()


def main():
    """Run all HT2ML workflow examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "HT2ML Hybrid System Examples" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    try:
        # Run examples
        analyze_optimal_split()
        simulate_ht2ml_workflow()
        compare_architectures()
        analyze_tradeoffs_example()
        security_analysis()

        print("\n")
        print("✓ All HT2ML workflow examples completed successfully!")
        print("\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
