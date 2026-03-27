"""
HT2ML Hybrid Inference Demo
============================

Demonstrates end-to-end hybrid HE/TEE inference for phishing detection.

This script shows:
1. Model initialization with random weights
2. HE key generation
3. Input encryption
4. Hybrid inference with multiple HE↔TEE handoffs
5. Result decryption
6. Performance statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from src.inference.hybrid_engine import create_hybrid_engine, run_single_inference
from config.model_config import create_default_config
from src.model.phishing_classifier import create_random_model


def print_section(title: str) -> None:
    """Print section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def demo_single_inference():
    """Demonstrate single inference."""
    print_section("HT2ML Hybrid Inference Demo")

    # Create model with random weights
    print("Creating phishing classifier model...")
    config = create_default_config()
    model = create_random_model(config)

    # Create hybrid inference engine
    print("Creating hybrid inference engine...")
    engine = create_hybrid_engine(model)

    # Generate sample input (50 features)
    print("\nGenerating sample input features...")
    features = np.random.randn(50).astype(np.float32)
    print(f"Input shape: {features.shape}")
    print(f"Input range: [{features.min():.3f}, {features.max():.3f}]")

    # Run inference
    print("\n" + "-"*70)
    print("Running hybrid inference...")
    print("-"*70)

    result = engine.run_inference(features)

    # Display results
    print_section("Inference Results")

    print(f"Predicted Class: {result.class_id}")
    print(f"Class Name: {result.get_class_name()}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"\nPerformance:")
    print(f"  Total time: {result.execution_time_ms:.2f}ms")
    print(f"  HE time: {result.he_time_ms:.2f}ms")
    print(f"  TEE time: {result.tee_time_ms:.2f}ms")
    print(f"  Handoff time: {result.handoff_time_ms:.2f}ms")
    print(f"  Handoffs: {result.num_handoffs}")
    print(f"\nNoise Budget:")
    print(f"  Consumed: {result.noise_budget_used} bits")
    print(f"  Remaining: {result.noise_budget_remaining} bits")

    # Display engine statistics
    print_section("Engine Statistics")
    engine.print_stats()


def demo_batch_inference():
    """Demonstrate batch inference."""
    print_section("HT2ML Batch Inference Demo")

    # Create engine
    config = create_default_config()
    model = create_random_model(config)
    engine = create_hybrid_engine(model)

    # Generate batch of inputs
    batch_size = 3
    print(f"Generating batch of {batch_size} samples...")
    features_batch = np.random.randn(batch_size, 50).astype(np.float32)

    # Run batch inference
    print("\nRunning batch inference...\n")
    results = engine.run_batch_inference(features_batch)

    # Display results
    print_section("Batch Inference Results")

    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Predicted: {result.get_class_name()} (class {result.class_id})")
        print(f"  Time: {result.execution_time_ms:.2f}ms")
        print(f"  Handoffs: {result.num_handoffs}")

    # Statistics
    avg_time = engine.get_average_inference_time()
    print(f"\nAverage inference time: {avg_time:.2f}ms")


def demo_architecture():
    """Demonstrate HT2ML architecture."""
    print_section("HT2ML Architecture Demo")

    # Create model
    config = create_default_config()
    model = create_random_model(config)

    # Display architecture
    model.print_summary()

    # Show execution domains
    print("\nExecution Domain Breakdown:")
    print("-"*70)

    for i, layer_spec in enumerate(config.layers):
        domain = layer_spec.domain.value.upper()
        layer_type = layer_spec.layer_type.value
        print(f"  {i+1}. {layer_spec.name:10s} | {layer_type:10s} | {domain:6s} | "
              f"({layer_spec.input_size:3d} → {layer_spec.output_size:3d})")

        # Show handoffs
        if i < len(config.layers) - 1:
            next_domain = config.layers[i+1].domain.value.upper()
            if layer_spec.domain != config.layers[i+1].domain:
                print(f"      ↓ HANDOFF: {domain} → {next_domain}")

    print(f"\nTotal HE/TEE Handoffs: {config.get_num_handoffs()}")


def main():
    """Run all demos."""
    print("\n" + "#"*70)
    print("# HT2ML Hybrid HE/TEE Phishing Detection")
    print("# Hybrid Homomorphic Encryption + Trusted Execution Environment")
    print("#"*70)

    # Demo 1: Architecture
    demo_architecture()

    # Demo 2: Single inference
    demo_single_inference()

    # Demo 3: Batch inference (optional - comment out to skip)
    # demo_batch_inference()

    print("\n" + "#"*70)
    print("# Demo Complete")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
