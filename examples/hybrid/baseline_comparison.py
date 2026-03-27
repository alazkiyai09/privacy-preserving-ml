"""
Baseline Comparison Demo for HT2ML
====================================

Compares performance of three inference approaches:
1. HE-only (fully encrypted, slow)
2. TEE-only (fast, requires trust)
3. Hybrid HE/TEE (balanced)

This demonstrates the trade-offs between:
- Privacy (HE > Hybrid > TEE)
- Performance (TEE > Hybrid > HE)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Any

from src.inference.hybrid_engine import create_hybrid_engine
from src.inference.he_only_engine import create_he_only_engine
from src.inference.tee_only_engine import create_tee_only_engine
from config.model_config import create_default_config
from src.model.phishing_classifier import create_random_model


def print_comparison_header() -> None:
    """Print comparison header."""
    print("\n" + "#"*70)
    print("# HT2ML Baseline Comparison")
    print("# Comparing HE-only, TEE-only, and Hybrid approaches")
    print("#"*70)


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print comparison table."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Approach':<15} {'Time (ms)':<12} {'HE Time':<10} {'TEE Time':<10} {'Handoffs':<10}")
    print("-"*70)

    for name, stats in results.items():
        print(f"{name:<15} {stats['time_ms']:<12.2f} {stats['he_time_ms']:<10.2f} "
              f"{stats['tee_time_ms']:<10.2f} {stats['handoffs']:<10}")

    print("="*70)

    # Calculate speedup
    he_time = results['HE-only']['time_ms']
    tee_time = results['TEE-only']['time_ms']
    hybrid_time = results['Hybrid']['time_ms']

    print(f"\nSpeedup Analysis:")
    print(f"  TEE-only vs HE-only: {he_time/tee_time:.1f}x faster")
    print(f"  Hybrid vs HE-only: {he_time/hybrid_time:.1f}x faster")
    print(f"  TEE-only vs Hybrid: {hybrid_time/tee_time:.1f}x faster")
    print()


def print_privacy_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """Print privacy comparison."""
    print("="*70)
    print("PRIVACY ANALYSIS")
    print("="*70)

    print("\nHE-only:")
    print("  ✓ Complete privacy - data never decrypted")
    print("  ✓ No trust required in server")
    print("  ✗ Slow performance")
    print(f"  Noise budget consumed: {results['HE-only']['noise_used']} bits")

    print("\nTEE-only:")
    print("  ✓ Fast performance")
    print("  ✓ Trusted execution environment")
    print("  ✗ Requires trust in TEE manufacturer")
    print("  ✗ Data decrypted in enclave")

    print("\nHybrid HE/TEE:")
    print("  ✓ Linear ops encrypted (HE)")
    print("  ✓ Non-linear ops in TEE")
    print("  ✓ Balanced performance")
    print(f"  ✓ Noise budget saved: {results['HE-only']['noise_used'] - results['Hybrid']['noise_used']} bits")
    print(f"  ✓ {results['Hybrid']['handoffs']} secure handoffs with attestation")
    print()


def run_single_comparison():
    """Run single inference comparison."""
    print_comparison_header()

    # Create model
    print("\nCreating phishing classifier model...")
    config = create_default_config()
    model = create_random_model(config)

    # Generate sample input
    np.random.seed(42)  # For reproducibility
    features = np.random.randn(50).astype(np.float32)

    print(f"Input features: {features.shape}")
    print(f"Input range: [{features.min():.3f}, {features.max():.3f}]")

    # Store results
    results = {}

    # 1. HE-only inference
    print("\n" + "-"*70)
    print("Running HE-only inference...")
    print("-"*70)

    he_engine = create_he_only_engine(model)
    he_result = he_engine.run_inference(features)

    results['HE-only'] = {
        'time_ms': he_result.execution_time_ms,
        'he_time_ms': he_result.he_time_ms,
        'tee_time_ms': he_result.tee_time_ms,
        'handoffs': he_result.num_handoffs,
        'noise_used': he_result.noise_budget_used,
        'noise_remaining': he_result.noise_budget_remaining,
    }

    # 2. TEE-only inference
    print("\n" + "-"*70)
    print("Running TEE-only inference...")
    print("-"*70)

    tee_engine = create_tee_only_engine(model)
    tee_result = tee_engine.run_inference(features)

    results['TEE-only'] = {
        'time_ms': tee_result.execution_time_ms,
        'he_time_ms': tee_result.he_time_ms,
        'tee_time_ms': tee_result.tee_time_ms,
        'handoffs': tee_result.num_handoffs,
        'noise_used': 0,
        'noise_remaining': 0,
    }

    # 3. Hybrid inference
    print("\n" + "-"*70)
    print("Running Hybrid HE/TEE inference...")
    print("-"*70)

    hybrid_engine = create_hybrid_engine(model)
    hybrid_result = hybrid_engine.run_inference(features)

    results['Hybrid'] = {
        'time_ms': hybrid_result.execution_time_ms,
        'he_time_ms': hybrid_result.he_time_ms,
        'tee_time_ms': hybrid_result.tee_time_ms,
        'handoffs': hybrid_result.num_handoffs,
        'noise_used': hybrid_result.noise_budget_used,
        'noise_remaining': hybrid_result.noise_budget_remaining,
    }

    # Print comparison
    print_comparison_table(results)
    print_privacy_comparison(results)

    # Print detailed results
    print("="*70)
    print("DETAILED RESULTS")
    print("="*70)

    for name, result in [('HE-only', he_result), ('TEE-only', tee_result), ('Hybrid', hybrid_result)]:
        print(f"\n{name}:")
        print(f"  Predicted: {result.get_class_name()} (class {result.class_id})")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")
        if result.num_handoffs > 0:
            print(f"  Handoffs: {result.num_handoffs}")
        if result.noise_budget_used > 0:
            print(f"  Noise used: {result.noise_budget_used} bits")
            print(f"  Noise remaining: {result.noise_budget_remaining} bits")


def run_batch_comparison(num_samples: int = 5):
    """Run batch inference comparison."""
    print_comparison_header()
    print(f"\nRunning batch comparison with {num_samples} samples...")

    # Create model
    config = create_default_config()
    model = create_random_model(config)

    # Generate batch
    np.random.seed(42)
    features_batch = np.random.randn(num_samples, 50).astype(np.float32)

    results = {
        'HE-only': [],
        'TEE-only': [],
        'Hybrid': [],
    }

    # Run inferences
    for i, features in enumerate(features_batch):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*70}")

        # HE-only (run once to show it works)
        if i == 0:
            he_engine = create_he_only_engine(model)
            he_result = he_engine.run_inference(features)
            results['HE-only'].append(he_result.execution_time_ms)

        # TEE-only
        tee_engine = create_tee_only_engine(model)
        tee_result = tee_engine.run_inference(features)
        results['TEE-only'].append(tee_result.execution_time_ms)

        # Hybrid
        hybrid_engine = create_hybrid_engine(model)
        hybrid_result = hybrid_engine.run_inference(features)
        results['Hybrid'].append(hybrid_result.execution_time_ms)

    # Calculate statistics
    print("\n" + "="*70)
    print("BATCH STATISTICS")
    print("="*70)
    print(f"{'Approach':<15} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*70)

    for name, times in results.items():
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{name:<15} {avg_time:<12.2f} {min_time:<12.2f} {max_time:<12.2f}")

    print("="*70)


def main():
    """Run comparison demo."""
    # Single inference comparison
    run_single_comparison()

    # Uncomment for batch comparison
    # run_batch_comparison(num_samples=5)

    print("\n" + "#"*70)
    print("# Comparison Complete")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
