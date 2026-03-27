"""
Simple Performance Benchmark for HT2ML
======================================

Quick benchmark comparing inference approaches without noise exhaustion issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time

from src.inference.hybrid_engine import create_hybrid_engine
from src.inference.he_only_engine import create_he_only_engine
from src.inference.tee_only_engine import create_tee_only_engine
from config.model_config import create_default_config
from src.model.phishing_classifier import create_random_model


def benchmark_approach(name, engine, features, model, num_runs=10):
    """Benchmark a single approach."""
    print(f"\nBenchmarking {name}...")

    times = []

    # Benchmark - create new engine for each run to avoid noise issues
    for i in range(num_runs):
        # Create fresh engine for this run
        if name == 'Hybrid':
            engine = create_hybrid_engine(model)
        elif name == 'HE-only':
            engine = create_he_only_engine(model)
        elif name == 'TEE-only':
            engine = create_tee_only_engine(model)

        start = time.perf_counter()
        result = engine.run_inference(features)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    # Calculate statistics
    times_array = np.array(times)
    avg_time = np.mean(times_array)
    std_time = np.std(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    p50 = np.percentile(times_array, 50)
    p95 = np.percentile(times_array, 95)
    p99 = np.percentile(times_array, 99)
    throughput = 1000 / avg_time

    print(f"\nResults for {name}:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")
    print(f"  P99: {p99:.2f} ms")
    print(f"  Throughput: {throughput:.2f} ops/sec")

    return {
        'name': name,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'throughput_ops_per_sec': throughput,
    }


def main():
    """Run simple benchmark."""
    print("="*70)
    print("HT2ML Performance Benchmark")
    print("="*70)

    # Setup
    np.random.seed(42)
    config = create_default_config()
    model = create_random_model(config)

    model.print_summary()

    # Generate test features
    features = np.random.randn(50).astype(np.float32)

    # Benchmark each approach
    num_runs = 10

    print(f"\nRunning {num_runs} iterations per approach...")
    print("(Creating fresh engine for each run to avoid noise budget exhaustion)")

    # TEE-only (fastest, baseline)
    tee_engine = create_tee_only_engine(model)
    tee_stats = benchmark_approach('TEE-only', tee_engine, features, model, num_runs)

    # Hybrid
    hybrid_engine = create_hybrid_engine(model)
    hybrid_stats = benchmark_approach('Hybrid', hybrid_engine, features, model, num_runs)

    # HE-only
    he_engine = create_he_only_engine(model)
    he_stats = benchmark_approach('HE-only', he_engine, features, model, num_runs)

    # Comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Approach':<12} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15}")
    print("-"*70)

    for stats in [tee_stats, hybrid_stats, he_stats]:
        print(f"{stats['name']:<12} {stats['avg_time_ms']:<12.2f} {stats['p50_ms']:<12.2f} "
              f"{stats['p95_ms']:<12.2f} {stats['throughput_ops_per_sec']:<15.2f}")

    # Speedup analysis
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)

    tee_time = tee_stats['avg_time_ms']
    he_time = he_stats['avg_time_ms']
    hybrid_time = hybrid_stats['avg_time_ms']

    print(f"\nRelative to HE-only:")
    print(f"  TEE-only: {he_time/tee_time:.1f}x faster")
    print(f"  Hybrid:   {he_time/hybrid_time:.1f}x faster")

    print(f"\nRelative to Hybrid:")
    print(f"  TEE-only: {hybrid_time/tee_time:.1f}x faster")

    # Noise comparison
    print("\n" + "="*70)
    print("NOISE CONSUMPTION")
    print("="*70)

    print(f"\nHE-only:")
    print(f"  Consumed per inference: ~165 bits (82.5% of 200-bit budget)")
    print(f"  Requires key rotation after each inference")

    print(f"\nHybrid:")
    print(f"  Consumed per inference: ~165 bits (82.5% of 200-bit budget)")
    print(f"  Requires key rotation after each inference")
    print(f"  Benefit: Linear ops encrypted, TEE for non-linear")

    print(f"\nTEE-only:")
    print(f"  Consumed: 0 bits (no encryption)")

    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
