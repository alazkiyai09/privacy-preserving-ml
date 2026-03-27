"""
Benchmarking Suite for Homomorphic Encryption ML
=================================================
Performance analysis and comparison tools.

This module provides:
1. Performance benchmarks (HE vs. plaintext)
2. Memory usage analysis
3. Scalability studies
4. Cost-benefit analysis
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import time
import psutil
import gc

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    operation: str

    # Timing metrics (seconds)
    total_time: float
    avg_time: float
    std_time: float
    min_time: float
    max_time: float

    # Throughput metrics
    throughput: float  # operations per second

    # Memory metrics (MB)
    memory_mb: float

    # Additional metadata
    num_runs: int
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from comparing HE vs. plaintext."""

    operation: str

    # Plaintext metrics
    plaintext_time: float
    plaintext_throughput: float

    # HE metrics
    he_time: float
    he_throughput: float

    # Comparison
    slowdown_factor: float  # How much slower HE is
    efficiency: float  # Relative performance

    # Memory overhead
    memory_overhead_mb: float

    # Verdict
    feasible: bool
    recommendation: str


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for HE operations.

    Measures:
    - Execution time (encryption, inference, decryption)
    - Memory usage
    - Throughput
    - Scalability with batch size
    """

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        """
        Initialize benchmark suite.

        Args:
            warmup_runs: Number of warmup runs before benchmarking
            benchmark_runs: Number of benchmark runs for statistics
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.process = psutil.Process()

    def _measure_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def _benchmark_operation(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single operation.

        Args:
            operation: Function to benchmark
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            BenchmarkResult with timing and memory metrics
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                result = operation(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Force garbage collection
        gc.collect()

        # Measure memory before
        memory_before = self._measure_memory()

        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            try:
                result = operation(*args, **kwargs)
            except Exception:
                # Skip failed runs
                continue
            end = time.perf_counter()
            times.append(end - start)

        # Measure memory after
        memory_after = self._measure_memory()

        if not times:
            raise RuntimeError("All benchmark runs failed")

        times = np.array(times)

        return BenchmarkResult(
            name=operation.__name__,
            operation=operation.__name__,
            total_time=np.sum(times),
            avg_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            throughput=1.0 / np.mean(times),
            memory_mb=memory_after - memory_before,
            num_runs=len(times),
        )

    def benchmark_encryption(
        self,
        data: np.ndarray,
        context: Any,
        num_runs: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark encryption operation.

        Args:
            data: Input data to encrypt
            context: TenSEAL context
            num_runs: Override default number of runs

        Returns:
            BenchmarkResult
        """
        from he_ml.core.encryptor import encrypt_vector

        def encrypt_single():
            return encrypt_vector(data, context, scheme='ckks')

        return self._benchmark_operation(
            encrypt_single,
            num_runs=num_runs or self.benchmark_runs
        )

    def benchmark_decryption(
        self,
        ciphertext: CiphertextVector,
        secret_key: SecretKey,
        num_runs: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark decryption operation.

        Args:
            ciphertext: Encrypted data
            secret_key: Secret key
            num_runs: Override default number of runs

        Returns:
            BenchmarkResult
        """
        from he_ml.core.encryptor import decrypt_vector

        def decrypt_single():
            return decrypt_vector(ciphertext, secret_key)

        return self._benchmark_operation(
            decrypt_single,
            num_runs=num_runs or self.benchmark_runs
        )

    def benchmark_inference(
        self,
        model: Any,
        input_data: np.ndarray,
        context: Any,
        keys: Dict[str, Any],
        apply_activations: bool = False,
        num_runs: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark encrypted inference.

        Args:
            model: EncryptedModel instance
            input_data: Input data
            context: TenSEAL context
            keys: Dictionary with secret_key and relin_key
            apply_activations: Whether to apply activations
            num_runs: Override default number of runs

        Returns:
            BenchmarkResult
        """
        def infer_single():
            _, metrics = model.predict(
                np.array([input_data]),
                context,
                keys['secret_key'],
                keys['relin_key'],
                apply_activations=apply_activations
            )
            return metrics.total_time

        return self._benchmark_operation(
            infer_single,
            num_runs=num_runs or self.benchmark_runs
        )

    def benchmark_plaintext_inference(
        self,
        model: Any,
        input_data: np.ndarray,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        activations: List[str],
        num_runs: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark plaintext inference for comparison.

        Args:
            model: Model architecture (for reference)
            input_data: Input data
            weights: List of weight matrices
            biases: List of bias vectors
            activations: List of activation names
            num_runs: Override default number of runs

        Returns:
            BenchmarkResult
        """
        from he_ml.ml_ops.activations import (
            sigmoid_approximation_coeffs,
            tanh_approximation_coeffs,
            relu_approximation_coeffs,
        )

        def infer_plaintext():
            x = input_data.copy()

            # Forward pass through layers
            for i, (W, b, act) in enumerate(zip(weights, biases, activations)):
                # Linear transformation
                x = x @ W.T + b

                # Apply activation
                if act == 'sigmoid':
                    coeffs = sigmoid_approximation_coeffs(degree=5)
                    x = np.polyval(coeffs[::-1], x)
                elif act == 'tanh':
                    coeffs = tanh_approximation_coeffs(degree=5)
                    x = np.polyval(coeffs[::-1], x)
                elif act == 'relu':
                    x = np.maximum(0, x)
                # 'none' means no activation

            return x

        return self._benchmark_operation(
            infer_plaintext,
            num_runs=num_runs or self.benchmark_runs
        )

    def compare_he_vs_plaintext(
        self,
        model: Any,
        input_data: np.ndarray,
        context: Any,
        keys: Dict[str, Any],
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        activations: List[str],
    ) -> ComparisonResult:
        """
        Compare HE vs. plaintext performance.

        Args:
            model: EncryptedModel instance
            input_data: Input data
            context: TenSEAL context
            keys: Dictionary with keys
            weights: List of weight matrices (for plaintext)
            biases: List of bias vectors (for plaintext)
            activations: List of activation names

        Returns:
            ComparisonResult with detailed comparison
        """
        # Benchmark plaintext
        plaintext_result = self.benchmark_plaintext_inference(
            model, input_data, weights, biases, activations
        )

        # Benchmark HE
        he_result = self.benchmark_inference(
            model, input_data, context, keys, apply_activations=False
        )

        # Calculate comparison metrics
        slowdown = he_result.avg_time / plaintext_result.avg_time
        efficiency = 1.0 / slowdown

        # Check feasibility (within 100x slowdown threshold)
        feasible = slowdown < 100

        # Generate recommendation
        if slowdown < 10:
            recommendation = "HE performance is acceptable for this workload"
        elif slowdown < 50:
            recommendation = "HE is usable but consider batching for efficiency"
        elif slowdown < 100:
            recommendation = "HE is slow; consider hybrid HE/TEE approach"
        else:
            recommendation = "HE is too slow; use TEE or plaintext"

        return ComparisonResult(
            operation="inference",
            plaintext_time=plaintext_result.avg_time,
            plaintext_throughput=plaintext_result.throughput,
            he_time=he_result.avg_time,
            he_throughput=he_result.throughput,
            slowdown_factor=slowdown,
            efficiency=efficiency,
            memory_overhead_mb=he_result.memory_mb,
            feasible=feasible,
            recommendation=recommendation,
        )


def generate_benchmark_report(
    results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive benchmark report.

    Args:
        results: List of benchmark results
        comparisons: List of comparison results
        output_path: Optional path to save report

    Returns:
        Report as string
    """
    report = []
    report.append("=" * 70)
    report.append("Homomorphic Encryption ML - Benchmark Report")
    report.append("=" * 70)
    report.append("")

    # Summary statistics
    report.append("## Executive Summary")
    report.append("")

    if comparisons:
        avg_slowdown = np.mean([c.slowdown_factor for c in comparisons])
        report.append(f"Average HE Slowdown: {avg_slowdown:.2f}x")
        report.append(f"Number of Benchmarks: {len(comparisons)}")
        report.append("")

        # Feasibility analysis
        feasible_count = sum(1 for c in comparisons if c.feasible)
        report.append(f"Feasible Workloads: {feasible_count}/{len(comparisons)}")
        report.append("")

    # Detailed benchmark results
    report.append("## Detailed Benchmark Results")
    report.append("")

    for result in results:
        report.append(f"### {result.name}")
        report.append(f"  Operation: {result.operation}")
        report.append(f"  Average Time: {result.avg_time*1000:.4f} ms")
        report.append(f"  Std Dev:      {result.std_time*1000:.4f} ms")
        report.append(f"  Min Time:     {result.min_time*1000:.4f} ms")
        report.append(f"  Max Time:     {result.max_time*1000:.4f} ms")
        report.append(f"  Throughput:   {result.throughput:.2f} ops/sec")
        report.append(f"  Memory:       {result.memory_mb:.2f} MB")
        report.append("")

    # Comparison results
    if comparisons:
        report.append("## HE vs. Plaintext Comparison")
        report.append("")

        for comp in comparisons:
            report.append(f"### {comp.operation}")
            report.append(f"  Plaintext Time:     {comp.plaintext_time*1000:.4f} ms")
            report.append(f"  HE Time:            {comp.he_time*1000:.4f} ms")
            report.append(f"  Slowdown Factor:    {comp.slowdown_factor:.2f}x")
            report.append(f"  Efficiency:         {comp.efficiency*100:.2f}%")
            report.append(f"  Memory Overhead:    {comp.memory_overhead_mb:.2f} MB")
            report.append(f"  Feasible:           {'✓ Yes' if comp.feasible else '✗ No'}")
            report.append(f"  Recommendation:     {comp.recommendation}")
            report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if comparisons:
        report.append("Based on benchmark results:")
        report.append("")

        # Check if any workloads are feasible
        if any(c.feasible for c in comparisons):
            report.append("✓ Some workloads are suitable for pure HE")
            report.append("  - Use for small-scale, privacy-critical applications")
            report.append("  - Consider batch processing for efficiency")
        else:
            report.append("✗ No workloads meet performance thresholds for pure HE")
            report.append("  - Strongly recommend hybrid HE/TEE approach")
            report.append("  - Use HE for initial layers (privacy-critical)")
            report.append("  - Use TEE for deeper layers (performance-critical)")

        report.append("")
        report.append("## Deployment Recommendations")
        report.append("")

        if avg_slowdown > 100:
            report.append("### Hybrid HE/TEE Architecture (HT2ML)")
            report.append("")
            report.append("Given the high overhead (>100x), we recommend:")
            report.append("")
            report.append("1. **Layer 1-2: Homomorphic Encryption**")
            report.append("   - Encrypt input data")
            report.append("   - Process initial layers in encrypted domain")
            report.append("   - Decrypt within secure enclave")
            report.append("")
            report.append("2. **Layer 3+: Trusted Execution Environment**")
            report.append("   - Fast computation on decrypted data")
            report.append("   - Unlimited depth and activations")
            report.append("   - Hardware-based privacy guarantees")
            report.append("")
            report.append("This approach provides:")
            report.append("- ✓ Privacy for sensitive inputs")
            report.append("- ✓ Performance for complex inference")
            report.append("- ✓ Practical for real-world applications")
        else:
            report.append("### Pure HE Architecture")
            report.append("")
            report.append("Performance is acceptable for pure HE deployment:")
            report.append("- ✓ End-to-end encryption")
            report.append("- ✓ Maximum privacy guarantees")
            report.append("- ✓ Simple deployment")

    report.append("")
    report.append("=" * 70)

    report_str = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)

    return report_str


def analyze_scalability(
    benchmark_suite: BenchmarkSuite,
    model: Any,
    context: Any,
    keys: Dict[str, Any],
    input_sizes: List[int],
) -> Dict[str, Any]:
    """
    Analyze how performance scales with input size.

    Args:
        benchmark_suite: BenchmarkSuite instance
        model: Model to test
        context: TenSEAL context
        keys: Dictionary with keys
        input_sizes: List of input sizes to test

    Returns:
        Dictionary with scalability analysis
    """
    results = []

    for size in input_sizes:
        # Generate random input
        input_data = np.random.randn(size)

        # Create simple model matching input size
        from he_ml.inference.pipeline import create_simple_model
        test_model = create_simple_model(
            input_size=size,
            hidden_size=max(2, size // 2),
            output_size=2,
            activation='none',
            seed=42
        )

        # Benchmark
        result = benchmark_suite.benchmark_inference(
            test_model,
            input_data,
            context,
            keys,
            apply_activations=False
        )

        results.append({
            'input_size': size,
            'time_ms': result.avg_time * 1000,
            'throughput': result.throughput,
            'memory_mb': result.memory_mb,
        })

    # Analyze trends
    sizes = np.array([r['input_size'] for r in results])
    times = np.array([r['time_ms'] for r in results])

    # Fit polynomial to estimate scaling
    coeffs = np.polyfit(sizes, times, 2)

    return {
        'results': results,
        'scaling_coefficients': coeffs.tolist(),
        'complexity': 'quadratic' if abs(coeffs[0]) > 1e-6 else 'linear',
    }
