"""
Performance Benchmarking Framework for HT2ML
=============================================

Comprehensive benchmarking suite for measuring:
- Inference latency
- Throughput
- Noise consumption
- Handoff overhead
- Memory usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
import psutil
import os
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    total_time_ms: float
    operations_count: int
    throughput_ops_per_sec: float
    memory_mb: float

    # Component-specific metrics
    he_time_ms: float = 0.0
    tee_time_ms: float = 0.0
    handoff_time_ms: float = 0.0
    num_handoffs: int = 0

    # Noise metrics
    noise_consumed: int = 0
    noise_remaining: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add benchmark result."""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {}

        times = [r.total_time_ms for r in self.results]
        memory = [r.memory_mb for r in self.results]

        return {
            'name': self.name,
            'num_runs': len(self.results),
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'avg_memory_mb': np.mean(memory),
            'total_time_ms': np.sum(times),
        }


class PerformanceBenchmark:
    """
    Performance benchmarking framework.

    Measures inference performance across multiple dimensions.
    """

    def __init__(self):
        """Initialize benchmark framework."""
        self.process = psutil.Process(os.getpid())
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}

    def measure_inference_time(
        self,
        inference_fn: Callable,
        features: np.ndarray,
        num_runs: int = 10,
        warmup_runs: int = 2
    ) -> BenchmarkResult:
        """
        Measure inference execution time.

        Args:
            inference_fn: Function to benchmark
            features: Input features
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not timed)

        Returns:
            BenchmarkResult with timing metrics
        """
        # Warmup
        for _ in range(warmup_runs):
            _ = inference_fn(features)

        # Measure memory before
        memory_before = self.process.memory_info().rss / (1024 * 1024)

        # Benchmark runs
        times = []
        noise_consumed = 0
        noise_remaining = 0
        he_time = 0
        tee_time = 0
        handoff_time = 0
        num_handoffs = 0

        for _ in range(num_runs):
            start = time.perf_counter()
            result = inference_fn(features)
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

            # Extract metrics if available
            if hasattr(result, 'execution_time_ms'):
                he_time = result.he_time_ms
                tee_time = result.tee_time_ms
                handoff_time = result.handoff_time_ms
                num_handoffs = result.num_handoffs
                noise_consumed = result.noise_budget_used
                noise_remaining = result.noise_budget_remaining

        # Measure memory after
        memory_after = self.process.memory_info().rss / (1024 * 1024)
        avg_memory = memory_after - memory_before

        # Calculate statistics
        total_time = np.sum(times)
        avg_time = np.mean(times)
        throughput = 1000 / avg_time  # ops per second

        # Determine name from function
        name = inference_fn.__name__ if hasattr(inference_fn, '__name__') else 'inference'

        return BenchmarkResult(
            name=name,
            total_time_ms=total_time,
            operations_count=num_runs,
            throughput_ops_per_sec=throughput,
            memory_mb=avg_memory,
            he_time_ms=he_time,
            tee_time_ms=tee_time,
            handoff_time_ms=handoff_time,
            num_handoffs=num_handoffs,
            noise_consumed=noise_consumed,
            noise_remaining=noise_remaining
        )

    def measure_throughput(
        self,
        inference_fn: Callable,
        features_batch: np.ndarray,
        duration_seconds: float = 5.0
    ) -> BenchmarkResult:
        """
        Measure sustained throughput.

        Args:
            inference_fn: Function to benchmark
            features_batch: Batch of input features
            duration_seconds: Benchmark duration

        Returns:
            BenchmarkResult with throughput metrics
        """
        start_time = time.time()
        count = 0

        memory_before = self.process.memory_info().rss / (1024 * 1024)

        while (time.time() - start_time) < duration_seconds:
            for features in features_batch:
                _ = inference_fn(features)
                count += 1

                if (time.time() - start_time) >= duration_seconds:
                    break

        elapsed = time.time() - start_time
        memory_after = self.process.memory_info().rss / (1024 * 1024)

        throughput = count / elapsed

        return BenchmarkResult(
            name=f"{inference_fn.__name__}_throughput",
            total_time_ms=elapsed * 1000,
            operations_count=count,
            throughput_ops_per_sec=throughput,
            memory_mb=memory_after - memory_before
        )

    def measure_scalability(
        self,
        create_engine_fn: Callable,
        batch_sizes: List[int],
        feature_dim: int = 50,
        runs_per_size: int = 5
    ) -> BenchmarkSuite:
        """
        Measure scaling with batch size.

        Args:
            create_engine_fn: Function to create inference engine
            batch_sizes: List of batch sizes to test
            feature_dim: Input feature dimension
            runs_per_size: Runs per batch size

        Returns:
            BenchmarkSuite with scalability results
        """
        suite = BenchmarkSuite(name="Scalability")

        for batch_size in batch_sizes:
            # Create engine
            engine = create_engine_fn()

            # Generate batch
            features_batch = np.random.randn(batch_size, feature_dim).astype(np.float32)

            # Measure time for batch
            start = time.perf_counter()

            for i in range(batch_size):
                features = features_batch[i]
                _ = engine.run_inference(features)

            end = time.perf_counter()
            total_time_ms = (end - start) * 1000
            time_per_sample_ms = total_time_ms / batch_size

            result = BenchmarkResult(
                name=f"batch_{batch_size}",
                total_time_ms=total_time_ms,
                operations_count=batch_size,
                throughput_ops_per_sec=1000 / time_per_sample_ms,
                memory_mb=0,
                metadata={
                    'batch_size': batch_size,
                    'time_per_sample_ms': time_per_sample_ms
                }
            )

            suite.add_result(result)

        suite.statistics = suite.get_summary()

        return suite

    def compare_approaches(
        self,
        engines: Dict[str, Any],
        features: np.ndarray,
        num_runs: int = 10
    ) -> BenchmarkSuite:
        """
        Compare different inference approaches.

        Args:
            engines: Dictionary of engine name -> engine object
            features: Input features
            num_runs: Number of benchmark runs

        Returns:
            BenchmarkSuite with comparison results
        """
        suite = BenchmarkSuite(name="Approach Comparison")

        for name, engine in engines.items():
            times = []

            # Warmup
            for _ in range(2):
                _ = engine.run_inference(features)

            # Benchmark
            for _ in range(num_runs):
                start = time.perf_counter()
                result = engine.run_inference(features)
                end = time.perf_counter()

                times.append((end - start) * 1000)

            avg_time = np.mean(times)
            throughput = 1000 / avg_time

            benchmark_result = BenchmarkResult(
                name=name,
                total_time_ms=np.sum(times),
                operations_count=num_runs,
                throughput_ops_per_sec=throughput,
                memory_mb=0,
                he_time_ms=result.he_time_ms,
                tee_time_ms=result.tee_time_ms,
                handoff_time_ms=result.handoff_time_ms,
                num_handoffs=result.num_handoffs,
                noise_consumed=result.noise_budget_used,
                noise_remaining=result.noise_budget_remaining,
                metadata={
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                }
            )

            suite.add_result(benchmark_result)

        suite.statistics = suite.get_summary()

        return suite

    def measure_noise_consumption(
        self,
        engine,
        features: np.ndarray,
        num_inferences: int = 10
    ) -> Dict[str, Any]:
        """
        Measure noise budget consumption over multiple inferences.

        Args:
            engine: HE or Hybrid inference engine
            features: Input features
            num_inferences: Number of inferences to run

        Returns:
            Dictionary with noise consumption statistics
        """
        noise_consumed = []
        noise_remaining = []

        for i in range(num_inferences):
            result = engine.run_inference(features)

            noise_consumed.append(result.noise_budget_used)
            noise_remaining.append(result.noise_budget_remaining)

            # Reset for next inference if applicable
            if hasattr(engine, 'server') and hasattr(engine.server, 'he_engine'):
                engine.server.he_engine.noise_tracker.reset()

        return {
            'avg_noise_consumed': np.mean(noise_consumed),
            'max_noise_consumed': np.max(noise_consumed),
            'min_noise_consumed': np.min(noise_consumed),
            'avg_noise_remaining': np.mean(noise_remaining),
            'total_noise_consumed': np.sum(noise_consumed),
            'inferences_performed': num_inferences,
        }

    def print_benchmark_result(self, result: BenchmarkResult) -> None:
        """Print benchmark result."""
        print(f"\n{'='*70}")
        print(f"Benchmark: {result.name}")
        print(f"{'='*70}")
        print(f"Total time: {result.total_time_ms:.2f}ms")
        print(f"Operations: {result.operations_count}")
        print(f"Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"Memory: {result.memory_mb:.2f} MB")

        if result.he_time_ms > 0 or result.tee_time_ms > 0:
            print(f"\nBreakdown:")
            if result.he_time_ms > 0:
                print(f"  HE time: {result.he_time_ms:.2f}ms")
            if result.tee_time_ms > 0:
                print(f"  TEE time: {result.tee_time_ms:.2f}ms")
            if result.handoff_time_ms > 0:
                print(f"  Handoff time: {result.handoff_time_ms:.2f}ms")
            if result.num_handoffs > 0:
                print(f"  Handoffs: {result.num_handoffs}")

        if result.noise_consumed > 0:
            print(f"\nNoise budget:")
            print(f"  Consumed: {result.noise_consumed} bits")
            print(f"  Remaining: {result.noise_remaining} bits")

        print(f"{'='*70}\n")

    def print_suite_summary(self, suite: BenchmarkSuite) -> None:
        """Print benchmark suite summary."""
        stats = suite.statistics

        print(f"\n{'='*70}")
        print(f"Benchmark Suite: {suite.name}")
        print(f"{'='*70}")
        print(f"Number of runs: {stats['num_runs']}")
        print(f"\nTiming Statistics:")
        print(f"  Average: {stats['avg_time_ms']:.2f}ms")
        print(f"  Min: {stats['min_time_ms']:.2f}ms")
        print(f"  Max: {stats['max_time_ms']:.2f}ms")
        print(f"  Std Dev: {stats['std_time_ms']:.2f}ms")
        print(f"\nMemory:")
        print(f"  Average: {stats['avg_memory_mb']:.2f}MB")
        print(f"  Total time: {stats['total_time_ms']:.2f}ms")
        print(f"{'='*70}\n")

    def export_results_json(self, output_path: str) -> None:
        """
        Export benchmark results to JSON.

        Args:
            output_path: Path to save results
        """
        import json

        results_dict = {}

        for name, suite in self.benchmark_suites.items():
            results_dict[name] = {
                'summary': suite.statistics,
                'results': [
                    {
                        'name': r.name,
                        'total_time_ms': r.total_time_ms,
                        'operations_count': r.operations_count,
                        'throughput_ops_per_sec': r.throughput_ops_per_sec,
                        'memory_mb': r.memory_mb,
                        'he_time_ms': r.he_time_ms,
                        'tee_time_ms': r.tee_time_ms,
                        'handoff_time_ms': r.handoff_time_ms,
                        'num_handoffs': r.num_handoffs,
                        'noise_consumed': r.noise_consumed,
                        'noise_remaining': r.noise_remaining,
                        'metadata': r.metadata,
                    }
                    for r in suite.results
                ]
            }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results exported to: {output_path}")


def create_benchmark() -> PerformanceBenchmark:
    """
    Factory function to create benchmark framework.

    Returns:
        PerformanceBenchmark instance
    """
    return PerformanceBenchmark()
