"""
Comprehensive Benchmark Runner for HT2ML
==========================================

Runs comprehensive benchmarks comparing:
- HE-only vs TEE-only vs Hybrid inference
- Latency and throughput
- Noise consumption
- Scalability with batch size
- Memory usage
"""

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from pathlib import Path

from benchmarks.performance_benchmark import (
    PerformanceBenchmark,
    BenchmarkResult,
    BenchmarkSuite,
    create_benchmark,
)
from src.inference.hybrid_engine import create_hybrid_engine
from src.inference.he_only_engine import create_he_only_engine
from src.inference.tee_only_engine import create_tee_only_engine
from config.model_config import create_default_config
from src.model.phishing_classifier import create_random_model


class HT2MLBenchmarkRunner:
    """
    Comprehensive benchmark runner for HT2ML.

    Runs all benchmarks and generates reports.
    """

    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize benchmark runner."""
        self.benchmark = create_benchmark()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create model
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)

        # Generate test data
        self.test_features = np.random.randn(50).astype(np.float32)

    def run_latency_benchmark(self, num_runs: int = 20) -> BenchmarkSuite:
        """
        Benchmark inference latency.

        Args:
            num_runs: Number of benchmark runs

        Returns:
            BenchmarkSuite with latency results
        """
        print(f"\n{'#'*70}")
        print(f"# LATENCY BENCHMARK")
        print(f"# Running {num_runs} iterations per approach")
        print(f"{'#'*70}")

        suite = BenchmarkSuite(name="Latency")

        # Create engines
        engines = {
            'TEE-only': create_tee_only_engine(self.model),
            'Hybrid': create_hybrid_engine(self.model),
            'HE-only': create_he_only_engine(self.model),
        }

        # Benchmark each approach
        for name, engine in engines.items():
            print(f"\nBenchmarking {name}...")

            times = []
            he_times = []
            tee_times = []
            handoff_times = []
            num_handoffs_list = []
            noise_used_list = []

            # Warmup
            for _ in range(3):
                _ = engine.run_inference(self.test_features)
                # Reset after each warmup run
                if hasattr(engine, 'server') and hasattr(engine.server, 'he_engine'):
                    engine.server.he_engine.noise_tracker.reset()
                elif hasattr(engine, 'he_engine'):
                    engine.he_engine.noise_tracker.reset()

            # Benchmark
            for i in range(num_runs):
                start = time.perf_counter()
                result = engine.run_inference(self.test_features)
                end = time.perf_counter()

                times.append((end - start) * 1000)
                he_times.append(result.he_time_ms)
                tee_times.append(result.tee_time_ms)
                handoff_times.append(result.handoff_time_ms)
                num_handoffs_list.append(result.num_handoffs)
                noise_used_list.append(result.noise_budget_used)

                # Reset noise tracker for next run (after every run, including last)
                # This ensures the next approach starts fresh
                if hasattr(engine, 'server') and hasattr(engine.server, 'he_engine'):
                    engine.server.he_engine.noise_tracker.reset()
                elif hasattr(engine, 'he_engine'):
                    engine.he_engine.noise_tracker.reset()

            # Create result
            avg_time = np.mean(times)
            throughput = 1000 / avg_time

            result = BenchmarkResult(
                name=name,
                total_time_ms=np.sum(times),
                operations_count=num_runs,
                throughput_ops_per_sec=throughput,
                memory_mb=0,
                he_time_ms=np.mean(he_times),
                tee_time_ms=np.mean(tee_times),
                handoff_time_ms=np.mean(handoff_times),
                num_handoffs=int(np.mean(num_handoffs_list)),
                noise_consumed=int(np.mean(noise_used_list)),
                metadata={
                    'std_time_ms': float(np.std(times)),
                    'min_time_ms': float(np.min(times)),
                    'max_time_ms': float(np.max(times)),
                    'p50_time_ms': float(np.percentile(times, 50)),
                    'p95_time_ms': float(np.percentile(times, 95)),
                    'p99_time_ms': float(np.percentile(times, 99)),
                }
            )

            suite.add_result(result)
            self.benchmark.print_benchmark_result(result)

        suite.statistics = suite.get_summary()

        return suite

    def run_throughput_benchmark(self, duration_seconds: float = 10.0) -> BenchmarkSuite:
        """
        Benchmark sustained throughput.

        Args:
            duration_seconds: Benchmark duration

        Returns:
            BenchmarkSuite with throughput results
        """
        print(f"\n{'#'*70}")
        print(f"# THROUGHPUT BENCHMARK")
        print(f"# Duration: {duration_seconds} seconds per approach")
        print(f"{'#'*70}")

        suite = BenchmarkSuite(name="Throughput")

        approaches = [
            ('TEE-only', lambda: create_tee_only_engine(self.model)),
            ('Hybrid', lambda: create_hybrid_engine(self.model)),
            ('HE-only', lambda: create_he_only_engine(self.model)),
        ]

        for name, create_engine in approaches:
            print(f"\nBenchmarking {name}...")

            engine = create_engine()

            # Warmup
            for _ in range(3):
                _ = engine.run_inference(self.test_features)

            # Measure throughput
            count = 0
            start_time = time.time()

            while (time.time() - start_time) < duration_seconds:
                _ = engine.run_inference(self.test_features)
                count += 1

                # Reset for HE engines to avoid noise exhaustion
                if hasattr(engine, 'server') and hasattr(engine.server, 'he_engine'):
                    engine.server.he_engine.noise_tracker.reset()
                elif hasattr(engine, 'he_engine'):
                    engine.he_engine.noise_tracker.reset()

            elapsed = time.time() - start_time
            throughput = count / elapsed

            result = BenchmarkResult(
                name=name,
                total_time_ms=elapsed * 1000,
                operations_count=count,
                throughput_ops_per_sec=throughput,
                memory_mb=0,
            )

            suite.add_result(result)

            print(f"\n{name}:")
            print(f"  Operations: {count}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.2f} ops/sec")

        suite.statistics = suite.get_summary()

        return suite

    def run_scalability_benchmark(self, batch_sizes: list = None) -> BenchmarkSuite:
        """
        Benchmark scalability with batch size.

        Args:
            batch_sizes: List of batch sizes to test

        Returns:
            BenchmarkSuite with scalability results
        """
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20, 50, 100]

        print(f"\n{'#'*70}")
        print(f"# SCALABILITY BENCHMARK")
        print(f"# Batch sizes: {batch_sizes}")
        print(f"{'#'*70}")

        suite = BenchmarkSuite(name="Scalability")

        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch size {batch_size}...")

            # Create TEE-only engine (fastest for scalability test)
            engine = create_tee_only_engine(self.model)

            # Generate batch
            features_batch = np.random.randn(batch_size, 50).astype(np.float32)

            # Measure time
            start = time.perf_counter()

            for i in range(batch_size):
                _ = engine.run_inference(features_batch[i])

            end = time.perf_counter()
            total_time_ms = (end - start) * 1000
            time_per_sample_ms = total_time_ms / batch_size
            throughput = 1000 / time_per_sample_ms

            result = BenchmarkResult(
                name=f"batch_{batch_size}",
                total_time_ms=total_time_ms,
                operations_count=batch_size,
                throughput_ops_per_sec=throughput,
                memory_mb=0,
                metadata={
                    'batch_size': batch_size,
                    'time_per_sample_ms': time_per_sample_ms,
                }
            )

            suite.add_result(result)

            print(f"  Total time: {total_time_ms:.2f}ms")
            print(f"  Time per sample: {time_per_sample_ms:.3f}ms")
            print(f"  Throughput: {throughput:.2f} ops/sec")

        suite.statistics = suite.get_summary()

        return suite

    def run_noise_benchmark(self, num_inferences: int = 10) -> dict:
        """
        Benchmark noise budget consumption.

        Args:
            num_inferences: Number of inferences per approach

        Returns:
            Dictionary with noise consumption statistics
        """
        print(f"\n{'#'*70}")
        print(f"# NOISE CONSUMPTION BENCHMARK")
        print(f"# Running {num_inferences} inferences per approach")
        print(f"{'#'*70}")

        noise_stats = {}

        # HE-only
        print("\nBenchmarking HE-only...")
        he_engine = create_he_only_engine(self.model)
        noise_stats['HE-only'] = self.benchmark.measure_noise_consumption(
            he_engine, self.test_features, num_inferences
        )

        # Hybrid
        print("Benchmarking Hybrid...")
        hybrid_engine = create_hybrid_engine(self.model)
        noise_stats['Hybrid'] = self.benchmark.measure_noise_consumption(
            hybrid_engine, self.test_features, num_inferences
        )

        # TEE-only (no noise)
        print("Benchmarking TEE-only...")
        tee_engine = create_tee_only_engine(self.model)
        _ = tee_engine.run_inference(self.test_features)
        noise_stats['TEE-only'] = {
            'avg_noise_consumed': 0,
            'max_noise_consumed': 0,
            'min_noise_consumed': 0,
            'avg_noise_remaining': 200,
            'total_noise_consumed': 0,
            'inferences_performed': 1,
        }

        # Print results
        print("\n" + "="*70)
        print("NOISE CONSUMPTION SUMMARY")
        print("="*70)

        for name, stats in noise_stats.items():
            print(f"\n{name}:")
            print(f"  Avg noise per inference: {stats['avg_noise_consumed']:.1f} bits")
            print(f"  Total noise consumed: {stats['total_noise_consumed']} bits")
            if stats['avg_noise_remaining'] > 0:
                print(f"  Avg remaining: {stats['avg_noise_remaining']:.1f} bits")

        return noise_stats

    def generate_comparison_report(self) -> dict:
        """
        Generate comprehensive comparison report.

        Returns:
            Dictionary with all benchmark results
        """
        print(f"\n{'#'*70}")
        print(f"# COMPREHENSIVE BENCHMARK REPORT")
        print(f"# HT2ML: Hybrid HE/TEE Phishing Detection")
        print(f"{'#'*70}")

        # Run all benchmarks
        latency_suite = self.run_latency_benchmark(num_runs=20)
        throughput_suite = self.run_throughput_benchmark(duration_seconds=5.0)
        scalability_suite = self.run_scalability_benchmark()
        noise_stats = self.run_noise_benchmark(num_inferences=10)

        # Compile report
        report = {
            'model_info': {
                'name': 'HT2ML Phishing Classifier',
                'input_size': 50,
                'hidden_size': 64,
                'output_size': 2,
                'parameters': self.model.get_num_parameters(),
            },
            'latency': latency_suite.statistics,
            'throughput': throughput_suite.statistics,
            'scalability': scalability_suite.statistics,
            'noise_consumption': noise_stats,
        }

        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Benchmark report saved to: {report_path}")
        print(f"{'='*70}")

        return report

    def print_final_summary(self, report: dict) -> None:
        """Print final benchmark summary."""
        print(f"\n{'#'*70}")
        print(f"# FINAL BENCHMARK SUMMARY")
        print(f"{'#'*70}")

        # Model info
        info = report['model_info']
        print(f"\nModel Configuration:")
        print(f"  Input features: {info['input_size']}")
        print(f"  Hidden size: {info['hidden_size']}")
        print(f"  Output classes: {info['output_size']}")
        print(f"  Total parameters: {info['parameters']:,}")

        # Latency comparison
        print(f"\nLatency Comparison (ms):")
        for result in report['latency']['results']:
            name = result['name']
            avg = result['total_time_ms'] / result['operations_count']
            print(f"  {name:<12} {avg:>8.2f} ms")

        # Throughput comparison
        print(f"\nThroughput Comparison (ops/sec):")
        for result in report['throughput']['results']:
            name = result['name']
            ops = result['throughput_ops_per_sec']
            print(f"  {name:<12} {ops:>10.2f} ops/sec")

        # Calculate speedup
        tee_time = next(r['total_time_ms'] / r['operations_count'] for r in report['latency']['results'] if r['name'] == 'TEE-only')
        he_time = next(r['total_time_ms'] / r['operations_count'] for r in report['latency']['results'] if r['name'] == 'HE-only')
        hybrid_time = next(r['total_time_ms'] / r['operations_count'] for r in report['latency']['results'] if r['name'] == 'Hybrid')

        print(f"\nSpeedup Analysis:")
        print(f"  TEE-only vs HE-only: {he_time/tee_time:.1f}x faster")
        print(f"  Hybrid vs HE-only: {he_time/hybrid_time:.1f}x faster")
        print(f"  TEE-only vs Hybrid: {hybrid_time/tee_time:.1f}x faster")

        print(f"\n{'#'*70}\n")


def main():
    """Run comprehensive benchmarks."""
    runner = HT2MLBenchmarkRunner()

    # Generate report
    report = runner.generate_comparison_report()

    # Print summary
    runner.print_final_summary(report)

    return 0


if __name__ == "__main__":
    import time

    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time

    print(f"\nTotal benchmark time: {elapsed:.2f} seconds")

    sys.exit(exit_code)
