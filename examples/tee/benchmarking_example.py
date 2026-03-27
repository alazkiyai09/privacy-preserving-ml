"""
TEE Benchmarking Example
=========================

Demonstrates benchmarking TEE operations:
- Benchmarking individual operations
- Comparing TEE vs plaintext
- Analyzing scalability
- Generating performance reports
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave, create_enclave
from tee_ml.benchmarking import (
    create_benchmark,
    create_performance_report,
    create_scalability_report,
    ReportFormat,
    run_standard_benchmark_suite,
)


def benchmark_basic_operations():
    """Benchmark basic operations."""
    print("=" * 70)
    print("Benchmarking Basic Operations")
    print("=" * 70)
    print()

    # Create enclave and benchmark
    enclave = create_enclave(enclave_id="benchmark-example")
    benchmark = create_benchmark(enclave)

    # Define operations to benchmark
    operations = {
        "Addition": lambda x: x + 1,
        "Multiplication": lambda x: x * 2,
        "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "ReLU": lambda x: np.maximum(0, x),
        "Softmax": lambda x: np.exp(x) / np.sum(np.exp(x)),
    }

    print("Benchmarking plaintext operations...")
    results = {}

    for name, op in operations.items():
        result = benchmark.benchmark_plaintext_operation(
            operation=op,
            data_size=1000,
            iterations=100,
            name=f"{name.lower()}_plaintext",
        )
        results[name] = result
        print(f"  {name:15s}: {result.avg_time_ns / 1000:8.2f} μs  "
              f"({result.throughput_ops_per_sec:8.0f} ops/sec)")

    print()
    return results


def benchmark_tee_overhead():
    """Benchmark TEE overhead."""
    print("=" * 70)
    print("Benchmarking TEE Overhead")
    print("=" * 70)
    print()

    enclave = create_enclave(enclave_id="overhead-benchmark")
    benchmark = create_benchmark(enclave)

    # Benchmark enclave entry/exit overhead
    print("Measuring enclave entry/exit overhead...")
    overhead_result = benchmark.benchmark_enclave_entry_exit(
        data_size=1000,
        iterations=100,
    )

    print(f"  Entry/Exit Time: {overhead_result.avg_time_ns / 1000:.2f} μs")
    print(f"  Throughput: {overhead_result.throughput_ops_per_sec:.2f} ops/sec")
    print()

    # Compare plaintext vs TEE
    print("\nComparing Plaintext vs TEE:")

    def plaintext_op(data):
        return data * 2 + 1

    def tee_op(data, session):
        return session.execute(lambda arr: data * 2 + 1)

    comparison = benchmark.benchmark_tee_vs_plaintext(
        operation=plaintext_op,
        tee_operation=tee_op,
        data_size=1000,
        iterations=50,
        name="arithmetic_comparison",
    )

    print(f"  Plaintext Time: {comparison.baseline_avg_ns / 1000:.2f} μs")
    print(f"  TEE Time: {comparison.comparison_avg_ns / 1000:.2f} μs")
    print(f"  Slowdown: {comparison.slowdown_factor:.2f}x")
    print(f"  {comparison.conclusion}")
    print()


def benchmark_scalability():
    """Benchmark scalability with input size."""
    print("=" * 70)
    print("Scalability Analysis")
    print("=" * 70)
    print()

    enclave = create_enclave(enclave_id="scalability-benchmark")
    benchmark = create_benchmark(enclave)

    # Test scalability
    print("Testing scalability with different input sizes...")

    scalability_results = benchmark.benchmark_scalability(
        operation=lambda x: x * 2 + 1,
        data_sizes=[100, 500, 1000, 5000, 10000],
        iterations_per_size=50,
        name="scalability_test",
    )

    print(f"\n{'Size':<10} {'Avg Time':<15} {'Throughput':<15} {'Time/Element':<15}")
    print("-" * 70)

    for result in scalability_results:
        size = result.metadata.get('data_size', 0)
        avg_time_us = result.avg_time_ns / 1000
        throughput = result.throughput_ops_per_sec
        time_per_element_ns = result.avg_time_ns / size if size > 0 else 0

        print(f"{size:<10} {avg_time_us:<15.2f} {throughput:<15.2f} {time_per_element_ns:<15.2f}")

    print()

    # Analyze scaling
    first = scalability_results[0]
    last = scalability_results[-1]

    first_size = first.metadata.get('data_size', 1)
    last_size = last.metadata.get('data_size', 1)

    size_ratio = last_size / first_size
    time_ratio = last.avg_time_ns / first.avg_time_ns

    print("Scaling Analysis:")
    print(f"  Size increase: {size_ratio:.2f}x")
    print(f"  Time increase: {time_ratio:.2f}x")

    if time_ratio < size_ratio * 1.2:
        scaling = "Sub-linear (efficient)"
    elif time_ratio < size_ratio * 1.5:
        scaling = "Linear (expected)"
    else:
        scaling = "Super-linear (inefficient)"

    print(f"  Scaling pattern: {scaling}")
    print()


def generate_performance_report():
    """Generate comprehensive performance report."""
    print("=" * 70)
    print("Generating Performance Report")
    print("=" * 70)
    print()

    enclave = create_enclave(enclave_id="report-example")
    benchmark = create_benchmark(enclave)

    # Run benchmarks
    print("Running benchmarks...")

    # 1. Plaintext operations
    for op_name, op_func in [
        ("add", lambda x: x + 1),
        ("multiply", lambda x: x * 2),
        ("sigmoid", lambda x: 1 / (1 + np.exp(-x))),
    ]:
        benchmark.benchmark_plaintext_operation(
            operation=op_func,
            data_size=1000,
            iterations=50,
            name=f"{op_name}_plaintext",
        )

    # 2. TEE operations
    def tee_add(data, session):
        return session.execute(lambda arr: data + 1)

    benchmark.benchmark_tee_operation(
        operation=tee_add,
        data_size=1000,
        iterations=50,
        name="add_tee",
    )

    # 3. Comparison
    def plaintext_op(data):
        return data + 1

    comparison = benchmark.benchmark_tee_vs_plaintext(
        operation=plaintext_op,
        tee_operation=tee_add,
        data_size=1000,
        iterations=50,
        name="add_comparison",
    )

    # 4. Scalability
    scalability_results = benchmark.benchmark_scalability(
        operation=lambda x: x * 2,
        data_sizes=[100, 500, 1000],
        iterations_per_size=30,
        name="scale_test",
    )

    # Create report
    print("Creating performance report...")

    report = create_performance_report(
        title="TEE Performance Analysis Report",
        author="Benchmark Suite",
    )

    # Add all results
    for result in benchmark.results:
        report.add_benchmark_result(result)

    report.add_comparison_result(comparison)

    report.set_metadata(
        test_type="comprehensive",
        total_benchmarks=len(benchmark.results),
        test_date="2025-01-29",
    )

    # Generate reports in different formats
    print("\n" + "=" * 70)
    print("TEXT REPORT")
    print("=" * 70)
    text_report = report.generate_summary(ReportFormat.TEXT)
    print(text_report)

    print("\n" + "=" * 70)
    print("MARKDOWN REPORT")
    print("=" * 70)
    md_report = report.generate_summary(ReportFormat.MARKDOWN)
    print(md_report)

    # Save reports
    output_dir = Path("/tmp/tee_benchmark_reports")
    output_dir.mkdir(exist_ok=True)

    report.save_report(output_dir / "performance_report.txt", ReportFormat.TEXT)
    report.save_report(output_dir / "performance_report.md", ReportFormat.MARKDOWN)
    report.save_detailed_analysis(output_dir / "detailed_analysis.txt")

    print(f"\n✓ Reports saved to: {output_dir}")
    print()


def run_standard_suite():
    """Run the standard benchmark suite."""
    print("=" * 70)
    print("Running Standard Benchmark Suite")
    print("=" * 70)
    print()

    enclave = create_enclave(enclave_id="standard-suite")
    print("Running comprehensive benchmark suite...")
    print("This may take a moment...\n")

    results = run_standard_benchmark_suite(
        enclave=enclave,
        data_size=1000,
        iterations=50,
    )

    # Display summary
    print("Benchmark Suite Complete!")
    print()

    # Enclave overhead
    overhead = results['results']['enclave_overhead']
    print(f"Enclave Overhead:")
    print(f"  Average time: {overhead.avg_time_ns / 1000:.2f} μs")
    print()

    # Operations summary
    print("Operations Summary:")
    for op_name, op_result in results['results']['operations'].items():
        print(f"  {op_name}: {op_result.avg_time_ns / 1000:.2f} μs")
    print()

    # TEE vs HE comparison
    tee_vs_he = results['results']['tee_vs_he']
    print("TEE vs HE Comparison:")
    print(f"  {tee_vs_he.conclusion}")
    print()

    # Overall summary
    print("Overall Summary:")
    print(f"  Total benchmarks: {results['summary']['total_benchmarks']}")
    print(f"  Average time: {results['summary']['avg_time_ns'] / 1000:.2f} μs")
    print()


def compare_ml_operations():
    """Compare different ML operations."""
    print("=" * 70)
    print("Comparing ML Operations Performance")
    print("=" * 70)
    print()

    from tee_ml.operations.activations import (
        tee_relu,
        tee_sigmoid,
        tee_softmax,
    )
    from tee_ml.operations.comparisons import tee_argmax

    enclave = create_enclave(enclave_id="ml-ops-benchmark")
    benchmark = create_benchmark(enclave)

    operations = {
        "ReLU": (lambda x: np.maximum(0, x),
                lambda x, s: tee_relu(x, s)),
        "Sigmoid": (lambda x: 1 / (1 + np.exp(-x)),
                   lambda x, s: tee_sigmoid(x, s)),
        "Argmax": (lambda x: np.argmax(x),
                  lambda x, s: tee_argmax(x, s)),
    }

    print("Comparing ML operations:\n")
    print(f"{'Operation':<15} {'Plaintext':<15} {'TEE':<15} {'Slowdown':<10}")
    print("-" * 70)

    for op_name, (plaintext_op, tee_op) in operations.items():
        data = np.random.randn(100)

        # Benchmark plaintext
        plaintext_result = benchmark.benchmark_function(
            func=plaintext_op,
            func_args=(data,),
            iterations=50,
            name=f"{op_name.lower()}_plaintext",
        )

        # Benchmark TEE
        tee_result = benchmark.benchmark_tee_operation(
            operation=tee_op,
            data_size=len(data),
            iterations=50,
            name=f"{op_name.lower()}_tee",
        )

        # Calculate slowdown
        slowdown = tee_result.avg_time_ns / plaintext_result.avg_time_ns

        print(f"{op_name:<15} "
              f"{plaintext_result.avg_time_ns / 1000:<15.2f} "
              f"{tee_result.avg_time_ns / 1000:<15.2f} "
              f"{slowdown:<10.2f}x")

    print()


def main():
    """Run all benchmarking examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "TEE ML Framework - Benchmarking Examples" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    try:
        # Run examples
        benchmark_basic_operations()
        benchmark_tee_overhead()
        benchmark_scalability()
        generate_performance_report()
        run_standard_suite()
        compare_ml_operations()

        print("\n")
        print("✓ All benchmarking examples completed successfully!")
        print("\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
