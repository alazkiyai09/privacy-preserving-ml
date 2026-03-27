"""
TEE Performance Benchmarks
===========================

Comprehensive benchmarking framework for TEE operations.

Benchmarks:
- TEE vs Plaintext Performance
- TEE vs HE Performance
- Enclave Overhead Analysis
- Operation-Specific Benchmarks
- Scalability Studies
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from tee_ml.core.enclave import Enclave
from tee_ml.simulation.overhead import OverheadModel, OverheadSimulator


class BenchmarkType(Enum):
    """Type of benchmark."""

    TEE_VS_PLAINTEXT = "tee_vs_plaintext"
    """Compare TEE performance vs plaintext execution"""

    TEE_VS_HE = "tee_vs_he"
    """Compare TEE performance vs homomorphic encryption"""

    ENCLAVE_OVERHEAD = "enclave_overhead"
    """Measure enclave entry/exit overhead"""

    OPERATION_SPECIFIC = "operation_specific"
    """Benchmark specific operations"""

    SCALABILITY = "scalability"
    """Test scalability with input size"""


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    benchmark_type: BenchmarkType
    iterations: int
    total_time_ns: float
    avg_time_ns: float
    min_time_ns: float
    max_time_ns: float
    std_time_ns: float
    throughput_ops_per_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'benchmark_type': self.benchmark_type.value,
            'iterations': int(self.iterations),
            'total_time_ns': float(self.total_time_ns),
            'avg_time_ns': float(self.avg_time_ns),
            'min_time_ns': float(self.min_time_ns),
            'max_time_ns': float(self.max_time_ns),
            'std_time_ns': float(self.std_time_ns),
            'throughput_ops_per_sec': float(self.throughput_ops_per_sec),
            'metadata': self.metadata,
        }

    def get_slowdown_factor(self, baseline_time_ns: float) -> float:
        """Calculate slowdown factor compared to baseline."""
        if baseline_time_ns == 0:
            return float('inf')
        return self.avg_time_ns / baseline_time_ns


@dataclass
class ComparisonResult:
    """Result of comparing two benchmarks."""

    name: str
    baseline_name: str
    baseline_avg_ns: float
    comparison_avg_ns: float
    slowdown_factor: float
    speedup_factor: float
    percentage_difference: float
    conclusion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'baseline_name': self.baseline_name,
            'baseline_avg_ns': float(self.baseline_avg_ns),
            'comparison_avg_ns': float(self.comparison_avg_ns),
            'slowdown_factor': float(self.slowdown_factor),
            'speedup_factor': float(self.speedup_factor),
            'percentage_difference': float(self.percentage_difference),
            'conclusion': self.conclusion,
        }


class TEEBenchmark:
    """
    TEE performance benchmarking framework.

    Benchmarks TEE operations against plaintext and HE baselines.
    """

    def __init__(
        self,
        enclave: Enclave,
        overhead_model: Optional[OverheadModel] = None,
    ):
        """
        Initialize benchmark framework.

        Args:
            enclave: TEE enclave to benchmark
            overhead_model: Optional overhead model
        """
        self.enclave = enclave
        self.overhead_model = overhead_model or OverheadModel()
        self.simulator = OverheadSimulator(self.overhead_model)
        self.results: List[BenchmarkResult] = []

    def benchmark_function(
        self,
        func: Callable,
        func_args: Tuple = (),
        func_kwargs: Dict = None,
        iterations: int = 100,
        warmup_iterations: int = 10,
        name: str = "benchmark",
        benchmark_type: BenchmarkType = BenchmarkType.OPERATION_SPECIFIC,
        metadata: Dict = None,
    ) -> BenchmarkResult:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            func_args: Positional arguments for function
            func_kwargs: Keyword arguments for function
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not timed)
            name: Benchmark name
            benchmark_type: Type of benchmark
            metadata: Additional metadata

        Returns:
            BenchmarkResult with timing statistics
        """
        if func_kwargs is None:
            func_kwargs = {}
        if metadata is None:
            metadata = {}

        # Warmup
        for _ in range(warmup_iterations):
            func(*func_args, **func_kwargs)

        # Benchmark
        times_ns = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = func(*func_args, **func_kwargs)
            end = time.perf_counter_ns()
            times_ns.append(end - start)

        times_array = np.array(times_ns)

        result = BenchmarkResult(
            name=name,
            benchmark_type=benchmark_type,
            iterations=iterations,
            total_time_ns=np.sum(times_array),
            avg_time_ns=np.mean(times_array),
            min_time_ns=np.min(times_array),
            max_time_ns=np.max(times_array),
            std_time_ns=np.std(times_array),
            throughput_ops_per_sec=1e9 / np.mean(times_array),
            metadata=metadata,
        )

        self.results.append(result)
        return result

    def benchmark_plaintext_operation(
        self,
        operation: Callable,
        data_size: int = 1000,
        iterations: int = 100,
        name: str = "plaintext_op",
    ) -> BenchmarkResult:
        """
        Benchmark plaintext operation.

        Args:
            operation: Operation to benchmark (takes numpy array)
            data_size: Size of input data
            iterations: Number of iterations
            name: Benchmark name

        Returns:
            BenchmarkResult
        """
        data = np.random.randn(data_size)

        return self.benchmark_function(
            func=operation,
            func_args=(data,),
            iterations=iterations,
            name=name,
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            metadata={'data_size': data_size, 'execution': 'plaintext'},
        )

    def benchmark_tee_operation(
        self,
        operation: Callable,
        data_size: int = 1000,
        iterations: int = 100,
        name: str = "tee_op",
    ) -> BenchmarkResult:
        """
        Benchmark TEE operation.

        Args:
            operation: TEE operation to benchmark (takes array and session)
            data_size: Size of input data
            iterations: Number of iterations
            name: Benchmark name

        Returns:
            BenchmarkResult
        """
        data = np.random.randn(data_size)

        def tee_wrapper():
            # Wrapper that doesn't take parameters
            session = self.enclave.enter(data)
            try:
                result = operation(data, session)
            finally:
                self.enclave.exit(session)
            return result

        return self.benchmark_function(
            func=tee_wrapper,
            func_args=(),
            iterations=iterations,
            name=name,
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            metadata={'data_size': data_size, 'execution': 'tee'},
        )

    def benchmark_enclave_entry_exit(
        self,
        data_size: int = 1000,
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Benchmark enclave entry/exit overhead.

        Args:
            data_size: Size of data to pass to enclave
            iterations: Number of iterations

        Returns:
            BenchmarkResult with overhead timing
        """
        data = np.random.randn(data_size)

        def entry_exit_wrapper():
            session = self.enclave.enter(data)
            self.enclave.exit(session)

        return self.benchmark_function(
            func=entry_exit_wrapper,
            iterations=iterations,
            name="enclave_entry_exit",
            benchmark_type=BenchmarkType.ENCLAVE_OVERHEAD,
            metadata={'data_size': data_size},
        )

    def benchmark_tee_vs_plaintext(
        self,
        operation: Callable,
        tee_operation: Callable,
        data_size: int = 1000,
        iterations: int = 100,
        name: str = "comparison",
    ) -> ComparisonResult:
        """
        Benchmark TEE vs plaintext performance.

        Args:
            operation: Plaintext operation
            tee_operation: TEE operation
            data_size: Size of input data
            iterations: Number of iterations
            name: Benchmark name

        Returns:
            ComparisonResult
        """
        # Benchmark plaintext
        plaintext_result = self.benchmark_plaintext_operation(
            operation=operation,
            data_size=data_size,
            iterations=iterations,
            name=f"{name}_plaintext",
        )

        # Benchmark TEE
        tee_result = self.benchmark_tee_operation(
            operation=tee_operation,
            data_size=data_size,
            iterations=iterations,
            name=f"{name}_tee",
        )

        # Calculate comparison
        slowdown = tee_result.avg_time_ns / plaintext_result.avg_time_ns
        speedup = 1.0 / slowdown if slowdown > 0 else 0
        pct_diff = ((tee_result.avg_time_ns - plaintext_result.avg_time_ns) /
                   plaintext_result.avg_time_ns * 100)

        # Generate conclusion
        if slowdown < 1.1:
            conclusion = f"TEE is nearly as fast as plaintext ({slowdown:.2f}x slowdown)"
        elif slowdown < 2.0:
            conclusion = f"TEE has moderate overhead ({slowdown:.2f}x slowdown)"
        elif slowdown < 10.0:
            conclusion = f"TEE has significant overhead ({slowdown:.2f}x slowdown)"
        else:
            conclusion = f"TEE has very high overhead ({slowdown:.2f}x slowdown)"

        return ComparisonResult(
            name=name,
            baseline_name="plaintext",
            baseline_avg_ns=plaintext_result.avg_time_ns,
            comparison_avg_ns=tee_result.avg_time_ns,
            slowdown_factor=slowdown,
            speedup_factor=speedup,
            percentage_difference=pct_diff,
            conclusion=conclusion,
        )

    def benchmark_scalability(
        self,
        operation: Callable,
        data_sizes: List[int] = None,
        iterations_per_size: int = 50,
        name: str = "scalability",
    ) -> List[BenchmarkResult]:
        """
        Benchmark scalability with input size.

        Args:
            operation: Operation to benchmark
            data_sizes: List of input sizes to test
            iterations_per_size: Iterations per data size
            name: Benchmark name

        Returns:
            List of BenchmarkResults, one per data size
        """
        if data_sizes is None:
            data_sizes = [100, 500, 1000, 5000, 10000]

        results = []

        for size in data_sizes:
            result = self.benchmark_plaintext_operation(
                operation=operation,
                data_size=size,
                iterations=iterations_per_size,
                name=f"{name}_size_{size}",
            )
            result.metadata['data_size'] = size
            results.append(result)

        return results

    def benchmark_memory_scalability(
        self,
        data_sizes_mb: List[float] = None,
        iterations: int = 50,
    ) -> List[BenchmarkResult]:
        """
        Benchmark memory scalability.

        Args:
            data_sizes_mb: List of data sizes in MB
            iterations: Number of iterations

        Returns:
            List of BenchmarkResults
        """
        if data_sizes_mb is None:
            data_sizes_mb = [0.1, 0.5, 1.0, 5.0, 10.0]

        results = []

        for size_mb in data_sizes_mb:
            # Create data of approximately the right size
            num_elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64

            def memory_op(data):
                # Simple memory operation
                return data.sum()

            result = self.benchmark_plaintext_operation(
                operation=memory_op,
                data_size=num_elements,
                iterations=iterations,
                name=f"memory_{size_mb:.1f}mb",
            )
            result.metadata['data_size_mb'] = size_mb
            results.append(result)

        return results

    def estimate_he_performance(
        self,
        data_size: int,
        he_slowdown_factor: float = 100.0,
    ) -> float:
        """
        Estimate HE performance based on plaintext baseline.

        Args:
            data_size: Input data size
            he_slowdown_factor: Expected HE slowdown (default 100x)

        Returns:
            Estimated HE time in nanoseconds
        """
        # Get plaintext baseline
        def simple_op(data):
            return data * 2 + 1

        plaintext_result = self.benchmark_plaintext_operation(
            operation=simple_op,
            data_size=data_size,
            iterations=10,
            name="he_baseline",
        )

        return plaintext_result.avg_time_ns * he_slowdown_factor

    def compare_tee_vs_he(
        self,
        data_size: int = 1000,
        he_slowdown_factor: float = 100.0,
        name: str = "tee_vs_he",
    ) -> ComparisonResult:
        """
        Compare TEE vs HE performance.

        Args:
            data_size: Input data size
            he_slowdown_factor: Expected HE slowdown
            name: Comparison name

        Returns:
            ComparisonResult
        """
        # Benchmark TEE
        def simple_op(data):
            return data * 2 + 1

        def tee_simple_op(data, session):
            return session.execute(lambda arr: data * 2 + 1)

        tee_result = self.benchmark_tee_operation(
            operation=tee_simple_op,
            data_size=data_size,
            iterations=50,
            name=f"{name}_tee",
        )

        # Estimate HE performance
        he_time_ns = self.estimate_he_performance(
            data_size=data_size,
            he_slowdown_factor=he_slowdown_factor,
        )

        # Calculate comparison
        speedup = he_time_ns / tee_result.avg_time_ns
        slowdown = 1.0 / speedup if speedup > 0 else 0
        pct_diff = ((tee_result.avg_time_ns - he_time_ns) / he_time_ns * 100)

        # Generate conclusion
        if speedup > 50:
            conclusion = f"TEE is {speedup:.0f}x faster than HE"
        elif speedup > 10:
            conclusion = f"TEE is {speedup:.0f}x faster than HE"
        elif speedup > 2:
            conclusion = f"TEE is {speedup:.1f}x faster than HE"
        else:
            conclusion = f"TEE has similar performance to HE"

        return ComparisonResult(
            name=name,
            baseline_name="he",
            baseline_avg_ns=he_time_ns,
            comparison_avg_ns=tee_result.avg_time_ns,
            slowdown_factor=slowdown,
            speedup_factor=speedup,
            percentage_difference=pct_diff,
            conclusion=conclusion,
        )

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmark results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {'total_benchmarks': 0}

        return {
            'total_benchmarks': len(self.results),
            'benchmark_types': {
                bt.value: sum(1 for r in self.results if r.benchmark_type == bt)
                for bt in BenchmarkType
            },
            'avg_time_ns': float(np.mean([r.avg_time_ns for r in self.results])),
            'total_time_ns': float(sum(r.total_time_ns for r in self.results)),
        }

    def save_results(self, filepath: str):
        """
        Save benchmark results to file.

        Args:
            filepath: Path to save results
        """
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'enclave_id': self.enclave.enclave_id,
            'results': [r.to_dict() for r in self.results],
            'summary': self.get_results_summary(),
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def load_results(self, filepath: str):
        """
        Load benchmark results from file.

        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.results = []
        for r_dict in data['results']:
            result = BenchmarkResult(
                name=r_dict['name'],
                benchmark_type=BenchmarkType(r_dict['benchmark_type']),
                iterations=r_dict['iterations'],
                total_time_ns=r_dict['total_time_ns'],
                avg_time_ns=r_dict['avg_time_ns'],
                min_time_ns=r_dict['min_time_ns'],
                max_time_ns=r_dict['max_time_ns'],
                std_time_ns=r_dict['std_time_ns'],
                throughput_ops_per_sec=r_dict['throughput_ops_per_sec'],
                metadata=r_dict.get('metadata', {}),
            )
            self.results.append(result)


def create_benchmark(
    enclave: Enclave,
    overhead_model: Optional[OverheadModel] = None,
) -> TEEBenchmark:
    """
    Factory function to create benchmark.

    Args:
        enclave: TEE enclave to benchmark
        overhead_model: Optional overhead model

    Returns:
        TEEBenchmark instance
    """
    return TEEBenchmark(enclave, overhead_model)


def run_standard_benchmark_suite(
    enclave: Enclave,
    data_size: int = 1000,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Run standard benchmark suite.

    Args:
        enclave: TEE enclave to benchmark
        data_size: Input data size
        iterations: Number of iterations

    Returns:
        Dictionary with benchmark results
    """
    benchmark = create_benchmark(enclave)

    results = {}

    # 1. Enclave overhead
    results['enclave_overhead'] = benchmark.benchmark_enclave_entry_exit(
        data_size=data_size,
        iterations=iterations,
    )

    # 2. TEE vs plaintext for various operations
    operations = {
        'add': lambda x: x + 1,
        'multiply': lambda x: x * 2,
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'relu': lambda x: np.maximum(0, x),
    }

    results['operations'] = {}

    for op_name, op_func in operations.items():
        # Plaintext baseline
        plaintext_result = benchmark.benchmark_plaintext_operation(
            operation=op_func,
            data_size=data_size,
            iterations=iterations,
            name=f"{op_name}_plaintext",
        )

        results['operations'][f'{op_name}_plaintext'] = plaintext_result

    # 3. Scalability
    results['scalability'] = benchmark.benchmark_scalability(
        operation=lambda x: x * 2 + 1,
        data_sizes=[100, 500, 1000, 5000],
        iterations_per_size=iterations // 2,
    )

    # 4. TEE vs HE comparison
    results['tee_vs_he'] = benchmark.compare_tee_vs_he(
        data_size=data_size,
        he_slowdown_factor=100.0,
    )

    return {
        'results': results,
        'summary': benchmark.get_results_summary(),
    }
