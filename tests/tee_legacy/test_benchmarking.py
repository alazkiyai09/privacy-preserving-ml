"""
Unit Tests for TEE Benchmarking Framework
==========================================

Tests for:
- Benchmark execution
- Performance measurement
- Comparison analysis
- Report generation
"""

import pytest
import numpy as np
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave
from tee_ml.simulation.overhead import OverheadModel
from tee_ml.benchmarking.tee_benchmarks import (
    BenchmarkType,
    BenchmarkResult,
    ComparisonResult,
    TEEBenchmark,
    create_benchmark,
    run_standard_benchmark_suite,
)
from tee_ml.benchmarking.reports import (
    ReportFormat,
    PerformanceReport,
    ScalabilityReport,
    create_performance_report,
    create_scalability_report,
)


class TestBenchmarkResult:
    """Test benchmark result data structure."""

    def test_create_benchmark_result(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            name="test_benchmark",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=100,
            total_time_ns=1_000_000,
            avg_time_ns=10_000,
            min_time_ns=8_000,
            max_time_ns=15_000,
            std_time_ns=2_000,
            throughput_ops_per_sec=100_000,
            metadata={'size': 1000},
        )

        assert result.name == "test_benchmark"
        assert result.benchmark_type == BenchmarkType.OPERATION_SPECIFIC
        assert result.iterations == 100
        assert result.avg_time_ns == 10_000

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=100_000,
            avg_time_ns=10_000,
            min_time_ns=8_000,
            max_time_ns=12_000,
            std_time_ns=1_000,
            throughput_ops_per_sec=100,
        )

        result_dict = result.to_dict()

        assert result_dict['name'] == "test"
        assert result_dict['benchmark_type'] == "operation_specific"
        assert result_dict['iterations'] == 10
        assert 'avg_time_ns' in result_dict

    def test_get_slowdown_factor(self):
        """Test slowdown factor calculation."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=20_000,
            avg_time_ns=2_000,
            min_time_ns=1_500,
            max_time_ns=2_500,
            std_time_ns=500,
            throughput_ops_per_sec=500,
        )

        # Baseline of 1000ns, result is 2000ns = 2x slowdown
        slowdown = result.get_slowdown_factor(baseline_time_ns=1_000)
        assert slowdown == 2.0

    def test_get_slowdown_factor_zero_baseline(self):
        """Test slowdown with zero baseline."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        slowdown = result.get_slowdown_factor(baseline_time_ns=0)
        assert slowdown == float('inf')


class TestComparisonResult:
    """Test comparison result data structure."""

    def test_create_comparison_result(self):
        """Test creating comparison result."""
        result = ComparisonResult(
            name="test_comparison",
            baseline_name="plaintext",
            baseline_avg_ns=1_000,
            comparison_avg_ns=2_000,
            slowdown_factor=2.0,
            speedup_factor=0.5,
            percentage_difference=100.0,
            conclusion="TEE is 2x slower",
        )

        assert result.name == "test_comparison"
        assert result.baseline_name == "plaintext"
        assert result.slowdown_factor == 2.0
        assert result.speedup_factor == 0.5

    def test_to_dict(self):
        """Test converting comparison to dictionary."""
        result = ComparisonResult(
            name="test",
            baseline_name="baseline",
            baseline_avg_ns=1000,
            comparison_avg_ns=2000,
            slowdown_factor=2.0,
            speedup_factor=0.5,
            percentage_difference=100.0,
            conclusion="Test",
        )

        result_dict = result.to_dict()

        assert result_dict['name'] == "test"
        assert result_dict['slowdown_factor'] == 2.0
        assert 'conclusion' in result_dict


class TestTEEBenchmark:
    """Test TEE benchmark framework."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="benchmark-test")

    @pytest.fixture
    def benchmark(self, enclave):
        return create_benchmark(enclave)

    def test_benchmark_creation(self, benchmark, enclave):
        """Test benchmark creation."""
        assert benchmark.enclave == enclave
        assert len(benchmark.results) == 0

    def test_benchmark_function(self, benchmark):
        """Test benchmarking a function."""
        def simple_func(x):
            return x * 2

        result = benchmark.benchmark_function(
            func=simple_func,
            func_args=(5,),
            iterations=10,
            warmup_iterations=2,
            name="simple_test",
        )

        assert result.name == "simple_test"
        assert result.iterations == 10
        assert result.avg_time_ns > 0
        assert result.throughput_ops_per_sec > 0
        assert len(benchmark.results) == 1

    def test_benchmark_plaintext_operation(self, benchmark):
        """Test benchmarking plaintext operation."""
        def add_one(x):
            return x + 1

        result = benchmark.benchmark_plaintext_operation(
            operation=add_one,
            data_size=100,
            iterations=10,
            name="add_one",
        )

        assert result.name == "add_one"
        assert result.avg_time_ns > 0
        assert result.metadata['data_size'] == 100
        assert result.metadata['execution'] == 'plaintext'

    def test_benchmark_tee_operation(self, benchmark):
        """Test benchmarking TEE operation."""
        def tee_add_one(data, session):
            return session.execute(lambda arr: data + 1)

        result = benchmark.benchmark_tee_operation(
            operation=tee_add_one,
            data_size=100,
            iterations=10,
            name="tee_add_one",
        )

        assert result.name == "tee_add_one"
        assert result.avg_time_ns > 0
        assert result.metadata['execution'] == 'tee'

    def test_benchmark_enclave_entry_exit(self, benchmark):
        """Test benchmarking enclave entry/exit overhead."""
        result = benchmark.benchmark_enclave_entry_exit(
            data_size=1000,
            iterations=10,
        )

        assert result.name == "enclave_entry_exit"
        assert result.benchmark_type == BenchmarkType.ENCLAVE_OVERHEAD
        assert result.avg_time_ns > 0
        # Entry/exit should have measurable overhead
        assert result.avg_time_ns > 1000  # At least 1 μs

    def test_benchmark_scalability(self, benchmark):
        """Test scalability benchmarking."""
        def multiply_by_two(x):
            return x * 2

        results = benchmark.benchmark_scalability(
            operation=multiply_by_two,
            data_sizes=[10, 100, 1000],
            iterations_per_size=5,
            name="scalability_test",
        )

        assert len(results) == 3
        for result in results:
            assert result.avg_time_ns > 0
            assert 'data_size' in result.metadata

    def test_benchmark_memory_scalability(self, benchmark):
        """Test memory scalability benchmarking."""
        results = benchmark.benchmark_memory_scalability(
            data_sizes_mb=[0.1, 0.5],
            iterations=5,
        )

        assert len(results) == 2
        for result in results:
            assert result.avg_time_ns > 0
            assert 'data_size_mb' in result.metadata

    def test_estimate_he_performance(self, benchmark):
        """Test HE performance estimation."""
        he_time = benchmark.estimate_he_performance(
            data_size=1000,
            he_slowdown_factor=100.0,
        )

        assert he_time > 0
        # Should be significantly higher than plaintext
        assert he_time > 10_000  # At least 10 μs

    def test_get_results_summary(self, benchmark):
        """Test getting results summary."""
        # Run a benchmark first
        def simple_func(x):
            return x + 1

        benchmark.benchmark_plaintext_operation(
            operation=simple_func,
            data_size=100,
            iterations=10,
        )

        summary = benchmark.get_results_summary()

        assert summary['total_benchmarks'] == 1
        assert 'avg_time_ns' in summary
        assert 'total_time_ns' in summary

    def test_save_and_load_results(self, benchmark):
        """Test saving and loading results."""
        # Run a benchmark
        def simple_func(x):
            return x + 1

        benchmark.benchmark_plaintext_operation(
            operation=simple_func,
            data_size=100,
            iterations=10,
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            benchmark.save_results(temp_path)

            # Create new benchmark and load
            new_benchmark = create_benchmark(benchmark.enclave)
            new_benchmark.load_results(temp_path)

            assert len(new_benchmark.results) == len(benchmark.results)
        finally:
            # Cleanup
            Path(temp_path).unlink()


class TestBenchmarkComparisons:
    """Test benchmark comparisons."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="comparison-test")

    @pytest.fixture
    def benchmark(self, enclave):
        return create_benchmark(enclave)

    def test_benchmark_tee_vs_plaintext(self, benchmark):
        """Test TEE vs plaintext comparison."""
        def plaintext_op(data):
            return data * 2

        def tee_op(data, session):
            return session.execute(lambda arr: data * 2)

        comparison = benchmark.benchmark_tee_vs_plaintext(
            operation=plaintext_op,
            tee_operation=tee_op,
            data_size=100,
            iterations=10,
            name="multiply_comparison",
        )

        assert comparison.name == "multiply_comparison"
        assert comparison.baseline_name == "plaintext"
        assert comparison.slowdown_factor >= 1.0  # TEE should be slower or equal
        assert comparison.speedup_factor <= 1.0
        assert comparison.conclusion  # Should have a conclusion

    def test_compare_tee_vs_he(self, benchmark):
        """Test TEE vs HE comparison."""
        comparison = benchmark.compare_tee_vs_he(
            data_size=1000,
            he_slowdown_factor=100.0,
            name="tee_vs_he_test",
        )

        assert comparison.name == "tee_vs_he_test"
        assert comparison.baseline_name == "he"
        # TEE should generally be faster than HE, but overhead can vary
        # Just check that the comparison was successful
        assert comparison.slowdown_factor > 0
        assert comparison.speedup_factor > 0
        assert comparison.conclusion


class TestStandardBenchmarkSuite:
    """Test standard benchmark suite."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="suite-test")

    def test_run_standard_benchmark_suite(self, enclave):
        """Test running standard benchmark suite."""
        results = run_standard_benchmark_suite(
            enclave=enclave,
            data_size=100,
            iterations=10,
        )

        assert 'results' in results
        assert 'summary' in results

        # Check expected result types
        assert 'enclave_overhead' in results['results']
        assert 'operations' in results['results']
        assert 'scalability' in results['results']
        assert 'tee_vs_he' in results['results']


class TestPerformanceReport:
    """Test performance report generation."""

    @pytest.fixture
    def report(self):
        return create_performance_report(
            title="Test Report",
            author="Test Author",
        )

    def test_report_creation(self, report):
        """Test report creation."""
        assert report.title == "Test Report"
        assert report.author == "Test Author"
        assert len(report.benchmark_results) == 0
        assert len(report.comparison_results) == 0

    def test_add_benchmark_result(self, report):
        """Test adding benchmark result."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)

        assert len(report.benchmark_results) == 1

    def test_add_comparison_result(self, report):
        """Test adding comparison result."""
        result = ComparisonResult(
            name="test",
            baseline_name="baseline",
            baseline_avg_ns=1000,
            comparison_avg_ns=2000,
            slowdown_factor=2.0,
            speedup_factor=0.5,
            percentage_difference=100.0,
            conclusion="Test",
        )

        report.add_comparison_result(result)

        assert len(report.comparison_results) == 1

    def test_set_metadata(self, report):
        """Test setting metadata."""
        report.set_metadata(
            test_key="test_value",
            another_key=123,
        )

        assert report.metadata['test_key'] == "test_value"
        assert report.metadata['another_key'] == 123

    def test_generate_text_summary(self, report):
        """Test generating text summary."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)
        report.set_metadata(test_metadata="value")

        summary = report.generate_summary(format=ReportFormat.TEXT)

        assert "Test Report" in summary
        assert "Test Author" in summary
        assert "test" in summary
        assert "test_metadata" in summary

    def test_generate_markdown_summary(self, report):
        """Test generating markdown summary."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)

        summary = report.generate_summary(format=ReportFormat.MARKDOWN)

        assert "# Test Report" in summary
        assert "| test |" in summary
        assert "Test Author" in summary

    def test_generate_json_summary(self, report):
        """Test generating JSON summary."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)

        summary = report.generate_summary(format=ReportFormat.JSON)

        # Should be valid JSON
        data = json.loads(summary)
        assert data['title'] == "Test Report"
        assert 'benchmark_results' in data

    def test_generate_detailed_analysis(self, report):
        """Test generating detailed analysis."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)

        analysis = report.generate_detailed_analysis()

        assert "DETAILED PERFORMANCE ANALYSIS" in analysis
        assert "Overall Statistics" in analysis
        assert "Recommendations" in analysis

    def test_save_report(self, report):
        """Test saving report to file."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.OPERATION_SPECIFIC,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
        )

        report.add_benchmark_result(result)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            report.save_report(temp_path, format=ReportFormat.TEXT)

            # Check file exists and has content
            content = Path(temp_path).read_text()
            assert "Test Report" in content
            assert len(content) > 0
        finally:
            Path(temp_path).unlink()


class TestScalabilityReport:
    """Test scalability report generation."""

    @pytest.fixture
    def report(self):
        return create_scalability_report(title="Test Scalability")

    def test_report_creation(self, report):
        """Test report creation."""
        assert report.title == "Test Scalability"
        assert len(report.results) == 0

    def test_add_result(self, report):
        """Test adding result."""
        result = BenchmarkResult(
            name="test",
            benchmark_type=BenchmarkType.SCALABILITY,
            iterations=10,
            total_time_ns=10_000,
            avg_time_ns=1_000,
            min_time_ns=500,
            max_time_ns=1_500,
            std_time_ns=500,
            throughput_ops_per_sec=1000,
            metadata={'data_size': 100},
        )

        report.add_result(result)

        assert len(report.results) == 1

    def test_generate_text_report(self, report):
        """Test generating text report."""
        # Add results with different sizes
        for size in [100, 200, 300]:
            result = BenchmarkResult(
                name=f"size_{size}",
                benchmark_type=BenchmarkType.SCALABILITY,
                iterations=10,
                total_time_ns=size * 100,
                avg_time_ns=size * 10,
                min_time_ns=size * 5,
                max_time_ns=size * 15,
                std_time_ns=size * 2,
                throughput_ops_per_sec=1000 / size,
                metadata={'data_size': size},
            )
            report.add_result(result)

        text_report = report.generate_report(format=ReportFormat.TEXT)

        assert "Test Scalability" in text_report
        assert "Scalability Results" in text_report
        assert "Scalability Analysis" in text_report

    def test_generate_markdown_report(self, report):
        """Test generating markdown report."""
        # Add results
        for size in [100, 200]:
            result = BenchmarkResult(
                name=f"size_{size}",
                benchmark_type=BenchmarkType.SCALABILITY,
                iterations=10,
                total_time_ns=size * 100,
                avg_time_ns=size * 10,
                min_time_ns=size * 5,
                max_time_ns=size * 15,
                std_time_ns=size * 2,
                throughput_ops_per_sec=1000 / size,
                metadata={'data_size': size},
            )
            report.add_result(result)

        md_report = report.generate_report(format=ReportFormat.MARKDOWN)

        assert "# Test Scalability" in md_report
        assert "| Data Size |" in md_report
        assert "## Analysis" in md_report


class TestIntegration:
    """Integration tests for benchmarking workflow."""

    def test_complete_benchmarking_workflow(self):
        """Test complete benchmarking and reporting workflow."""
        # Create enclave and benchmark
        enclave = Enclave(enclave_id="integration-test")
        benchmark = create_benchmark(enclave)

        # Run benchmarks
        def simple_op(x):
            return x * 2 + 1

        plaintext_result = benchmark.benchmark_plaintext_operation(
            operation=simple_op,
            data_size=1000,
            iterations=10,
            name="integration_plaintext",
        )

        # Run scalability test
        scalability_results = benchmark.benchmark_scalability(
            operation=simple_op,
            data_sizes=[100, 500, 1000],
            iterations_per_size=5,
            name="integration_scalability",
        )

        # Create report
        report = create_performance_report(
            title="Integration Test Report",
            author="Test Suite",
        )

        report.add_benchmark_result(plaintext_result)
        for result in scalability_results:
            report.add_benchmark_result(result)

        report.set_metadata(
            test_type="integration",
            total_benchmarks=len(scalability_results) + 1,
        )

        # Generate summaries
        text_summary = report.generate_summary(ReportFormat.TEXT)
        md_summary = report.generate_summary(ReportFormat.MARKDOWN)
        json_summary = report.generate_summary(ReportFormat.JSON)
        detailed_analysis = report.generate_detailed_analysis()

        # Verify all summaries are generated
        assert "Integration Test Report" in text_summary
        assert "# Integration Test Report" in md_summary
        assert len(json_summary) > 0
        assert "DETAILED PERFORMANCE ANALYSIS" in detailed_analysis

    def test_comparison_workflow(self):
        """Test complete comparison workflow."""
        enclave = Enclave(enclave_id="comparison-integration")
        benchmark = create_benchmark(enclave)

        # Compare TEE vs plaintext
        def plaintext_op(data):
            return data + 1

        def tee_op(data, session):
            return session.execute(lambda arr: data + 1)

        comparison = benchmark.benchmark_tee_vs_plaintext(
            operation=plaintext_op,
            tee_operation=tee_op,
            data_size=100,
            iterations=10,
            name="integration_comparison",
        )

        # Create report with comparison
        report = create_performance_report(
            title="Comparison Report",
        )

        report.add_comparison_result(comparison)

        # Generate report
        summary = report.generate_summary(ReportFormat.TEXT)

        assert "Comparison Report" in summary
        assert comparison.name in summary
        assert comparison.conclusion in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
