"""
TEE Benchmarking Module
========================

Performance benchmarking and reporting for TEE operations.

Components:
- tee_benchmarks: Benchmarking framework
- reports: Report generation
"""

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

__all__ = [
    # Benchmark types
    'BenchmarkType',
    'BenchmarkResult',
    'ComparisonResult',
    'TEEBenchmark',
    'create_benchmark',
    'run_standard_benchmark_suite',
    # Reports
    'ReportFormat',
    'PerformanceReport',
    'ScalabilityReport',
    'create_performance_report',
    'create_scalability_report',
]
