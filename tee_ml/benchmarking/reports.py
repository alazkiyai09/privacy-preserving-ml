"""
Performance Report Generation
==============================

Generate comprehensive performance reports from benchmark results.

Report Types:
- Summary Reports
- Comparison Reports
- Scalability Reports
- Detailed Analysis
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from tee_ml.benchmarking.tee_benchmarks import (
    BenchmarkResult,
    ComparisonResult,
    BenchmarkType,
)


class ReportFormat(Enum):
    """Report output format."""

    TEXT = "text"
    """Plain text report"""

    MARKDOWN = "markdown"
    """Markdown report"""

    JSON = "json"
    """JSON report"""

    HTML = "html"
    """HTML report (future)"""


@dataclass
class PerformanceMetrics:
    """Performance metrics summary."""

    total_time_ns: float
    avg_time_ns: float
    min_time_ns: float
    max_time_ns: float
    std_time_ns: float
    throughput_ops_per_sec: float

    def format_time(self, time_ns: float) -> str:
        """Format time in appropriate units."""
        if time_ns < 1000:
            return f"{time_ns:.2f} ns"
        elif time_ns < 1_000_000:
            return f"{time_ns / 1000:.2f} μs"
        elif time_ns < 1_000_000_000:
            return f"{time_ns / 1_000_000:.2f} ms"
        else:
            return f"{time_ns / 1_000_000_000:.2f} s"


@dataclass
class ComparisonMetrics:
    """Comparison metrics summary."""

    slowdown_factor: float
    speedup_factor: float
    percentage_difference: float
    conclusion: str

    def format_slowdown(self) -> str:
        """Format slowdown factor."""
        if self.slowdown_factor < 1.1:
            return f"Negligible ({self.slowdown_factor:.2f}x)"
        elif self.slowdown_factor < 2.0:
            return f"Moderate ({self.slowdown_factor:.2f}x)"
        elif self.slowdown_factor < 10.0:
            return f"Significant ({self.slowdown_factor:.2f}x)"
        else:
            return f"Severe ({self.slowdown_factor:.1f}x)"

    def format_speedup(self) -> str:
        """Format speedup factor."""
        if self.speedup_factor < 1.1:
            return f"Negligible ({self.speedup_factor:.2f}x)"
        elif self.speedup_factor < 2.0:
            return f"Moderate ({self.speedup_factor:.2f}x)"
        elif self.speedup_factor < 10.0:
            return f"Significant ({self.speedup_factor:.2f}x)"
        else:
            return f"Huge ({self.speedup_factor:.1f}x)"


class PerformanceReport:
    """
    Performance report generator.

    Creates comprehensive reports from benchmark results.
    """

    def __init__(
        self,
        title: str = "TEE Performance Report",
        author: str = "TEE ML Framework",
    ):
        """
        Initialize report generator.

        Args:
            title: Report title
            author: Report author
        """
        self.title = title
        self.author = author
        self.benchmark_results: List[BenchmarkResult] = []
        self.comparison_results: List[ComparisonResult] = []
        self.metadata: Dict[str, Any] = {}

    def add_benchmark_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.benchmark_results.append(result)

    def add_comparison_result(self, result: ComparisonResult):
        """Add a comparison result."""
        self.comparison_results.append(result)

    def set_metadata(self, **kwargs):
        """Set report metadata."""
        self.metadata.update(kwargs)

    def generate_summary(
        self,
        format: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """
        Generate summary report.

        Args:
            format: Output format

        Returns:
            Formatted report string
        """
        if format == ReportFormat.TEXT:
            return self._generate_text_summary()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_summary()
        elif format == ReportFormat.JSON:
            return self._generate_json_summary()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_text_summary(self) -> str:
        """Generate text format summary."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{self.title}")
        lines.append(f"Author: {self.author}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Metadata
        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Benchmark Results Summary
        if self.benchmark_results:
            lines.append("Benchmark Results Summary:")
            lines.append("-" * 70)

            # Group by type
            by_type = {}
            for result in self.benchmark_results:
                btype = result.benchmark_type.value
                if btype not in by_type:
                    by_type[btype] = []
                by_type[btype].append(result)

            for btype, results in by_type.items():
                lines.append(f"\n{btype.upper()}:")
                for result in results:
                    lines.append(f"  {result.name}:")
                    lines.append(f"    Avg Time: {result.avg_time_ns / 1000:.2f} μs")
                    lines.append(f"    Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
            lines.append("")

        # Comparison Results Summary
        if self.comparison_results:
            lines.append("Comparison Results Summary:")
            lines.append("-" * 70)

            for comparison in self.comparison_results:
                lines.append(f"\n{comparison.name}:")
                lines.append(f"  Baseline: {comparison.baseline_name}")
                lines.append(f"  Baseline Time: {comparison.baseline_avg_ns / 1000:.2f} μs")
                lines.append(f"  Comparison Time: {comparison.comparison_avg_ns / 1000:.2f} μs")
                lines.append(f"  Slowdown: {comparison.slowdown_factor:.2f}x")
                lines.append(f"  Speedup: {comparison.speedup_factor:.2f}x")
                lines.append(f"  Conclusion: {comparison.conclusion}")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_summary(self) -> str:
        """Generate markdown format summary."""
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Author:** {self.author}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Metadata
        if self.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Benchmark Results
        if self.benchmark_results:
            lines.append("## Benchmark Results")
            lines.append("")

            # Group by type
            by_type = {}
            for result in self.benchmark_results:
                btype = result.benchmark_type.value
                if btype not in by_type:
                    by_type[btype] = []
                by_type[btype].append(result)

            for btype, results in by_type.items():
                lines.append(f"### {btype.replace('_', ' ').title()}")
                lines.append("")
                lines.append("| Benchmark | Avg Time | Min Time | Max Time | Std Dev | Throughput |")
                lines.append("|-----------|----------|----------|----------|---------|------------|")

                for result in results:
                    avg_us = result.avg_time_ns / 1000
                    min_us = result.min_time_ns / 1000
                    max_us = result.max_time_ns / 1000
                    std_us = result.std_time_ns / 1000
                    throughput = result.throughput_ops_per_sec

                    lines.append(
                        f"| {result.name} | {avg_us:.2f} μs | {min_us:.2f} μs | "
                        f"{max_us:.2f} μs | {std_us:.2f} μs | {throughput:.2f} |"
                    )
                lines.append("")

        # Comparison Results
        if self.comparison_results:
            lines.append("## Comparison Results")
            lines.append("")
            lines.append("| Comparison | Baseline | Slowdown | Speedup | Conclusion |")
            lines.append("|------------|----------|----------|---------|------------|")

            for comparison in self.comparison_results:
                lines.append(
                    f"| {comparison.name} | {comparison.baseline_name} | "
                    f"{comparison.slowdown_factor:.2f}x | {comparison.speedup_factor:.2f}x | "
                    f"{comparison.conclusion} |"
                )

            lines.append("")

        return "\n".join(lines)

    def _generate_json_summary(self) -> str:
        """Generate JSON format summary."""
        data = {
            'title': self.title,
            'author': self.author,
            'generated': datetime.now().isoformat(),
            'metadata': self.metadata,
            'benchmark_results': [r.to_dict() for r in self.benchmark_results],
            'comparison_results': [r.to_dict() for r in self.comparison_results],
        }

        return json.dumps(data, indent=2)

    def generate_detailed_analysis(self) -> str:
        """
        Generate detailed analysis report.

        Returns:
            Detailed analysis text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DETAILED PERFORMANCE ANALYSIS")
        lines.append("=" * 70)
        lines.append("")

        # Overall Statistics
        if self.benchmark_results:
            lines.append("Overall Statistics:")
            lines.append("-" * 70)

            total_ops = sum(r.iterations for r in self.benchmark_results)
            total_time_ns = sum(r.total_time_ns for r in self.benchmark_results)
            avg_time_ns = total_time_ns / total_ops if total_ops > 0 else 0

            lines.append(f"  Total Operations: {total_ops}")
            lines.append(f"  Total Time: {total_time_ns / 1e9:.2f} seconds")
            lines.append(f"  Average Time per Op: {avg_time_ns / 1000:.2f} μs")
            lines.append("")

        # Performance by Type
        if self.benchmark_results:
            lines.append("Performance by Benchmark Type:")
            lines.append("-" * 70)

            by_type = {}
            for result in self.benchmark_results:
                btype = result.benchmark_type.value
                if btype not in by_type:
                    by_type[btype] = []
                by_type[btype].append(result)

            for btype, results in by_type.items():
                avg_times = [r.avg_time_ns for r in results]
                lines.append(f"\n  {btype.upper()}:")
                lines.append(f"    Count: {len(results)}")
                lines.append(f"    Avg Time: {np.mean(avg_times) / 1000:.2f} μs")
                lines.append(f"    Min Time: {np.min(avg_times) / 1000:.2f} μs")
                lines.append(f"    Max Time: {np.max(avg_times) / 1000:.2f} μs")

        # Comparison Analysis
        if self.comparison_results:
            lines.append("\n")
            lines.append("Comparison Analysis:")
            lines.append("-" * 70)

            avg_slowdown = np.mean([c.slowdown_factor for c in self.comparison_results])
            avg_speedup = np.mean([c.speedup_factor for c in self.comparison_results])

            lines.append(f"\n  Average Slowdown: {avg_slowdown:.2f}x")
            lines.append(f"  Average Speedup: {avg_speedup:.2f}x")

            # Find best and worst
            best = max(self.comparison_results, key=lambda c: c.speedup_factor)
            worst = min(self.comparison_results, key=lambda c: c.speedup_factor)

            lines.append(f"\n  Best Performance: {best.name} ({best.speedup_factor:.2f}x speedup)")
            lines.append(f"  Worst Performance: {worst.name} ({worst.speedup_factor:.2f}x speedup)")

        # Recommendations
        lines.append("\n")
        lines.append("Recommendations:")
        lines.append("-" * 70)

        if self.comparison_results:
            high_overhead = [c for c in self.comparison_results if c.slowdown_factor > 5.0]
            if high_overhead:
                lines.append("\n  High Overhead Operations:")
                for c in high_overhead:
                    lines.append(f"    - {c.name}: {c.slowdown_factor:.2f}x slowdown")
                    lines.append(f"      Consider optimization or batching")

            low_overhead = [c for c in self.comparison_results if c.slowdown_factor < 1.5]
            if low_overhead:
                lines.append("\n  Low Overhead Operations:")
                for c in low_overhead:
                    lines.append(f"    - {c.name}: {c.slowdown_factor:.2f}x slowdown")
                    lines.append(f"      Good candidates for frequent use")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_report(
        self,
        filepath: str,
        format: ReportFormat = ReportFormat.TEXT,
    ):
        """
        Save report to file.

        Args:
            filepath: Path to save report
            format: Report format
        """
        report_content = self.generate_summary(format=format)

        with open(filepath, 'w') as f:
            f.write(report_content)

    def save_detailed_analysis(self, filepath: str):
        """
        Save detailed analysis to file.

        Args:
            filepath: Path to save analysis
        """
        analysis = self.generate_detailed_analysis()

        with open(filepath, 'w') as f:
            f.write(analysis)


class ScalabilityReport:
    """
    Scalability analysis report.

    Analyzes how performance scales with input size.
    """

    def __init__(self, title: str = "Scalability Analysis"):
        """
        Initialize scalability report.

        Args:
            title: Report title
        """
        self.title = title
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def generate_report(self, format: ReportFormat = ReportFormat.TEXT) -> str:
        """
        Generate scalability report.

        Args:
            format: Output format

        Returns:
            Formatted report
        """
        if format == ReportFormat.TEXT:
            return self._generate_text_report()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_text_report(self) -> str:
        """Generate text format report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{self.title}")
        lines.append("=" * 70)
        lines.append("")

        # Sort by data size
        sorted_results = sorted(
            self.results,
            key=lambda r: r.metadata.get('data_size', 0),
        )

        lines.append("Scalability Results:")
        lines.append("-" * 70)
        lines.append(f"{'Size':<15} {'Avg Time':<15} {'Throughput':<15} {'Time/Element':<15}")
        lines.append("-" * 70)

        for result in sorted_results:
            size = result.metadata.get('data_size', 0)
            avg_time_us = result.avg_time_ns / 1000
            throughput = result.throughput_ops_per_sec
            time_per_element_ns = result.avg_time_ns / size if size > 0 else 0

            lines.append(
                f"{size:<15} {avg_time_us:<15.2f} {throughput:<15.2f} {time_per_element_ns:<15.2f}"
            )

        lines.append("")

        # Analysis
        if len(sorted_results) >= 2:
            lines.append("Scalability Analysis:")
            lines.append("-" * 70)

            # Calculate scaling factor
            first = sorted_results[0]
            last = sorted_results[-1]

            first_size = first.metadata.get('data_size', 1)
            last_size = last.metadata.get('data_size', 1)

            size_ratio = last_size / first_size if first_size > 0 else 1
            time_ratio = last.avg_time_ns / first.avg_time_ns if first.avg_time_ns > 0 else 1

            lines.append(f"  Size Ratio: {size_ratio:.2f}x")
            lines.append(f"  Time Ratio: {time_ratio:.2f}x")

            if time_ratio < size_ratio * 1.2:
                lines.append("  Scaling: Sub-linear (better than linear)")
            elif time_ratio < size_ratio * 1.5:
                lines.append("  Scaling: Approximately linear")
            else:
                lines.append("  Scaling: Super-linear (worse than linear)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report."""
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")

        # Sort by data size
        sorted_results = sorted(
            self.results,
            key=lambda r: r.metadata.get('data_size', 0),
        )

        lines.append("## Scalability Results")
        lines.append("")
        lines.append("| Data Size | Avg Time (μs) | Throughput (ops/sec) | Time/Element (ns) |")
        lines.append("|-----------|---------------|---------------------|-------------------|")

        for result in sorted_results:
            size = result.metadata.get('data_size', 0)
            avg_time_us = result.avg_time_ns / 1000
            throughput = result.throughput_ops_per_sec
            time_per_element_ns = result.avg_time_ns / size if size > 0 else 0

            lines.append(
                f"| {size} | {avg_time_us:.2f} | {throughput:.2f} | {time_per_element_ns:.2f} |"
            )

        lines.append("")

        # Analysis
        if len(sorted_results) >= 2:
            lines.append("## Analysis")
            lines.append("")

            first = sorted_results[0]
            last = sorted_results[-1]

            first_size = first.metadata.get('data_size', 1)
            last_size = last.metadata.get('data_size', 1)

            size_ratio = last_size / first_size if first_size > 0 else 1
            time_ratio = last.avg_time_ns / first.avg_time_ns if first.avg_time_ns > 0 else 1

            lines.append(f"- **Size Ratio:** {size_ratio:.2f}x")
            lines.append(f"- **Time Ratio:** {time_ratio:.2f}x")

            if time_ratio < size_ratio * 1.2:
                lines.append("- **Scaling:** Sub-linear (better than linear)")
            elif time_ratio < size_ratio * 1.5:
                lines.append("- **Scaling:** Approximately linear")
            else:
                lines.append("- **Scaling:** Super-linear (worse than linear)")

        return "\n".join(lines)


def create_performance_report(
    title: str = "TEE Performance Report",
    author: str = "TEE ML Framework",
) -> PerformanceReport:
    """
    Factory function to create performance report.

    Args:
        title: Report title
        author: Report author

    Returns:
        PerformanceReport instance
    """
    return PerformanceReport(title, author)


def create_scalability_report(
    title: str = "Scalability Analysis",
) -> ScalabilityReport:
    """
    Factory function to create scalability report.

    Args:
        title: Report title

    Returns:
        ScalabilityReport instance
    """
    return ScalabilityReport(title)
