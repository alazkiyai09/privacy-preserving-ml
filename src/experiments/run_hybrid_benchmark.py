"""Hybrid HE+TEE benchmark runner."""

from __future__ import annotations

from src.encryption.hybrid.benchmarks import benchmark_hybrid_stack


def run() -> dict[str, object]:
    return {"experiment": "hybrid_benchmark", "results": benchmark_hybrid_stack()}
