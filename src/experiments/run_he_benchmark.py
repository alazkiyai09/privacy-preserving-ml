"""Homomorphic-encryption benchmark runner."""

from __future__ import annotations

from src.encryption.he.benchmarks import benchmark_he_layers


def run() -> dict[str, object]:
    return {"experiment": "he_benchmark", "results": benchmark_he_layers()}
