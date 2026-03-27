"""TEE-overhead benchmark runner."""

from __future__ import annotations

from src.encryption.tee.overhead import estimate_overhead


def run() -> dict[str, object]:
    return {
        "experiment": "tee_benchmark",
        "results": estimate_overhead(context_switches=120, encryption_steps=80),
    }
