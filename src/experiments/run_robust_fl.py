"""Robust verification smoke runner."""

from __future__ import annotations

from src.verification.robust.verifiable_krum import verifiable_krum


def run() -> dict[str, object]:
    updates = [[0.2, 0.5, 0.7], [0.19, 0.52, 0.71], [9.0, 9.0, 9.0]]
    proofs = [True, True, False]
    aggregated = verifiable_krum(updates, proofs)
    return {"experiment": "robust_fl", "aggregated": aggregated}
