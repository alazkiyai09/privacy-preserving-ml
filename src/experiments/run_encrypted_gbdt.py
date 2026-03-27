"""Encrypted-GBDT training smoke runner."""

from __future__ import annotations

from src.models.encrypted_gbdt.encrypted_training import plan_encrypted_training


def run() -> dict[str, object]:
    return {
        "experiment": "encrypted_gbdt",
        "plan": plan_encrypted_training(num_trees=100, depth=6),
    }
