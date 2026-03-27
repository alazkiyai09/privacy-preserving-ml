"""Commitment-based verification smoke runner."""

from __future__ import annotations

from src.verification.commitments.protocol import commit_then_reveal


def run() -> dict[str, object]:
    payload = commit_then_reveal(value="model_update_hash", nonce="demo_nonce")
    return {"experiment": "commitment_fl", "commitment": payload["commitment"]}
