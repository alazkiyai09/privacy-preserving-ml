"""ZK proof-verification smoke runner."""

from __future__ import annotations

from src.verification.zkp.prover import generate_proof
from src.verification.zkp.verifier import verify_proof


def run() -> dict[str, object]:
    statement = "grad_norm <= bound"
    witness = "0.812"
    proof = generate_proof(statement, witness)
    valid = verify_proof(statement, witness, proof)
    return {"experiment": "zkp_verification", "valid": bool(valid), "proof": proof}
