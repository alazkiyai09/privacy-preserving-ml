import hashlib


def verify_proof(statement: str, witness: str, proof: dict) -> bool:
    expected = hashlib.sha256(f"{statement}:{witness}".encode()).hexdigest()[:64]
    return proof.get("proof") == expected
