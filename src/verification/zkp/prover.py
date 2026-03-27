import hashlib


def create_proof(statement: str, witness: str) -> dict:
    digest = hashlib.sha256(f"{statement}:{witness}".encode()).hexdigest()
    return {"statement": statement, "proof": digest[:64]}
