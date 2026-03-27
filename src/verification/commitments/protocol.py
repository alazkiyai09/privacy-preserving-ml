from src.verification.commitments.hash_commitment import hash_commit


def commit_then_reveal(value: str, nonce: str) -> dict:
    commitment = hash_commit(value, nonce)
    return {"commitment": commitment, "value": value, "nonce": nonce}
