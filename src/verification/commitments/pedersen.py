import hashlib


def pedersen_commit(value: str, randomness: str) -> str:
    return hashlib.sha256(f"pedersen:{value}:{randomness}".encode()).hexdigest()
