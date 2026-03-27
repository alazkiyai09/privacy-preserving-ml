import hashlib


def hash_commit(value: str, nonce: str) -> str:
    return hashlib.sha256(f"{value}:{nonce}".encode()).hexdigest()
