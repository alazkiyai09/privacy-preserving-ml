from src.verification.robust.krum import krum


def verifiable_krum(updates: list[list[float]], proofs: list[bool]) -> list[float]:
    filtered = [update for update, valid in zip(updates, proofs, strict=False) if valid]
    return krum(filtered or updates)
