from src.verification.commitments.hash_commitment import hash_commit


def batch_verify(items: list[dict]) -> bool:
    return all(hash_commit(item["value"], item["nonce"]) == item["commitment"] for item in items)
