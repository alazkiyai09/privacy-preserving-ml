from src.encryption.he.ckks import CKKSCipher
from src.encryption.hybrid.ht2ml_protocol import run_ht2ml_protocol
from src.verification.commitments.protocol import commit_then_reveal
from src.verification.commitments.batch_verify import batch_verify


def test_ckks_roundtrip() -> None:
    cipher = CKKSCipher()
    payload = cipher.encrypt([1.0, 2.0, 3.0])
    assert cipher.decrypt(payload) == [1.0, 2.0, 3.0]


def test_hybrid_protocol_smoke() -> None:
    result = run_ht2ml_protocol([0.2, 0.7, 0.8])
    assert "score" in result


def test_commitment_flow() -> None:
    item = commit_then_reveal("model-update", "nonce-1")
    assert batch_verify([item]) is True
