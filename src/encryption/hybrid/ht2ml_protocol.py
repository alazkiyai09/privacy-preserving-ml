from src.encryption.hybrid.client import encrypt_client_payload
from src.encryption.hybrid.server import serve_hybrid_inference


def run_ht2ml_protocol(features: list[float]) -> dict:
    request = encrypt_client_payload(features)
    return serve_hybrid_inference(request)
