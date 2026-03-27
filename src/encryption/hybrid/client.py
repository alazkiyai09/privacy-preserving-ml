def encrypt_client_payload(features: list[float]) -> dict:
    return {"transport": "he+tee", "ciphertext": list(features), "stage": "client"}
