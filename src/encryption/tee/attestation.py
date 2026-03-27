def attest(enclave_id: str) -> dict:
    return {"enclave_id": enclave_id, "status": "trusted", "evidence": f"quote:{enclave_id}"}
