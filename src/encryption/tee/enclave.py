from dataclasses import dataclass


@dataclass
class EnclaveSession:
    enclave_id: str

    def run(self, operation: str, payload: dict) -> dict:
        return {"enclave_id": self.enclave_id, "operation": operation, "payload": payload, "status": "ok"}
