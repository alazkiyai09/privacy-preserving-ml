from dataclasses import dataclass


@dataclass
class CKKSCipher:
    scale: float = 8192.0

    def encrypt(self, values: list[float]) -> dict:
        return {"scheme": "ckks", "scale": self.scale, "payload": list(values)}

    def decrypt(self, ciphertext: dict) -> list[float]:
        return list(ciphertext.get("payload", []))
