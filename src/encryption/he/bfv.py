from dataclasses import dataclass


@dataclass
class BFVCipher:
    modulus: int = 40961

    def encrypt(self, values: list[int]) -> dict:
        return {"scheme": "bfv", "modulus": self.modulus, "payload": [int(v) for v in values]}

    def decrypt(self, ciphertext: dict) -> list[int]:
        return [int(v) for v in ciphertext.get("payload", [])]
