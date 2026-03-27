def encrypted_add(left: dict, right: dict) -> dict:
    return {
        **left,
        "payload": [a + b for a, b in zip(left.get("payload", []), right.get("payload", []), strict=False)],
    }


def encrypted_scale(ciphertext: dict, factor: float) -> dict:
    return {**ciphertext, "payload": [value * factor for value in ciphertext.get("payload", [])]}
