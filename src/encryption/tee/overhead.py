def estimate_overhead(context_switches: int, encryption_steps: int) -> dict:
    return {"switch_ms": context_switches * 0.3, "crypto_ms": encryption_steps * 0.7}
