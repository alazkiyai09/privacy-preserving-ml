def trimmed_mean(values: list[float], trim: int = 1) -> float:
    ordered = sorted(values)
    kept = ordered[trim: len(ordered) - trim] if len(ordered) > trim * 2 else ordered
    return sum(kept) / max(len(kept), 1)
