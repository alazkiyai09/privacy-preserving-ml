def krum(updates: list[list[float]]) -> list[float]:
    return min(updates, key=lambda update: sum(abs(value) for value in update))
