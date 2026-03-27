def compute_gradient_norm(update: list[float]) -> float:
    return sum(abs(value) for value in update)


def aggregate_gradients_weighted(updates: list[list[float]], weights: list[float]) -> list[float]:
    if not updates:
        return []
    width = len(updates[0])
    result = []
    for index in range(width):
        numerator = sum(update[index] * weight for update, weight in zip(updates, weights, strict=False))
        denominator = sum(weights) or 1.0
        result.append(numerator / denominator)
    return result
