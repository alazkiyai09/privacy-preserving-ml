def secure_split_score(histogram_left: list[float], histogram_right: list[float]) -> float:
    return round(abs(sum(histogram_left) - sum(histogram_right)), 4)
