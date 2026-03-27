class PhishingClassifier:
    def predict(self, features: list[float]) -> int:
        return int(sum(features) / max(len(features), 1) >= 0.5)
