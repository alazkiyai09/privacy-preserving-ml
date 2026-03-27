def serve_hybrid_inference(request: dict) -> dict:
    score = min(sum(float(v) for v in request.get("ciphertext", [])) / 10.0, 1.0)
    return {"transport": "he+tee", "score": round(score, 4), "label": int(score >= 0.5)}
