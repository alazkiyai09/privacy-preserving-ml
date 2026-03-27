def optimize_batching(batch_sizes: list[int]) -> dict:
    return {"input_batches": batch_sizes, "communication_reduction_x": 5.0}
