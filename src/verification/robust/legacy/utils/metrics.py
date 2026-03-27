"""
Attack and Defense Metrics

Comprehensive metrics for evaluating attack impact and defense effectiveness.

Metrics:
- Attack success rate
- Clean vs poisoned accuracy
- False positive/negative rates
- Per-bank analysis
- Defense overhead
"""

import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


def compute_attack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    poisoned_indices: np.ndarray = None,
    backdoor_indices: List[int] = None
) -> Dict[str, float]:
    """
    Compute comprehensive attack metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        poisoned_indices: Indices of poisoned samples (for label flip)
        backdoor_indices: Indices with backdoor trigger

    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy_overall = np.mean(y_pred == y_true)

    metrics = {
        "accuracy_overall": float(accuracy_overall),
        "total_samples": len(y_true)
    }

    # If poisoned indices provided
    if poisoned_indices is not None and len(poisoned_indices) > 0:
        # Accuracy on clean samples
        clean_mask = ~poisoned_indices
        accuracy_clean = np.mean(y_pred[clean_mask] == y_true[clean_mask])

        # Accuracy on poisoned samples
        poisoned_mask = poisoned_indices
        if poisoned_mask.sum() > 0:
            accuracy_poisoned = np.mean(
                y_pred[poisoned_mask] == y_true[poisoned_mask]
            )
        else:
            accuracy_poisoned = 0.0

        # Attack success rate (how many poisoned samples worked)
        # For label flip (1→0), success = predicted as 0
        # Simplified: just use accuracy on poisoned
        attack_success_rate = 1.0 - accuracy_poisoned

        metrics.update({
            "accuracy_clean": float(accuracy_clean),
            "accuracy_poisoned": float(accuracy_poisoned),
            "num_poisoned": int(poisoned_indices.sum()),
            "attack_success_rate": float(attack_success_rate)
        })

    # If backdoor indices provided
    if backdoor_indices is not None and len(backdoor_indices) > 0:
        backdoor_mask = np.zeros(len(y_true), dtype=bool)
        backdoor_mask[list(backdoor_indices)] = True

        # Backdoor success rate
        if backdoor_mask.sum() > 0:
            backdoor_success = np.mean(y_pred[backdoor_mask] == 0)  # Target is 0
        else:
            backdoor_success = 0.0

        metrics.update({
            "backdoor_success_rate": float(backdoor_success),
            "backdoor_total": len(backdoor_indices)
        })

    # Binary classification metrics
    if len(np.unique(y_true)) == 2:
        # True positives, false positives, etc.
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # False positive rate (critical for phishing!)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # False negative rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics.update({
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr)
        })

    return metrics


def compute_per_bank_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bank_names: List[str],
    bank_triggers: Dict[str, Set[int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per bank (for adversarial bank analysis).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        bank_names: List of bank names for each sample
        bank_triggers: Dictionary mapping bank names to sample indices with triggers

    Returns:
        Dictionary mapping bank names to metrics
    """
    bank_metrics = {}

    # Get unique banks
    unique_banks = set(bank_names)

    for bank in unique_banks:
        # Get samples from this bank
        bank_mask = np.array([name == bank for name in bank_names])

        if bank_mask.sum() == 0:
            continue

        bank_y_true = y_true[bank_mask]
        bank_y_pred = y_pred[bank_mask]

        # Compute metrics for this bank
        accuracy = np.mean(bank_y_pred == bank_y_true)

        # For binary classification
        if len(np.unique(bank_y_true)) == 2:
            tp = np.sum((bank_y_pred == 1) & (bank_y_true == 1))
            tn = np.sum((bank_y_pred == 0) & (bank_y_true == 0))
            fp = np.sum((bank_y_pred == 1) & (bank_y_true == 0))
            fn = np.sum((bank_y_pred == 0) & (bank_y_true == 1))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            bank_metrics[bank] = {
                "accuracy": float(accuracy),
                "num_samples": int(bank_mask.sum()),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "false_positive_rate": float(fpr),
                "false_negative_rate": float(fnr)
            }
        else:
            bank_metrics[bank] = {
                "accuracy": float(accuracy),
                "num_samples": int(bank_mask.sum())
            }

        # Check if this bank has backdoor triggers
        if bank_triggers and bank in bank_triggers:
            trigger_indices = bank_triggers[bank]
            trigger_mask = np.zeros(len(y_true), dtype=bool)
            trigger_mask[list(trigger_indices)] = True

            if trigger_mask.sum() > 0:
                # Backdoor success for this bank
                trigger_y_pred = y_pred[trigger_mask]
                backdoor_success = np.mean(trigger_y_pred == 0)  # Target is 0

                bank_metrics[bank]["backdoor_success_rate"] = float(backdoor_success)
                bank_metrics[bank]["num_trigger_samples"] = len(trigger_indices)

    return bank_metrics


def compute_defense_overhead(
    defense_config: Dict[str, bool],
    num_clients: int,
    avg_proof_verify_time: float = 0.02,
    avg_aggregation_time: float = 0.03
) -> Dict[str, float]:
    """
    Compute computational overhead of defenses.

    Args:
        defense_config: Which defenses are enabled
        num_clients: Number of clients
        avg_proof_verify_time: Time to verify ZK proof (seconds)
        avg_aggregation_time: Time for normal aggregation (seconds)

    Returns:
        Dictionary with overhead metrics
    """
    overhead = {
        "num_clients": num_clients,
        "baseline_aggregation_time": avg_aggregation_time
    }

    total_time = avg_aggregation_time

    # ZK proof verification overhead
    if defense_config.get("zk", False):
        zk_time = num_clients * avg_proof_verify_time
        overhead["zk_proof_overhead"] = zk_time
        total_time += zk_time
    else:
        overhead["zk_proof_overhead"] = 0.0

    # Byzantine aggregation overhead
    # Krum: O(n^2) distance computations
    if defense_config.get("byzantine", False):
        byzantine_time = avg_aggregation_time * 1.5  # Approx 1.5x slower
        overhead["byzantine_overhead"] = byzantine_time - avg_aggregation_time
        total_time += overhead["byzantine_overhead"]
    else:
        overhead["byzantine_overhead"] = 0.0

    # Anomaly detection overhead
    if defense_config.get("anomaly_detection", False):
        anomaly_time = avg_aggregation_time * 0.8  # Approx 0.8x
        overhead["anomaly_detection_overhead"] = anomaly_time
        total_time += anomaly_time
    else:
        overhead["anomaly_detection_overhead"] = 0.0

    # Reputation system overhead
    if defense_config.get("reputation", False):
        reputation_time = avg_aggregation_time * 0.2  # Approx 0.2x
        overhead["reputation_overhead"] = reputation_time
        total_time += reputation_time
    else:
        overhead["reputation_overhead"] = 0.0

    overhead["total_overhead"] = total_time
    overhead["overhead_per_client"] = total_time / num_clients if num_clients > 0 else 0.0
    overhead["speedup_factor"] = avg_aggregation_time / total_time if total_time > 0 else 0.0

    return overhead


def compute_statistical_significance(
    results_list: List[Dict[str, float]],
    metric_name: str = "attack_success_rate"
) -> Dict[str, float]:
    """
    Compute mean, std, and confidence interval from multiple runs.

    Args:
        results_list: List of result dictionaries from multiple runs
        metric_name: Name of metric to analyze

    Returns:
        Dictionary with statistics
    """
    values = [result[metric_name] for result in results_list]

    if len(values) == 0:
        return {}

    mean = np.mean(values)
    std = np.std(values)
    sem = std / np.sqrt(len(values))  # Standard error of mean

    # 95% confidence interval
    ci_95 = 1.96 * sem

    return {
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "ci_95_lower": float(mean - ci_95),
        "ci_95_upper": float(mean + ci_95),
        "num_runs": len(values)
    }


def compute_effect_size(
    control_results: List[float],
    treatment_results: List[float]
) -> Dict[str, float]:
    """
    Compute Cohen's d effect size.

    Args:
        control_results: Results from control group
        treatment_results: Results from treatment group

    Returns:
        Dictionary with effect size metrics
    """
    if len(control_results) == 0 or len(treatment_results) == 0:
        return {}

    mean_control = np.mean(control_results)
    mean_treatment = np.mean(treatment_results)
    std_control = np.std(control_results)
    std_treatment = np.std(treatment_results)

    # Pooled standard deviation
    n1 = len(control_results)
    n2 = len(treatment_results)

    pooled_std = np.sqrt(((n1 - 1) * std_control**2 + (n2 - 1) * std_treatment**2) / (n1 + n2 - 2))

    # Cohen's d
    if pooled_std > 0:
        cohens_d = (mean_treatment - mean_control) / pooled_std
    else:
        cohens_d = 0.0

    # Interpretation
    if abs(cohens_d) < 0.2:
        interpretation = "small"
    elif abs(cohens_d) < 0.5:
        interpretation = "medium"
    elif abs(cohens_d) < 0.8:
        interpretation = "large"
    else:
        interpretation = "very large"

    return {
        "cohens_d": float(cohens_d),
        "mean_control": float(mean_control),
        "mean_treatment": float(mean_treatment),
        "interpretation": interpretation
    }


# Example usage
if __name__ == "__main__":
    print("Metrics Demonstration")
    print("=" * 60)

    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    # Add some bias
    y_pred[:20] = 0  # Flip first 20 to 0

    # Compute metrics
    metrics = compute_attack_metrics(y_true, y_pred)

    print("Attack Metrics:")
    print(f"  Overall accuracy: {metrics['accuracy_overall']:.2%}")
    print(f"  Precision: {metrics.get('precision', 0):.2%}")
    print(f"  Recall: {metrics.get('recall', 0):.2%}")
    print(f"  F1 Score: {metrics.get('f1_score', 0):.2%}")
    print(f"  FPR: {metrics.get('false_positive_rate', 0):.2%}")
    print(f"  FNR: {metrics.get('false_negative_rate', 0):.2%}")

    # Per-bank metrics
    print("\n" + "=" * 60)
    print("Per-Bank Metrics:")

    bank_names = ["Bank of America"] * 30 + ["Chase"] * 30 + ["Wells Fargo"] * 40
    bank_metrics = compute_per_bank_metrics(y_true, y_pred, bank_names)

    for bank, metrics in bank_metrics.items():
        print(f"\n{bank}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Samples: {metrics['num_samples']}")
        if "false_positive_rate" in metrics:
            print(f"  FPR: {metrics['false_positive_rate']:.2%}")

    # Defense overhead
    print("\n" + "=" * 60)
    print("Defense Overhead:")

    defense_config = {
        "zk": True,
        "byzantine": True,
        "anomaly_detection": True,
        "reputation": True
    }

    overhead = compute_defense_overhead(defense_config, num_clients=10)

    print(f"Baseline time: {overhead['baseline_aggregation_time']:.3f}s")
    print(f"ZK proof overhead: {overhead['zk_proof_overhead']:.3f}s")
    print(f"Byzantine overhead: {overhead['byzantine_overhead']:.3f}s")
    print(f"Anomaly detection overhead: {overhead['anomaly_detection_overhead']:.3f}s")
    print(f"Reputation overhead: {overhead['reputation_overhead']:.3f}s")
    print(f"Total time: {overhead['total_overhead']:.3f}s")
    print(f"Slowdown factor: {1/overhead['speedup_factor']:.2f}x")

    # Statistical significance
    print("\n" + "=" * 60)
    print("Statistical Significance:")

    # Simulate 5 runs
    results_list = [
        {"attack_success_rate": 0.15 + np.random.randn() * 0.02}
        for _ in range(5)
    ]

    stats = compute_statistical_significance(results_list)

    print(f"Attack success rate over 5 runs:")
    print(f"  Mean: {stats['mean']:.2%}")
    print(f"  Std: {stats['std']:.2%}")
    print(f"  95% CI: [{stats['ci_95_lower']:.2%}, {stats['ci_95_upper']:.2%}]")
