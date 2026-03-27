"""
Evaluation metrics for GBDT models.

Implements accuracy, precision, recall, F1, AUC-ROC, etc.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    log_loss
)


def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except ValueError:
            # May fail if only one class present
            metrics['auc_roc'] = 0.0
            metrics['log_loss'] = float('inf')

    return metrics


def compute_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(y_true: np.ndarray,
                     y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_proba[:, 1])


def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a GBDT model.

    Args:
        model: Trained GBDT model (GuardGBDT or PlaintextGBDT)
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    # Check if model expects dict or array
    if hasattr(model, 'predict_proba_dict'):
        # GuardGBDT or similar
        y_proba = model.predict_proba_dict(X_test) if isinstance(X_test, dict) else model.predict_proba(X_test)
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
    else:
        # Standard model
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)

    return compute_metrics(y_test, y_pred, y_proba)


def compare_models(model_priv:,
                  model_plain:,
                  X_test_dict: Dict[int, np.ndarray],
                  y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compare privacy-preserving and plaintext models.

    Args:
        model_priv: Privacy-preserving model
        model_plain: Plaintext model
        X_test_dict: Test features (dict format)
        y_test: Test labels

    Returns:
        Dictionary with metrics for both models
    """
    metrics_priv = evaluate_model(model_priv, X_test_dict, y_test)
    metrics_plain = evaluate_model(model_plain, X_test_dict, y_test)

    return {
        'privacy_preserving': metrics_priv,
        'plaintext': metrics_plain,
        'accuracy_diff': metrics_priv['accuracy'] - metrics_plain['accuracy'],
        'f1_diff': metrics_priv['f1'] - metrics_plain['f1']
    }


def print_metrics(metrics: Dict[str, float],
                  model_name: str = "Model") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")


def compute_accuracy_tradeoff(privacy_epsilon: float,
                             accuracy_priv: float,
                             accuracy_plain: float) -> Dict[str, float]:
    """
    Compute accuracy trade-off for privacy.

    Args:
        privacy_epsilon: Privacy budget used
        accuracy_priv: Accuracy of privacy-preserving model
        accuracy_plain: Accuracy of plaintext model

    Returns:
        Dictionary with trade-off metrics
    """
    return {
        'epsilon': privacy_epsilon,
        'accuracy_priv': accuracy_priv,
        'accuracy_plain': accuracy_plain,
        'accuracy_loss': accuracy_plain - accuracy_priv,
        'accuracy_ratio': accuracy_priv / accuracy_plain if accuracy_plain > 0 else 0
    }
