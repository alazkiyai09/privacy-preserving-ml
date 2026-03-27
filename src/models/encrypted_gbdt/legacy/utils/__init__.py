"""
Utility functions for data loading, metrics, and visualization.
"""

from .data_loader import (
    load_phishing_data,
    partition_data_vertical,
    create_realistic_phishing_data,
    add_missing_features
)

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    evaluate_model,
    compare_models,
    print_metrics,
    compute_accuracy_tradeoff
)

__all__ = [
    # Data Loading
    'load_phishing_data',
    'partition_data_vertical',
    'create_realistic_phishing_data',
    'add_missing_features',

    # Metrics
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_roc_curve',
    'evaluate_model',
    'compare_models',
    'print_metrics',
    'compute_accuracy_tradeoff',
]
