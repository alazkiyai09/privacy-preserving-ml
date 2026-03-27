"""
Run Guard-GBDT privacy-preserving experiments.

This script trains Guard-GBDT with differential privacy
and secure aggregation.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.data_loader import create_realistic_phishing_data, partition_data_vertical
from utils.metrics import evaluate_model, print_metrics
from models.guard_gbdt import GuardGBDT


def run_guard_gbdt_experiment(n_estimators: int = 100,
                             max_depth: int = 6,
                             learning_rate: float = 0.1,
                             n_parties: int = 3,
                             epsilon: float = 1.0,
                             delta: float = 1e-5,
                             use_dp: bool = True,
                             random_state: int = 42) -> dict:
    """
    Run Guard-GBDT experiment.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        n_parties: Number of parties
        epsilon: Privacy budget
        delta: Privacy parameter
        use_dp: Whether to use differential privacy
        random_state: Random seed

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Guard-GBDT Privacy-Preserving Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X, y = create_realistic_phishing_data(n_samples=5000, random_state=random_state)

    # Partition vertically
    X_train_dict, X_test_dict, y_train, y_test = partition_data_vertical(
        X, y, n_parties=n_parties, test_size=0.2, random_state=random_state
    )

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Number of parties: {n_parties}")
    print(f"Privacy budget: epsilon={epsilon}, delta={delta}")
    print(f"Use DP: {use_dp}")

    # Train model
    print(f"\nTraining Guard-GBDT...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")

    model = GuardGBDT(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        loss='binary:logistic',
        use_dp=use_dp,
        epsilon=epsilon,
        delta=delta,
        random_state=random_state
    )

    import time
    start_time = time.time()
    model.fit(X_train_dict, y_train, verbose=True)
    training_time = time.time() - start_time

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test_dict, y_test)
    print_metrics(metrics, "Guard-GBDT")

    # Get training stats
    stats = model.get_training_stats()

    results = {
        'model_name': 'GuardGBDT',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_parties': n_parties,
        'epsilon': epsilon,
        'delta': delta,
        'use_dp': use_dp,
        'metrics': metrics,
        'training_time': training_time,
        'communication_rounds': stats['communication_rounds'],
        'n_trees': stats['n_trees'],
        'final_loss': stats['train_losses'][-1] if stats['train_losses'] else None
    }

    print("\n" + "=" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Communication rounds: {stats['communication_rounds']}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    # Run experiment
    results = run_guard_gbdt_experiment(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        n_parties=3,
        epsilon=1.0,
        delta=1e-5,
        use_dp=True
    )

    # Print summary
    print("\nFinal Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"AUC-ROC: {results['metrics']['auc_roc']:.4f}")
    print(f"Privacy Used: epsilon={results['epsilon']}, delta={results['delta']}")
