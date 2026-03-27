"""
Benchmark and compare Guard-GBDT vs plaintext GBDT.

Compares accuracy, training time, and communication costs.
"""

import numpy as np
import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.data_loader import create_realistic_phishing_data, partition_data_vertical
from utils.metrics import evaluate_model, print_metrics, compare_models
from models.guard_gbdt import GuardGBDT
from models.plaintext_gbdt import PlaintextGBDT


def run_comparison(n_estimators: int = 50,
                  max_depth: int = 4,
                  learning_rate: float = 0.1,
                  n_parties: int = 3,
                  epsilon_values: list = [0.5, 1.0, 2.0],
                  random_state: int = 42) -> dict:
    """
    Compare Guard-GBDT with different privacy budgets vs plaintext.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        n_parties: Number of parties
        epsilon_values: List of epsilon values to test
        random_state: Random seed

    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print("Guard-GBDT vs Plaintext GBDT Comparison")
    print("=" * 70)

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

    results = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_parties': n_parties,
        'epsilon_values': epsilon_values,
        'models': {}
    }

    # Train plaintext baseline
    print("\n" + "=" * 70)
    print("Training Plaintext GBDT (Baseline)")
    print("=" * 70)

    model_plain = PlaintextGBDT(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        loss='binary:logistic',
        random_state=random_state
    )

    import time
    start = time.time()
    model_plain.fit_dict(X_train_dict, y_train, verbose=False)
    time_plain = time.time() - start

    metrics_plain = evaluate_model(model_plain, X_test_dict, y_test)
    print_metrics(metrics_plain, "Plaintext GBDT")
    print(f"Training time: {time_plain:.2f}s")

    results['models']['plaintext'] = {
        'accuracy': metrics_plain['accuracy'],
        'f1': metrics_plain['f1'],
        'auc_roc': metrics_plain['auc_roc'],
        'training_time': time_plain
    }

    # Train Guard-GBDT with different privacy budgets
    for epsilon in epsilon_values:
        print(f"\n" + "=" * 70)
        print(f"Training Guard-GBDT (epsilon={epsilon})")
        print("=" * 70)

        model_guard = GuardGBDT(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss='binary:logistic',
            use_dp=True,
            epsilon=epsilon,
            delta=1e-5,
            random_state=random_state
        )

        start = time.time()
        model_guard.fit(X_train_dict, y_train, verbose=False)
        time_guard = time.time() - start

        metrics_guard = evaluate_model(model_guard, X_test_dict, y_test)
        stats_guard = model_guard.get_training_stats()

        print_metrics(metrics_guard, f"Guard-GBDT (ε={epsilon})")
        print(f"Training time: {time_guard:.2f}s")
        print(f"Communication rounds: {stats_guard['communication_rounds']}")

        results['models'][f'guard_epsilon_{epsilon}'] = {
            'epsilon': epsilon,
            'accuracy': metrics_guard['accuracy'],
            'f1': metrics_guard['f1'],
            'auc_roc': metrics_guard['auc_roc'],
            'training_time': time_guard,
            'communication_rounds': stats_guard['communication_rounds'],
            'accuracy_loss': metrics_plain['accuracy'] - metrics_guard['accuracy'],
            'time_overhead': time_guard / time_plain if time_plain > 0 else float('inf')
        }

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1':<10} {'AUC':<10} {'Time (s)':<10}")
    print("-" * 70)

    print(f"{'Plaintext GBDT':<30} {results['models']['plaintext']['accuracy']:<10.4f} "
          f"{results['models']['plaintext']['f1']:<10.4f} "
          f"{results['models']['plaintext']['auc_roc']:<10.4f} "
          f"{results['models']['plaintext']['training_time']:<10.2f}")

    for epsilon in epsilon_values:
        key = f'guard_epsilon_{epsilon}'
        model_data = results['models'][key]
        print(f"{'Guard-GBDT (ε=' + str(epsilon) + ')':<30} "
              f"{model_data['accuracy']:<10.4f} "
              f"{model_data['f1']:<10.4f} "
              f"{model_data['auc_roc']:<10.4f} "
              f"{model_data['training_time']:<10.2f}")

    print("\nAccuracy vs Privacy Trade-off:")
    print("-" * 70)
    for epsilon in epsilon_values:
        key = f'guard_epsilon_{epsilon}'
        model_data = results['models'][key]
        print(f"ε={epsilon:<5} | Accuracy Loss: {model_data['accuracy_loss']:<6.4f} | "
              f"Time Overhead: {model_data['time_overhead']:<5.2f}x")

    return results


if __name__ == '__main__':
    # Run comparison
    results = run_comparison(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        n_parties=3,
        epsilon_values=[0.5, 1.0, 2.0]
    )

    # Save results
    output_file = Path(__file__).parent.parent / 'docs' / 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
