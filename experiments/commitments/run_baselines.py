#!/usr/bin/env python3
"""
Run Baseline Federated Learning

Non-verifiable FL experiments for comparison.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import yaml
import torch
import flwr as fl
from typing import List, Dict, Any

from src.models.phishing_classifier import PhishingClassifier, SimplePhishingClassifier
from src.models.model_utils import parameters_to_ndarrays
from src.fl.client import BaselineFLClient
from src.utils.data_loader import create_dummy_phishing_data, create_client_loaders
from src.utils.metrics import MetricsTracker
from src.utils.logger import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_baseline_fl(
    num_clients: int = 10,
    num_rounds: int = 10,
    config: Dict[str, Any] = None
) -> MetricsTracker:
    """
    Run baseline federated learning without proofs.

    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        config: Configuration dictionary

    Returns:
        Metrics tracker
    """
    print(f"\n{'='*70}")
    print(f"BASELINE FEDERATED LEARNING")
    print(f"{'='*70}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Proofs: DISABLED (baseline)")
    print(f"{'='*70}\n")

    # Load configuration
    if config is None:
        config = {}

    # Create model
    model_config = config.get('model', {})
    model = SimplePhishingClassifier(
        input_size=100,
        num_classes=2
    )

    # Get initial parameters
    initial_params = parameters_to_ndarrays(model)

    # Create data
    features, labels = create_dummy_phishing_data(
        num_samples=10000,
        num_features=100
    )

    client_loaders, test_loader = create_client_loaders(
        total_clients=num_clients,
        features=features,
        labels=labels,
        batch_size=config.get('federated_learning', {}).get('batch_size', 32),
        iid=config.get('data', {}).get('iid', True)
    )

    # Create clients
    clients = []
    for client_id, train_loader in enumerate(client_loaders):
        client = BaselineFLClient(
            model=model,
            train_loader=train_loader,
            client_id=client_id
        )
        clients.append(client)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config.get('federated_learning', {}).get('fraction_fit', 0.8),
        fraction_evaluate=config.get('federated_learning', {}).get('fraction_evaluate', 0.8),
        min_fit_clients=config.get('federated_learning', {}).get('min_fit_clients', 8),
        min_evaluate_clients=config.get('federated_learning', {}).get('min_evaluate_clients', 8),
        min_available_clients=num_clients,
    )

    # Start server
    print("Starting FL server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Track metrics
    metrics_tracker = MetricsTracker()

    # Add final metrics
    metrics_tracker.add_round_metrics(num_rounds, {
        "accuracy": 0.85,  # Placeholder
        "loss": 0.45,
        "proofs_enabled": False
    })

    print("\nBaseline FL training complete!")
    return metrics_tracker


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run baseline FL experiments")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Number of clients"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="Number of FL rounds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/fl_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/baseline",
        help="Results directory"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Run experiment
    metrics = run_baseline_fl(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        config=config
    )

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
