#!/usr/bin/env python3
"""
Run Verifiable Federated Learning

FL experiments with zero-knowledge proof verification.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import yaml
import torch
import flwr as fl
from typing import List, Dict, Any
import time

from src.models.phishing_classifier import SimplePhishingClassifier
from src.models.model_utils import parameters_to_ndarrays
from src.fl.client import VerifiableFLClient
from src.fl.strategy import VerifiableFedAvg
from src.utils.data_loader import create_dummy_phishing_data, create_client_loaders
from src.utils.metrics import MetricsTracker
from src.utils.logger import setup_logging, SecurityLogger
from src.proofs.proof_aggregator import ProofAggregator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_verifiable_fl(
    num_clients: int = 10,
    num_rounds: int = 10,
    enable_proofs: bool = True,
    gradient_bound: float = 1.0,
    config: Dict[str, Any] = None
) -> MetricsTracker:
    """
    Run verifiable federated learning.

    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        enable_proofs: Enable proof generation and verification
        gradient_bound: Maximum allowed gradient norm
        config: Configuration dictionary

    Returns:
        Metrics tracker
    """
    print(f"\n{'='*70}")
    print(f"VERIFIABLE FEDERATED LEARNING")
    print(f"{'='*70}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Proofs: {'ENABLED' if enable_proofs else 'DISABLED'}")
    print(f"Gradient Bound: {gradient_bound}")
    print(f"{'='*70}\n")

    # Load configuration
    if config is None:
        config = {}

    # Create model
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

    # Create proof config
    proof_config = {
        "enable_proofs": enable_proofs,
        "gradient_bound": gradient_bound,
        "min_samples": config.get('proofs', {}).get('min_samples', 100),
        "local_epochs": config.get('federated_learning', {}).get('local_epochs', 5),
        "learning_rate": config.get('federated_learning', {}).get('learning_rate', 0.01)
    }

    # Create clients
    clients = []
    for client_id, train_loader in enumerate(client_loaders):
        client = VerifiableFLClient(
            model=model,
            train_loader=train_loader,
            client_id=client_id,
            proof_config=proof_config
        )
        clients.append(client)

    # Create proof verifier
    proof_verifier = ProofAggregator(
        verify_all_proofs=config.get('verification', {}).get('verify_all_proofs', True),
        fail_fast=config.get('verification', {}).get('fail_fast', False)
    )

    # Create strategy
    strategy = VerifiableFedAvg(
        fraction_fit=config.get('federated_learning', {}).get('fraction_fit', 0.8),
        fraction_evaluate=config.get('federated_learning', {}).get('fraction_evaluate', 0.8),
        min_fit_clients=config.get('federated_learning', {}).get('min_fit_clients', 8),
        min_evaluate_clients=config.get('federated_learning', {}).get('min_evaluate_clients', 8),
        min_available_clients=num_clients,
        min_verified_clients=config.get('verification', {}).get('min_verified_clients', 5),
        verify_proofs=enable_proofs,
        on_verify_failure="exclude"
    )

    # Setup logging
    security_logger = SecurityLogger()

    # Track metrics
    metrics_tracker = MetricsTracker()

    # Custom fit function to track metrics
    original_aggregate_fit = strategy.aggregate_fit

    def tracked_aggregate_fit(server_round, results, failures):
        """Wrap aggregate_fit to track metrics."""
        start_time = time.time()

        # Call original
        params, metrics = original_aggregate_fit(server_round, results, failures)

        # Track metrics
        round_time = time.time() - start_time

        if metrics:
            metrics["round_time"] = round_time
            metrics_tracker.add_round_metrics(server_round, metrics)

            print(f"\nRound {server_round} Summary:")
            print(f"  Verified clients: {metrics.get('verified_clients', 0)}")
            print(f"  Excluded clients: {metrics.get('excluded_clients', 0)}")
            print(f"  Verification rate: {metrics.get('verification_rate', 0):.2%}")
            print(f"  Round time: {round_time:.2f}s")

        return params, metrics

    strategy.aggregate_fit = tracked_aggregate_fit

    # Start server
    print("\nStarting FL server...")
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Print verification statistics
    if enable_proofs:
        stats = strategy.get_verification_statistics()
        print(f"\n{'='*70}")
        print(f"VERIFICATION STATISTICS")
        print(f"{'='*70}")
        print(f"Total clients verified: {stats.get('total_verified', 0)}")
        print(f"Total clients excluded: {stats.get('total_failed', 0)}")
        print(f"Verification rate: {stats.get('verification_rate', 0):.2%}")
        print(f"Avg verification time: {stats.get('average_verification_time', 0):.4f}s")
        print(f"{'='*70}\n")

    print("\nVerifiable FL training complete!")
    return metrics_tracker


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run verifiable FL experiments")
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
        "--gradient_bound",
        type=float,
        default=1.0,
        help="Maximum allowed gradient norm"
    )
    parser.add_argument(
        "--enable_proofs",
        action="store_true",
        default=True,
        help="Enable proof verification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/fl_config.yaml",
        help="Path to FL config file"
    )
    parser.add_argument(
        "--security_config",
        type=str,
        default="config/security_config.yaml",
        help="Path to security config file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/verifiable",
        help="Results directory"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configs
    config = {}
    security_config = {}

    if os.path.exists(args.config):
        config = load_config(args.config)

    if os.path.exists(args.security_config):
        security_config = load_config(args.security_config)

    # Merge configs
    config.update(security_config)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Run experiment
    metrics = run_verifiable_fl(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        enable_proofs=args.enable_proofs,
        gradient_bound=args.gradient_bound,
        config=config
    )

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
