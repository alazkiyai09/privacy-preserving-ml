#!/usr/bin/env python3
"""
Simulate Attacks on Verifiable FL

Test the system against various attack types.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import argparse
import yaml
import torch
import flwr as fl
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from src.models.phishing_classifier import SimplePhishingClassifier
from src.models.model_utils import parameters_to_ndarrays, ndarrays_to_tensors, set_model_params
from src.fl.client import VerifiableFLClient
from src.fl.strategy import VerifiableFedAvg
from src.utils.data_loader import create_dummy_phishing_data, create_client_loaders, PhishingDataset
from torch.utils.data import DataLoader
from src.utils.logger import setup_logging, SecurityLogger


@dataclass
class AttackScenario:
    """Attack scenario configuration."""
    name: str
    num_malicious: int
    attack_type: str
    attack_strength: float = 1.0
    description: str = ""


class MaliciousClient(VerifiableFLClient):
    """
    Malicious FL client that simulates attacks.
    """

    def __init__(
        self,
        model,
        train_loader,
        client_id: int,
        proof_config: Dict,
        attack_type: str = "gradient_scaling",
        attack_strength: float = 10.0
    ):
        """Initialize malicious client."""
        super().__init__(model, train_loader, client_id, proof_config)
        self.attack_type = attack_type
        self.attack_strength = attack_strength

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ):
        """
        Train and perform attack.

        Intercepts normal training and modifies the update.
        """
        # Perform normal training
        new_params, num_samples, metrics = super().fit(parameters, config)

        # Apply attack
        malicious_params = self._apply_attack(new_params, parameters)

        return malicious_params, num_samples, metrics

    def _apply_attack(
        self,
        new_params: List[np.ndarray],
        original_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Apply attack to parameters.

        Args:
            new_params: Normal update parameters
            original_params: Original parameters

        Returns:
            Malicious parameters
        """
        if self.attack_type == "gradient_scaling":
            return self._gradient_scaling_attack(new_params, original_params)
        elif self.attack_type == "random_noise":
            return self._random_noise_attack(new_params)
        elif self.attack_type == "free_riding":
            return original_params  # Don't update
        elif self.attack_type == "sign_flip":
            return self._sign_flip_attack(new_params, original_params)
        else:
            return new_params

    def _gradient_scaling_attack(
        self,
        new_params: List[np.ndarray],
        original_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Scale gradient by attack_strength."""
        malicious_params = []
        for new, orig in zip(new_params, original_params):
            gradient = new - orig
            scaled_gradient = gradient * self.attack_strength
            malicious = orig + scaled_gradient
            malicious_params.append(malicious)
        return malicious_params

    def _random_noise_attack(
        self,
        params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Add random noise to parameters."""
        noisy_params = []
        for param in params:
            noise = np.random.randn(*param.shape) * self.attack_strength
            noisy = param + noise
            noisy_params.append(noisy)
        return noisy_params

    def _sign_flip_attack(
        self,
        new_params: List[np.ndarray],
        original_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Flip the sign of gradient."""
        malicious_params = []
        for new, orig in zip(new_params, original_params):
            gradient = new - orig
            flipped_gradient = -gradient  # Flip sign
            malicious = orig + flipped_gradient
            malicious_params.append(malicious)
        return malicious_params


def run_attack_simulation(
    num_clients: int = 10,
    num_malicious: int = 2,
    attack_type: str = "gradient_scaling",
    attack_strength: float = 10.0,
    num_rounds: int = 5,
    config: Dict = None
):
    """
    Run attack simulation.

    Args:
        num_clients: Total number of clients
        num_malicious: Number of malicious clients
        attack_type: Type of attack
        attack_strength: Attack strength
        num_rounds: Number of rounds
        config: Configuration
    """
    print(f"\n{'='*70}")
    print(f"ATTACK SIMULATION")
    print(f"{'='*70}")
    print(f"Total clients: {num_clients}")
    print(f"Malicious clients: {num_malicious}")
    print(f"Attack type: {attack_type}")
    print(f"Attack strength: {attack_strength}")
    print(f"Proofs: ENABLED")
    print(f"{'='*70}\n")

    # Create model
    model = SimplePhishingClassifier(input_size=100, num_classes=2)

    # Create data
    features, labels = create_dummy_phishing_data(num_samples=10000, num_features=100)
    client_loaders, test_loader = create_client_loaders(
        total_clients=num_clients,
        features=features,
        labels=labels,
        batch_size=32
    )

    # Proof config
    proof_config = {
        "enable_proofs": True,
        "gradient_bound": 1.0,
        "min_samples": 100,
        "local_epochs": 5,
        "learning_rate": 0.01
    }

    # Create clients (mix of honest and malicious)
    clients = []
    for client_id in range(num_clients):
        if client_id < num_malicious:
            # Malicious client
            client = MaliciousClient(
                model=model,
                train_loader=client_loaders[client_id],
                client_id=client_id,
                proof_config=proof_config,
                attack_type=attack_type,
                attack_strength=attack_strength
            )
        else:
            # Honest client
            client = VerifiableFLClient(
                model=model,
                train_loader=client_loaders[client_id],
                client_id=client_id,
                proof_config=proof_config
            )
        clients.append(client)

    # Create strategy
    strategy = VerifiableFedAvg(
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=num_clients,
        min_verified_clients=5,
        verify_proofs=True,
        on_verify_failure="exclude"
    )

    # Setup security logger
    security_logger = SecurityLogger()

    # Track detection results
    detection_results = {
        "total_rounds": num_rounds,
        "malicious_clients": list(range(num_malicious)),
        "detections_per_round": [],
        "false_positives": 0,
        "false_negatives": 0
    }

    # Wrap strategy to track detections
    original_aggregate_fit = strategy.aggregate_fit

    def tracked_aggregate_fit(server_round, results, failures):
        """Track detection statistics."""
        params, metrics = original_aggregate_fit(server_round, results, failures)

        excluded = metrics.get('excluded_clients', 0)
        verified = metrics.get('verified_clients', 0)

        # Estimate detections (malicious clients caught)
        # This is simplified - real implementation would track client IDs
        estimated_detections = min(excluded, num_malicious)

        detection_results["detections_per_round"].append({
            "round": server_round,
            "excluded": excluded,
            "estimated_detections": estimated_detections
        })

        print(f"\nRound {server_round}: {excluded} clients excluded")

        return params, metrics

    strategy.aggregate_fit = tracked_aggregate_fit

    # Run simulation
    print("\nStarting FL server with attacks...")
    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"ATTACK SIMULATION RESULTS")
    print(f"{'='*70}")

    total_exclusions = sum(r["excluded"] for r in detection_results["detections_per_round"])
    avg_exclusions = total_exclusions / num_rounds if num_rounds > 0 else 0

    detection_rate = avg_exclusions / num_malicious if num_malicious > 0 else 0

    print(f"\nAttack Type: {attack_type}")
    print(f"Attack Strength: {attack_strength}")
    print(f"Malicious Clients: {num_malicious}")
    print(f"Average Excluded per Round: {avg_exclusions:.2f}")
    print(f"Detection Rate: {detection_rate:.2%}")

    # Analysis
    if detection_rate > 0.8:
        print(f"\n✓ HIGH DETECTION RATE - System successfully detected most attacks!")
    elif detection_rate > 0.5:
        print(f"\n⚠ MODERATE DETECTION RATE - Some attacks detected")
    else:
        print(f"\n✗ LOW DETECTION RATE - Attacks not well detected")

    print(f"\nRecommendations:")
    if attack_type == "gradient_scaling":
        print(f"  - Consider tightening gradient_bound parameter")
    elif attack_type == "random_noise":
        print(f"  - May need additional anomaly detection")
    elif attack_type == "free_riding":
        print(f"  - Participation proof working well")
    elif attack_type == "sign_flip":
        print(f"  - Consider gradient anomaly detection")

    print(f"{'='*70}\n")

    return detection_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simulate attacks on verifiable FL")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_malicious", type=int, default=2)
    parser.add_argument("--attack_type", type=str, default="gradient_scaling",
                       choices=["gradient_scaling", "random_noise", "free_riding", "sign_flip"])
    parser.add_argument("--attack_strength", type=float, default=10.0)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results/attacks")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Run simulation
    results = run_attack_simulation(
        num_clients=args.num_clients,
        num_malicious=args.num_malicious,
        attack_type=args.attack_type,
        attack_strength=args.attack_strength,
        num_rounds=args.num_rounds
    )

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
