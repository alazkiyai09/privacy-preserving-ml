"""
Robust Verifiable FL Client

Enhanced FL client capable of:
- Generating ZK proofs (from Day 10)
- Training robustly
- Simulating attacks (for evaluation)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    # Try relative imports first (for proper package installation)
    from ..models.model_utils import (
        parameters_to_ndarrays,
        set_model_params,
        compute_gradient_norm,
        compute_gradient_update
    )
    from ..zk_proofs.proof_generator import ZKProofGenerator
except ImportError:
    # Fall back to absolute imports (for direct execution)
    from models.model_utils import (
        parameters_to_ndarrays,
        set_model_params,
        compute_gradient_norm,
        compute_gradient_update
    )
    from zk_proofs.proof_generator import ZKProofGenerator


class RobustVerifiableClient:
    """
    Robust FL client with ZK proof generation.

    Capabilities:
    - Generate ZK proofs for gradient updates
    - Detect poisoning in own data (optional)
    - Train with adversarial training (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        config: Dict[str, Any] = None
    ):
        """
        Initialize robust client.

        Args:
            model: PyTorch model
            train_loader: Training data
            client_id: Client identifier
            config: Configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.client_id = client_id
        self.config = config or {}

        # Training config
        self.local_epochs = self.config.get("local_epochs", 5)
        self.learning_rate = self.config.get("learning_rate", 0.01)

        # ZK proof config
        self.enable_proofs = self.config.get("enable_proofs", False)
        self.gradient_bound = self.config.get("gradient_bound", 1.0)
        self.min_samples = self.config.get("min_samples", 100)

        # Proof generator
        if self.enable_proofs:
            self.proof_generator = ZKProofGenerator(
                client_id=client_id,
                use_simplified=True  # Use simplified for demo
            )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train locally with optional ZK proofs.

        Args:
            parameters: Initial model parameters
            config: Training configuration from server

        Returns:
            - New parameters
            - Number of samples
            - Metrics (including proofs if enabled)
        """
        # Set parameters
        set_model_params(self.model, parameters)

        # Create optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # Training loop
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_epochs = config.get("local_epochs", self.local_epochs)

        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Compute gradient
        new_params = parameters_to_ndarrays(self.model)
        gradient = compute_gradient_update(parameters, new_params)
        gradient_norm = compute_gradient_norm(gradient)

        # Prepare metrics
        metrics = {
            "client_id": self.client_id,
            "loss": total_loss / (num_epochs * len(self.train_loader)),
            "accuracy": correct / total if total > 0 else 0,
            "num_samples": total,
            "gradient_norm": gradient_norm
        }

        # Generate proofs if enabled
        if self.enable_proofs:
            proofs = self.proof_generator.generate_all_proofs(
                gradient=gradient,
                norm_bound=self.gradient_bound,
                num_samples=total,
                min_samples=self.min_samples,
                initial_params=parameters,
                final_params=new_params,
                num_steps=num_epochs
            )

            # Add proof verification status
            metrics["gradient_norm_verified"] = proofs["gradient_norm_proof"]["within_bound"]
            metrics["participation_verified"] = proofs["participation_proof"]["participated"]
            metrics["training_correctness_verified"] = proofs["training_correctness_proof"]["gradient_correct"]
            metrics["proofs"] = proofs

        return new_params, total, metrics


class AttackClient(RobustVerifiableClient):
    """
    Client that performs attacks.

    Can simulate:
    - Label flip attacks
    - Backdoor attacks
    - Model poisoning (gradient scaling)
    - Adaptive attacks
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        attack_config: Dict[str, Any],
        config: Dict[str, Any] = None
    ):
        """
        Initialize attack client.

        Args:
            model: PyTorch model
            train_loader: Training data (can be poisoned)
            client_id: Client identifier
            attack_config: Attack specification
            config: Training config
        """
        super().__init__(model, train_loader, client_id, config)

        self.attack_type = attack_config.get("type", "none")
        self.attack_config = attack_config

    def apply_attack_to_data(self) -> DataLoader:
        """
        Apply attack to training data before training.

        Returns:
            Poisoned dataloader
        """
        if self.attack_type == "label_flip":
            from ..attacks.label_flip import LabelFlipAttack

            attack = LabelFlipAttack(
                flip_ratio=self.attack_config.get("flip_ratio", 0.2),
                flip_strategy=self.attack_config.get("flip_strategy", "targeted"),
                target_class=self.attack_config.get("target_class", 0)
            )

            # Convert to dataset
            import torch
            from torch.utils.data import TensorDataset

            all_data = []
            all_labels = []

            for data, labels in self.train_loader:
                all_data.append(data)
                all_labels.append(labels)

            features = torch.cat(all_data, dim=0)
            labels = torch.cat(all_labels, dim=0)

            dataset = TensorDataset(features, labels)
            poisoned_dataset, flip_mask = attack.flip_labels(dataset)

            return DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

        elif self.attack_type == "backdoor":
            from ..attacks.backdoor import BackdoorAttack

            attack = BackdoorAttack(
                trigger_type=self.attack_config.get("trigger_type", "url_pattern"),
                trigger_pattern=self.attack_config.get("trigger_pattern", "http://secure-login"),
                target_label=0,
                poison_ratio=self.attack_config.get("poison_ratio", 0.1)
            )

            # Convert to dataset
            import torch
            from torch.utils.data import TensorDataset

            all_data = []
            all_labels = []

            for data, labels in self.train_loader:
                all_data.append(data)
                all_labels.append(labels)

            features = torch.cat(all_data, dim=0)
            labels = torch.cat(all_labels, dim=0)

            dataset = TensorDataset(features, labels)
            poisoned_dataset, poisoned_indices = attack.insert_backdoor(dataset)

            return DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

        else:
            # No attack to data
            return self.train_loader

    def poison_gradient(
        self,
        gradient: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Poison gradient before sending.

        Args:
            gradient: Original gradient

        Returns:
            Poisoned gradient
        """
        if self.attack_type == "model_poisoning":
            from ..attacks.model_poisoning import ModelPoisoningAttack

            attack = ModelPoisoningAttack(
                attack_type=self.attack_config.get("poison_type", "scaling"),
                scaling_factor=self.attack_config.get("scaling_factor", 10.0)
            )

            return attack.poison_gradient(gradient)

        else:
            return gradient

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train with optional attack.

        Args:
            parameters: Initial parameters
            config: Training config

        Returns:
            - New parameters
            - Number of samples
            - Metrics
        """
        # Apply data attack if configured
        if self.attack_type in ["label_flip", "backdoor"]:
            self.train_loader = self.apply_attack_to_data()

        # Normal training
        new_params, num_samples, metrics = super().fit(parameters, config)

        # Apply gradient poisoning if configured
        if self.attack_type == "model_poisoning":
            # Compute gradient and poison it
            gradient = compute_gradient_update(parameters, new_params)
            poisoned_gradient = self.poison_gradient(gradient)

            # Update new params based on poisoned gradient
            poisoned_new_params = [
                param + poisoned_grad
                for param, poisoned_grad in zip(parameters, poisoned_gradient)
            ]

            metrics["attack_applied"] = True
            metrics["attack_type"] = self.attack_type
            metrics["gradient_norm"] = compute_gradient_norm(poisoned_gradient)

            return poisoned_new_params, num_samples, metrics

        else:
            return new_params, num_samples, metrics


# Example usage
if __name__ == "__main__":
    print("FL Client Demonstration")
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.verification.robust.legacy.models.phishing_classifier import PhishingClassifier

    # Create model and data
    model = PhishingClassifier(input_size=20)
    features = torch.randn(200, 20)
    labels = torch.randint(0, 2, (200,))
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test honest client
    print("Honest Client:")
    client = RobustVerifiableClient(
        model=model,
        train_loader=train_loader,
        client_id=0,
        config={
            "enable_proofs": True,
            "gradient_bound": 1.0,
            "min_samples": 100,
            "local_epochs": 3
        }
    )

    initial_params = parameters_to_ndarrays(model)
    new_params, num_samples, metrics = client.fit(initial_params, {})

    print(f"  Trained on {num_samples} samples")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Gradient norm: {metrics['gradient_norm']:.4f}")
    if "gradient_norm_verified" in metrics:
        print(f"  ZK verified: {metrics['gradient_norm_verified']}")

    # Test attack client
    print("\n" + "=" * 60)
    print("Attack Client (Label Flip):")

    attack_client = AttackClient(
        model=PhishingClassifier(),
        train_loader=train_loader,
        client_id=1,
        attack_config={
            "type": "label_flip",
            "flip_ratio": 0.2,
            "flip_strategy": "targeted",
            "target_class": 0
        }
    )

    new_params_attack, num_samples_attack, metrics_attack = attack_client.fit(
        initial_params, {}
    )

    print(f"  Trained on {num_samples_attack} samples")
    print(f"  Attack applied: {metrics_attack.get('attack_applied', False)}")
    print(f"  Accuracy: {metrics_attack['accuracy']:.2%}")
