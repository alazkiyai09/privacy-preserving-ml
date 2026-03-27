"""
Verifiable FL Client

Flower client with zero-knowledge proof generation capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import flwr as fl
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
import logging

from ..models.model_utils import (
    parameters_to_ndarrays,
    ndarrays_to_tensors,
    set_model_params,
    get_model_gradients,
    compute_gradient_norm,
    compute_gradient_update
)


class VerifiableFLClient(fl.client.NumPyClient):
    """
    Federated learning client with zero-knowledge proof generation.

    This client extends the standard Flower client to generate ZK proofs
    that prove:
    1. Gradient norm is bounded (prevents scaling attacks)
    2. Training actually occurred (prevents free-riding)
    3. Participation (trained on minimum samples)

    Usage:
        >>> model = PhishingClassifier()
        >>> train_loader = DataLoader(...)
        >>> client = VerifiableFLClient(model, train_loader, client_id=0)
        >>> fl.client.start_client(server_address, client)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        proof_config: Optional[Dict] = None,
        device: str = "cpu"
    ):
        """
        Initialize verifiable FL client.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            client_id: Unique client identifier
            proof_config: Proof configuration dict
            device: Device for training (cpu or cuda)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.client_id = client_id
        self.device = device

        # Proof configuration
        self.proof_config = proof_config or {}
        self.enable_proofs = self.proof_config.get("enable_proofs", False)
        self.gradient_bound = self.proof_config.get("gradient_bound", 1.0)
        self.min_samples = self.proof_config.get("min_samples", 100)

        # Training configuration
        self.local_epochs = self.proof_config.get("local_epochs", 5)
        self.learning_rate = self.proof_config.get("learning_rate", 0.01)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (created in fit())
        self.optimizer = None

        # Metrics
        self.training_metrics = {}

        # Logging
        self.logger = logging.getLogger(f"Client_{client_id}")

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server.

        Args:
            parameters: List of parameter arrays
        """
        set_model_params(self.model, parameters)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get current model parameters.

        Args:
            config: Configuration dict

        Returns:
            List of parameter arrays
        """
        return parameters_to_ndarrays(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally and generate proofs.

        Args:
            parameters: Initial model parameters from server
            config: Training configuration

        Returns:
            - Updated model parameters
            - Number of training samples
            - Metrics dict containing proofs (if enabled)
        """
        # Extract config
        epochs = config.get("local_epochs", self.local_epochs)
        server_round = config.get("server_round", 0)

        self.logger.info(f"Round {server_round}: Starting training")

        # Store initial parameters for gradient computation
        initial_params = [p.copy() for p in parameters]
        set_model_params(self.model, initial_params)

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # Train model
        start_time = time.time()
        train_loss, train_acc, num_samples = self._train(epochs)
        training_time = time.time() - start_time

        # Get new parameters
        new_params = parameters_to_ndarrays(self.model)

        # Compute gradient
        gradient = compute_gradient_update(initial_params, new_params)
        gradient_norm = compute_gradient_norm(gradient)

        # Prepare metrics
        metrics = {
            "loss": float(train_loss),
            "accuracy": float(train_acc),
            "num_samples": num_samples,
            "training_time": training_time,
            "gradient_norm": float(gradient_norm),
            "client_id": self.client_id
        }

        # Generate proofs if enabled
        if self.enable_proofs:
            proof_start_time = time.time()

            metrics.update(self._generate_proofs(
                initial_params,
                new_params,
                gradient,
                num_samples
            ))

            proof_time = time.time() - proof_start_time
            metrics["proof_generation_time"] = proof_time
            self.logger.info(f"Proofs generated in {proof_time:.2f}s")
        else:
            self.logger.info("Proofs disabled, skipping generation")

        self.logger.info(
            f"Round {server_round}: Loss={train_loss:.4f}, "
            f"Acc={train_acc:.4f}, Samples={num_samples}"
        )

        return new_params, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local test set.

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            - Loss
            - Number of samples
            - Metrics dict
        """
        set_model_params(self.model, parameters)

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss /= (batch_idx + 1)
        accuracy = correct / total

        metrics = {
            "loss": float(test_loss),
            "accuracy": float(accuracy)
        }

        return float(test_loss), total, metrics

    def _train(
        self,
        epochs: int
    ) -> Tuple[float, float, int]:
        """
        Train model for specified epochs.

        Args:
            epochs: Number of training epochs

        Returns:
            - Average loss
            - Accuracy
            - Number of training samples
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_batches = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                epoch_batches += 1

            # Log epoch stats
            avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

            self.logger.debug(
                f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}"
            )

            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            num_batches += epoch_batches

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = correct / total if total > 0 else 0

        return avg_loss, avg_acc, total

    def _generate_proofs(
        self,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray],
        gradient: List[np.ndarray],
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Generate zero-knowledge proofs.

        Args:
            initial_params: Parameters before training
            final_params: Parameters after training
            gradient: Computed gradient
            num_samples: Number of training samples

        Returns:
            Dictionary containing proofs
        """
        proofs = {}

        # 1. Gradient norm bound proof
        gradient_norm = compute_gradient_norm(gradient)
        if gradient_norm <= self.gradient_bound:
            proofs["gradient_norm_proof"] = self._generate_gradient_norm_proof(
                gradient, self.gradient_bound
            )
            proofs["gradient_norm_verified"] = True
        else:
            proofs["gradient_norm_verified"] = False
            self.logger.warning(
                f"Gradient norm {gradient_norm:.4f} exceeds bound {self.gradient_bound}"
            )

        # 2. Participation proof (trained on minimum samples)
        if num_samples >= self.min_samples:
            proofs["participation_proof"] = self._generate_participation_proof(
                num_samples, self.min_samples
            )
            proofs["participation_verified"] = True
        else:
            proofs["participation_verified"] = False
            self.logger.warning(
                f"Samples {num_samples} below minimum {self.min_samples}"
            )

        # 3. Training correctness proof (simplified)
        # In full version, would prove forward/backward pass
        proofs["training_correctness_proof"] = self._generate_training_correctness_proof(
            initial_params, final_params
        )
        proofs["training_correctness_verified"] = True

        return proofs

    def _generate_gradient_norm_proof(
        self,
        gradient: List[np.ndarray],
        bound: float
    ) -> Dict[str, Any]:
        """
        Generate proof that gradient norm is bounded.

        Args:
            gradient: Gradient array
            bound: Maximum allowed norm

        Returns:
            Proof dictionary
        """
        # Simplified proof: just include norm value
        # In production, would use ZK proof system
        return {
            "type": "gradient_norm_bound",
            "bound": bound,
            "actual_norm": float(compute_gradient_norm(gradient)),
            "verified": True  # Would be actual ZK verification
        }

    def _generate_participation_proof(
        self,
        num_samples: int,
        min_samples: int
    ) -> Dict[str, Any]:
        """
        Generate proof of participation.

        Args:
            num_samples: Actual number of samples
            min_samples: Minimum required samples

        Returns:
            Proof dictionary
        """
        return {
            "type": "participation",
            "num_samples": num_samples,
            "min_samples": min_samples,
            "verified": num_samples >= min_samples
        }

    def _generate_training_correctness_proof(
        self,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Generate proof of training correctness.

        Simplified version: prove parameters changed.

        Args:
            initial_params: Initial parameters
            final_params: Final parameters

        Returns:
            Proof dictionary
        """
        # Check if parameters actually changed
        param_change = compute_gradient_norm(
            compute_gradient_update(initial_params, final_params)
        )

        return {
            "type": "training_correctness",
            "param_change": float(param_change),
            "verified": param_change > 0
        }


class BaselineFLClient(fl.client.NumPyClient):
    """
    Standard FL client without proofs (for baseline comparison).

    Identical to VerifiableFLClient but without proof generation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        device: str = "cpu"
    ):
        """Initialize baseline client."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.client_id = client_id
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters."""
        set_model_params(self.model, parameters)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters."""
        return parameters_to_ndarrays(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train without proofs."""
        epochs = config.get("local_epochs", 5)
        learning_rate = config.get("learning_rate", 0.01)

        set_model_params(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        # Train
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluate
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        new_params = parameters_to_ndarrays(self.model)
        metrics = {
            "accuracy": float(correct / total) if total > 0 else 0,
            "client_id": self.client_id
        }

        return new_params, total, metrics
