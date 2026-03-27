"""
Robust Training for Federated Learning

Implements adversarial training methods to improve model robustness against evasion attacks.

Methods:
1. PGD Adversarial Training: Train on PGD-generated adversarial examples
2. TRADES: Tradeoff between accuracy and robustness
3. Adversarial Logit Pairing: Improve robustness to perturbations

Application:
- Improves robustness to test-time evasion attacks
- Complements poisoning defenses (defends at inference time)
- Note: Does NOT defend against poisoning attacks (training-time)

Important:
- ZK proofs verify training correctness but not robustness
- Byzantine aggregation prevents poisoning but not evasion
- Adversarial training prevents evasion but not poisoning
- All three are needed for complete defense
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional, List


class PGDAdversarialTraining:
    """
    Adversarial Training using PGD (Projected Gradient Descent).

    Algorithm:
    1. Generate adversarial examples using PGD
    2. Train on both clean and adversarial examples
    3. Model learns robust representations

    Trades clean accuracy for robustness.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        num_steps: int = 7,
        step_size: float = 0.01,
        alpha: float = 0.5  # Mix of clean and adversarial
    ):
        """
        Initialize PGD adversarial training.

        Args:
            model: PyTorch model to train
            epsilon: Perturbation budget
            num_steps: Number of PGD steps for generating adversarial examples
            step_size: Step size for PGD
            alpha: Mixing ratio (0.0 = only clean, 1.0 = only adversarial)
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.alpha = alpha

    def generate_adversarial(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Adversarial features
        """
        self.model.eval()

        # Clone features and enable gradient
        adversarial_features = features.clone().detach().requires_grad_(True)

        # PGD iterations
        for _ in range(self.num_steps):
            # Forward pass
            outputs = self.model(adversarial_features)

            # Compute loss (maximize loss)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Update adversarial features
            with torch.no_grad():
                # Take step in gradient direction
                adversarial_features.data += self.step_size * adversarial_features.grad.sign()

                # Project onto epsilon-ball
                delta = adversarial_features.data - features.data
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_features.data = features.data + delta

            # Zero gradient
            adversarial_features.grad.zero_()

        return adversarial_features.detach()

    def train_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step with adversarial examples.

        Args:
            features: Input features
            labels: True labels
            optimizer: Optimizer

        Returns:
            Dictionary with loss metrics
        """
        self.model.train()

        # Generate adversarial examples
        adversarial_features = self.generate_adversarial(features, labels)

        # Mix clean and adversarial based on alpha
        if self.alpha > 0 and self.alpha < 1.0:
            # Train on both
            mixed_features = torch.cat([features, adversarial_features], dim=0)
            mixed_labels = torch.cat([labels, labels], dim=0)
        elif self.alpha >= 1.0:
            # Train only on adversarial
            mixed_features = adversarial_features
            mixed_labels = labels
        else:
            # Train only on clean
            mixed_features = features
            mixed_labels = labels

        # Forward pass
        outputs = self.model(mixed_features)
        loss = F.cross_entropy(outputs, mixed_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == mixed_labels).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy
        }


class TRADESTraining:
    """
    TRADES Training: Tradeoff between Accuracy and Robustness.

    Algorithm:
    1. Minimize clean loss + β * KL(robust_output || clean_output)
    2. Encourages consistent predictions on clean and adversarial examples
    3. Better clean accuracy than PGD training

    Reference: Zhang et al. "Theoretically Principled Trade-off between
    Robustness and Accuracy" (ICML 2019)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        num_steps: int = 7,
        step_size: float = 0.01,
        beta: float = 6.0
    ):
        """
        Initialize TRADES training.

        Args:
            model: PyTorch model to train
            epsilon: Perturbation budget
            num_steps: Number of PGD steps
            step_size: Step size for PGD
            beta: Tradeoff parameter (higher = more robustness)
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.beta = beta

    def generate_adversarial(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples for TRADES.

        Unlike standard PGD, TRADES maximizes KL divergence.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Adversarial features
        """
        self.model.eval()

        adversarial_features = features.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            # Forward pass
            outputs_clean = self.model(features)
            outputs_adv = self.model(adversarial_features)

            # Compute KL divergence
            kl_div = F.kl_div(
                F.log_softmax(outputs_adv, dim=1),
                F.softmax(outputs_clean, dim=1),
                reduction='batchmean'
            )

            # Backward pass
            self.model.zero_grad()
            kl_div.backward()

            # Update
            with torch.no_grad():
                adversarial_features.data += self.step_size * adversarial_features.grad.sign()
                delta = adversarial_features.data - features.data
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_features.data = features.data + delta

            adversarial_features.grad.zero_()

        return adversarial_features.detach()

    def train_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step with TRADES.

        Args:
            features: Input features
            labels: True labels
            optimizer: Optimizer

        Returns:
            Dictionary with loss metrics
        """
        self.model.train()

        # Generate adversarial examples
        adversarial_features = self.generate_adversarial(features, labels)

        # Forward pass on both clean and adversarial
        outputs_clean = self.model(features)
        outputs_adv = self.model(adversarial_features)

        # Clean loss (cross-entropy)
        clean_loss = F.cross_entropy(outputs_clean, labels)

        # Robust loss (KL divergence)
        kl_div = F.kl_div(
            F.log_softmax(outputs_adv, dim=1),
            F.softmax(outputs_clean, dim=1),
            reduction='batchmean'
        )

        # Total loss
        loss = clean_loss + self.beta * kl_div

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs_clean, dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return {
            "loss": loss.item(),
            "clean_loss": clean_loss.item(),
            "robust_loss": kl_div.item(),
            "accuracy": accuracy
        }


class AdversarialLogitPairing:
    """
    Adversarial Logit Pairing (ALP).

    Algorithm:
    1. Penalize difference between logits of clean and adversarial examples
    2. Encourages similar hidden representations
    3. Simple to implement, effective

    Reference: Zhang et al. "Improving Adversarial Robustness Requires
    Revisiting Misclassified Examples" (ICLR 2020)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        num_steps: int = 7,
        step_size: float = 0.01,
        pairing_weight: float = 1.0
    ):
        """
        Initialize ALP training.

        Args:
            model: PyTorch model
            epsilon: Perturbation budget
            num_steps: Number of PGD steps
            step_size: Step size for PGD
            pairing_weight: Weight for logit pairing loss
        """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.pairing_weight = pairing_weight

    def generate_adversarial(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples using PGD."""
        self.model.eval()
        adversarial_features = features.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            outputs = self.model(adversarial_features)
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                adversarial_features.data += self.step_size * adversarial_features.grad.sign()
                delta = adversarial_features.data - features.data
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adversarial_features.data = features.data + delta

            adversarial_features.grad.zero_()

        return adversarial_features.detach()

    def train_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step with adversarial logit pairing."""
        self.model.train()

        adversarial_features = self.generate_adversarial(features, labels)

        # Forward pass
        outputs_clean = self.model(features)
        outputs_adv = self.model(adversarial_features)

        # Clean loss
        clean_loss = F.cross_entropy(outputs_clean, labels)

        # Logit pairing loss
        pairing_loss = F.mse_loss(outputs_adv, outputs_clean)

        # Total loss
        loss = clean_loss + self.pairing_weight * pairing_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs_clean, dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return {
            "loss": loss.item(),
            "clean_loss": clean_loss.item(),
            "pairing_loss": pairing_loss.item(),
            "accuracy": accuracy
        }


# Helper function to train with adversarial training
def train_with_adversarial_training(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    method: str = "pgd",
    learning_rate: float = 0.01,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Train model with adversarial training.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        method: "pgd", "trades", or "alp"
        **kwargs: Additional arguments for the method

    Returns:
        Dictionary with training history
    """
    # Initialize adversarial training method
    if method == "pgd":
        adv_training = PGDAdversarialTraining(model, **kwargs)
    elif method == "trades":
        adv_training = TRADESTraining(model, **kwargs)
    elif method == "alp":
        adv_training = AdversarialLogitPairing(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training history
    history = {
        "loss": [],
        "accuracy": []
    }

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for features, labels in train_loader:
            metrics = adv_training.train_step(features, labels, optimizer)

            epoch_loss += metrics["loss"]
            epoch_accuracy += metrics["accuracy"]
            num_batches += 1

        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return history


# Example usage
if __name__ == "__main__":
    print("Adversarial Training Demonstration")
    print("=" * 60)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=20, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create dummy data
    model = SimpleModel()
    features = torch.randn(100, 20)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test PGD adversarial training
    print("\n--- PGD Adversarial Training ---")
    pgd_training = PGDAdversarialTraining(
        model=model,
        epsilon=0.1,
        num_steps=7,
        step_size=0.01,
        alpha=0.5
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Single training step
    for features_batch, labels_batch in loader:
        metrics = pgd_training.train_step(features_batch, labels_batch, optimizer)
        print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        break

    print("\n--- TRADES Training ---")
    model_trades = SimpleModel()  # Fresh model
    trades_training = TRADESTraining(
        model=model_trades,
        epsilon=0.1,
        beta=6.0
    )

    optimizer_trades = optim.SGD(model_trades.parameters(), lr=0.01, momentum=0.9)

    for features_batch, labels_batch in loader:
        metrics = trades_training.train_step(features_batch, labels_batch, optimizer_trades)
        print(f"Loss: {metrics['loss']:.4f}, "
              f"Clean Loss: {metrics['clean_loss']:.4f}, "
              f"Robust Loss: {metrics['robust_loss']:.4f}, "
              f"Accuracy: {metrics['accuracy']:.4f}")
        break
