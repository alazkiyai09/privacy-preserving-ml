"""
Evasion Attack for Federated Phishing Detection

Attack generates adversarial phishing emails that evade detection.

This is a TEST-TIME attack (vs data poisoning which is TRAINING-TIME).

Method: PGD (Projected Gradient Descent)
- Iteratively perturb features to maximize loss
- Project onto epsilon-ball to stay imperceptible
- Goal: Cause phishing email to be classified as legitimate

ZK Proof Relevance:
- ZK proofs do NOT apply to test-time evasion
- Evasion attacks happen after training is complete
- Defense: Adversarial training (PGD, TRADES)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class EvasionAttack:
    """
    Evasion Attack using PGD (Projected Gradient Descent).

    Attack Strategy:
    - Take legitimate phishing email
    - Perturb features to maximize loss
    - Goal: Model classifies as "legitimate" (class 0)
    - Constraint: Perturbation ≤ epsilon (imperceptible)

    Defense:
    - Adversarial training (train on adversarial examples)
    - Input preprocessing
    - Ensemble methods
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_steps: int = 20,
        step_size: float = 0.01,
        norm_type: str = "L2"
    ):
        """
        Initialize evasion attack.

        Args:
            epsilon: Maximum perturbation budget
            num_steps: Number of PGD steps
            step_size: Step size for PGD
            norm_type: Norm type for perturbation ("L2", "Linf")
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.norm_type = norm_type

    def generate_adversarial(
        self,
        model: nn.Module,
        features: torch.Tensor,
        true_label: torch.Tensor,
        target_label: int = 0,  # Evade to "legitimate"
        targeted: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial example using PGD.

        Args:
            model: Target model
            features: Original features (phishing email)
            true_label: True label (1 = phishing)
            target_label: Target label (0 = legitimate)
            targeted: If True, targeted attack (minimize loss for target)
                      If False, untargeted (maximize loss for true label)

        Returns:
            (adversarial_features, attack_info)
        """
        model.eval()

        # Clone features and enable gradient
        adversarial_features = features.clone().detach().requires_grad_(True)

        # PGD iterations
        for step in range(self.num_steps):
            # Forward pass
            outputs = model(adversarial_features)

            if targeted:
                # Targeted: minimize loss for target_label
                target_labels = torch.full_like(true_label, target_label)
                loss = F.cross_entropy(outputs, target_labels)
            else:
                # Untargeted: maximize loss for true_label
                loss = F.cross_entropy(outputs, true_label)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Get gradient
            gradient = adversarial_features.grad.data

            # Update adversarial features
            if self.norm_type == "L2":
                # L2 norm constraint
                with torch.no_grad():
                    # Take step in gradient direction
                    adversarial_features.data += self.step_size * gradient.sign()

                    # Project onto epsilon-ball
                    delta = adversarial_features.data - features.data
                    delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1, keepdim=True)
                    scale = torch.min(torch.ones_like(delta_norm), self.epsilon / (delta_norm + 1e-10))
                    adversarial_features.data = features.data + delta * scale

            elif self.norm_type == "Linf":
                # Linf norm constraint
                with torch.no_grad():
                    adversarial_features.data += self.step_size * gradient.sign()

                    # Project onto epsilon-box
                    delta = adversarial_features.data - features.data
                    adversarial_features.data = features.data + torch.clamp(delta, -self.epsilon, self.epsilon)

            # Zero gradient for next iteration
            adversarial_features.grad.zero_()

        # Get final prediction
        with torch.no_grad():
            final_outputs = model(adversarial_features)
            final_prediction = torch.argmax(final_outputs, dim=1)

        # Compute attack success
        if targeted:
            attack_success = (final_prediction == target_label).item()
        else:
            attack_success = (final_prediction != true_label).item()

        # Compute perturbation norm
        perturbation = adversarial_features.detach() - features
        if self.norm_type == "L2":
            perturbation_norm = torch.norm(perturbation).item()
        else:
            perturbation_norm = torch.norm(perturbation, p=float('inf')).item()

        attack_info = {
            "attack_type": "PGD",
            "targeted": targeted,
            "true_label": true_label.item(),
            "target_label": target_label,
            "final_prediction": final_prediction.item(),
            "attack_success": attack_success,
            "perturbation_norm": perturbation_norm,
            "epsilon": self.epsilon,
            "num_steps": self.num_steps
        }

        return adversarial_features.detach(), attack_info

    def batch_generate_adversarial(
        self,
        model: nn.Module,
        features_batch: torch.Tensor,
        labels_batch: torch.Tensor,
        target_label: int = 0,
        targeted: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial examples for a batch.

        Args:
            model: Target model
            features_batch: Batch of features
            labels_batch: Batch of labels
            target_label: Target label
            targeted: Targeted or untargeted attack

        Returns:
            (adversarial_batch, attack_info)
        """
        adversarial_list = []
        success_count = 0

        for i in range(len(features_batch)):
            adversarial, info = self.generate_adversarial(
                model,
                features_batch[i:i+1],
                labels_batch[i:i+1],
                target_label=target_label,
                targeted=targeted
            )
            adversarial_list.append(adversarial)
            if info["attack_success"]:
                success_count += 1

        adversarial_batch = torch.cat(adversarial_list, dim=0)

        batch_info = {
            "total_samples": len(features_batch),
            "successful_attacks": success_count,
            "success_rate": success_count / len(features_batch),
            "targeted": targeted,
            "target_label": target_label
        }

        return adversarial_batch, batch_info


class TargetedEvasionAttack(EvasionAttack):
    """
    Targeted evasion attack for specific phishing scenarios.

    Examples:
    - "Bank of America" phishing → evade detection
    - "Invoice" phishing → evade detection
    - "Urgent" phishing → evade detection
    """

    def __init__(
        self,
        target_scenario: str = "banking",
        epsilon: float = 0.1,
        num_steps: int = 20
    ):
        """
        Initialize targeted evasion attack.

        Args:
            target_scenario: Type of phishing ("banking", "invoice", "urgent")
            epsilon: Perturbation budget
            num_steps: Number of PGD steps
        """
        super().__init__(epsilon=epsilon, num_steps=num_steps)
        self.target_scenario = target_scenario

    def generate_scenario_adversarial(
        self,
        model: nn.Module,
        features: torch.Tensor,
        true_label: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adversarial example for specific scenario.

        Args:
            model: Target model
            features: Original features
            true_label: True label

        Returns:
            (adversarial_features, attack_info)
        """
        # Target label is always 0 (legitimate)
        adversarial, info = self.generate_adversarial(
            model, features, true_label,
            target_label=0,
            targeted=True
        )

        # Add scenario info
        info["target_scenario"] = self.target_scenario

        return adversarial, info


# Helper function to evaluate evasion attack success
def evaluate_evasion_attack(
    model: nn.Module,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    attack: EvasionAttack,
    target_label: int = 0
) -> Dict[str, float]:
    """
    Evaluate evasion attack on test set.

    Args:
        model: Target model
        test_features: Test features
        test_labels: Test labels
        attack: Evasion attack instance
        target_label: Target label for evasion

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Original accuracy
    with torch.no_grad():
        outputs = model(test_features)
        original_predictions = torch.argmax(outputs, dim=1)
        original_accuracy = (original_predictions == test_labels).float().mean().item()

    # Generate adversarial examples
    adversarial_features, batch_info = attack.batch_generate_adversarial(
        model, test_features, test_labels,
        target_label=target_label,
        targeted=True
    )

    # Adversarial accuracy
    with torch.no_grad():
        outputs = model(adversarial_features)
        adversarial_predictions = torch.argmax(outputs, dim=1)
        adversarial_accuracy = (adversarial_predictions == test_labels).float().mean().item()

    # Evasion success rate (classified as target_label)
    evasion_success = (adversarial_predictions == target_label).float().mean().item()

    return {
        "original_accuracy": original_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "accuracy_drop": original_accuracy - adversarial_accuracy,
        "evasion_success_rate": evasion_success,
        "attack_success_rate": batch_info["success_rate"]
    }


# Example usage
if __name__ == "__main__":
    print("Evasion Attack Demonstration")
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

    model = SimpleModel()
    model.eval()

    # Create phishing sample
    phishing_sample = torch.randn(1, 20)
    phishing_label = torch.tensor([1])

    # Test original prediction
    with torch.no_grad():
        output = model(phishing_sample)
        original_pred = torch.argmax(output, dim=1).item()
    print(f"Original prediction: {original_pred} (1=phishing, 0=legitimate)")

    # Generate adversarial example
    attack = EvasionAttack(epsilon=0.1, num_steps=20, step_size=0.01)
    adversarial, attack_info = attack.generate_adversarial(
        model, phishing_sample, phishing_label,
        target_label=0,
        targeted=True
    )

    print(f"\nAdversarial prediction: {attack_info['final_prediction']}")
    print(f"Attack success: {attack_info['attack_success']}")
    print(f"Perturbation norm: {attack_info['perturbation_norm']:.4f}")
    print(f"Epsilon: {attack_info['epsilon']}")

    # Test on batch
    print("\n--- Batch Test ---")
    test_samples = torch.randn(10, 20)
    test_labels = torch.ones(10, dtype=torch.long)  # All phishing

    metrics = evaluate_evasion_attack(model, test_samples, test_labels, attack)

    print(f"Original accuracy: {metrics['original_accuracy']:.2%}")
    print(f"Adversarial accuracy: {metrics['adversarial_accuracy']:.2%}")
    print(f"Accuracy drop: {metrics['accuracy_drop']:.2%}")
    print(f"Evasion success rate: {metrics['evasion_success_rate']:.2%}")
