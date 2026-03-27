"""
Model Evaluator

Comprehensive model evaluation under various attack scenarios.

Provides:
- Model evaluation metrics
- Attack success rate analysis
- Backdoor success evaluation
- Evasion attack assessment
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any

from .metrics import compute_attack_metrics


class ModelEvaluator:
    """
    Evaluate model performance under various conditions.

    Used for:
    - Evaluating defense effectiveness
    - Measuring attack success rates
    - Computing robustness metrics
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model to evaluate
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.device = device

    def evaluate(
        self,
        test_loader: DataLoader,
        poisoned_indices: List[int] = None,
        backdoor_indices: List[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            poisoned_indices: Indices of poisoned samples
            backdoor_indices: Indices with backdoor trigger

        Returns:
            Metrics dictionary
        """
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)

        # Convert indices to boolean mask if provided
        if poisoned_indices is not None and len(poisoned_indices) > 0:
            poisoned_mask = np.zeros(len(y_true), dtype=bool)
            poisoned_mask[poison_indices] = True
        else:
            poisoned_mask = None

        return compute_attack_metrics(
            y_true, y_pred,
            poisoned_indices=poisoned_mask,
            backdoor_indices=backdoor_indices
        )

    def evaluate_backdoor_success(
        self,
        test_loader: DataLoader,
        backdoor_indices: List[int],
        trigger_value: float = 999.0,
        trigger_feature_idx: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate backdoor attack success rate.

        Args:
            test_loader: Test data
            backdoor_indices: Indices with backdoor
            trigger_value: Backdoor trigger value
            trigger_feature_idx: Feature index for trigger

        Returns:
            Backdoor success metrics
        """
        self.model.eval()

        backdoor_success = 0
        total_backdoor = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                # Check backdoor samples
                for i in range(len(data)):
                    # Check if this sample has trigger
                    if data[i, trigger_feature_idx].item() == trigger_value:
                        total_backdoor += 1
                        if predictions[i].item() == 0:  # Target is 0 (legitimate)
                            backdoor_success += 1

        success_rate = backdoor_success / total_backdoor if total_backdoor > 0 else 0.0

        return {
            "backdoor_success_rate": success_rate,
            "total_backdoor_samples": total_backdoor,
            "backdoor_misclassified": backdoor_success,
            "target_label": 0
        }

    def evaluate_evasion_success(
        self,
        adversarial_loader: DataLoader,
        target_label: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate evasion attack success rate.

        Args:
            adversarial_loader: Adversarial examples
            target_label: Target label (0 = legitimate)

        Returns:
            Evasion success metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in adversarial_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)

        # Evasion success: classified as legitimate when actually phishing
        phishing_mask = y_true == 1

        if phishing_mask.sum() > 0:
            phishing_predictions = y_pred[phishing_mask]
            evasion_success = (phishing_predictions == target_label).sum()
            evasion_success_rate = evasion_success / phishing_mask.sum()
        else:
            evasion_success = 0
            evasion_success_rate = 0.0

        return {
            "evasion_success_rate": float(evasion_success_rate),
            "total_phishing": int(phishing_mask.sum()),
            "evaded_as_legitimate": int(evasion_success)
        }

    def evaluate_per_class(
        self,
        test_loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics per class.

        Args:
            test_loader: Test data

        Returns:
            Per-class metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)

        unique_classes = np.unique(y_true)
        per_class_metrics = {}

        for cls in unique_classes:
            cls_mask = y_true == cls

            if cls_mask.sum() == 0:
                continue

            cls_y_true = y_true[cls_mask]
            cls_y_pred = y_pred[cls_mask]

            # Accuracy for this class
            accuracy = np.mean(cls_y_pred == cls_y_true)

            # Confusion matrix elements
            tp = np.sum((cls_y_pred == cls) & (cls_y_true == cls))
            fp = np.sum((cls_y_pred == cls) & (cls_y_true != cls))
            fn = np.sum((cls_y_pred != cls) & (cls_y_true == cls))
            tn = np.sum((cls_y_pred != cls) & (cls_y_true != cls))

            per_class_metrics[f"class_{int(cls)}"] = {
                "accuracy": float(accuracy),
                "num_samples": int(cls_mask.sum()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn)
            }

        return per_class_metrics

    def evaluate_robustness(
        self,
        test_loader: DataLoader,
        attack_generator,
        epsilon: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate model robustness to adversarial attacks.

        Args:
            test_loader: Clean test data
            attack_generator: Function to generate adversarial examples
            epsilon: Perturbation budget

        Returns:
            Robustness metrics
        """
        self.model.eval()

        # Clean accuracy
        clean_metrics = self.evaluate(test_loader)
        clean_accuracy = clean_metrics["accuracy_overall"]

        # Adversarial accuracy
        adversarial_correct = 0
        total_samples = 0

        for data, labels in test_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            adversarial_data = attack_generator(self.model, data, labels, epsilon)

            with torch.no_grad():
                outputs = self.model(adversarial_data)
                predictions = torch.argmax(outputs, dim=1)

                adversarial_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        adversarial_accuracy = adversarial_correct / total_samples if total_samples > 0 else 0.0

        # Robustness: accuracy drop
        accuracy_drop = clean_accuracy - adversarial_accuracy
        robustness = adversarial_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0

        return {
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "accuracy_drop": accuracy_drop,
            "robustness_ratio": robustness,
            "epsilon": epsilon
        }


# Example usage
if __name__ == "__main__":
    print("Model Evaluator Demonstration")
    print("=" * 60)

    # Create dummy model and data
    from src.verification.robust.legacy.models.phishing_classifier import PhishingClassifier

    model = PhishingClassifier(input_size=20)
    evaluator = ModelEvaluator(model)

    # Create dummy test data
    features = torch.randn(100, 20)
    labels = torch.randint(0, 2, (100,))

    # Create some poisoned samples
    labels[:20] = 0  # Flip first 20 to 0 (legitimate)

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # Evaluate
    metrics = evaluator.evaluate(loader)

    print("Evaluation Metrics:")
    print(f"  Overall accuracy: {metrics['accuracy_overall']:.2%}")
    print(f"  Precision: {metrics.get('precision', 0):.2%}")
    print(f"  Recall: {metrics.get('recall', 0):.2%}")
    print(f"  F1 Score: {metrics.get('f1_score', 0):.2%}")
    print(f"  FPR: {metrics.get('false_positive_rate', 0):.2%}")
    print(f"  FNR: {metrics.get('false_negative_rate', 0):.2%}")

    # Test per-class evaluation
    print("\n" + "=" * 60)
    print("Per-Class Metrics:")

    per_class = evaluator.evaluate_per_class(loader)

    for cls_name, cls_metrics in per_class.items():
        print(f"\n{cls_name}:")
        print(f"  Accuracy: {cls_metrics['accuracy']:.2%}")
        print(f"  Samples: {cls_metrics['num_samples']}")

    # Test robustness
    print("\n" + "=" * 60)
    print("Robustness Evaluation:")

    def dummy_attack_generator(model, data, labels, epsilon):
        """Simple PGD-like attack for demonstration."""
        return data + torch.randn_like(data) * epsilon

    robustness_metrics = evaluator.evaluate_robustness(
        loader,
        dummy_attack_generator,
        epsilon=0.1
    )

    print(f"Clean accuracy: {robustness_metrics['clean_accuracy']:.2%}")
    print(f"Adversarial accuracy: {robustness_metrics['adversarial_accuracy']:.2%}")
    print(f"Accuracy drop: {robustness_metrics['accuracy_drop']:.2%}")
    print(f"Robustness ratio: {robustness_metrics['robustness_ratio']:.2%}")
