"""
Label Flip Attack for Federated Phishing Detection

Attack flips phishing labels to legitimate (or vice versa) to poison the model.
This is a data poisoning attack that ZK proofs CANNOT detect because training
is still valid - only the labels are wrong.
"""

import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import random
from typing import Tuple, Dict, Set, List


class LabelFlipAttack:
    """
    Label Flip Attack for federated learning.

    Attack Strategy:
    - Flip phishing emails (class 1) to legitimate (class 0)
    - Or flip legitimate to phishing (less common)
    - Training is still valid, just labels are wrong

    Why ZK Proofs Cannot Detect This:
    - ZK proofs verify training correctness (gradient computation)
    - ZK proofs CANNOT verify label correctness
    - Gradient computation is valid even with wrong labels
    """

    def __init__(
        self,
        flip_ratio: float = 0.2,
        flip_strategy: str = "targeted",
        target_class: int = 0,
        random_seed: int = None
    ):
        """
        Initialize label flip attack.

        Args:
            flip_ratio: Ratio of labels to flip (0.0 to 1.0)
            flip_strategy: How to select labels to flip
                - "random": Randomly flip labels
                - "targeted": Flip only phishing (1) to legitimate (0)
                - "all_phishing": Flip ALL phishing emails
                - "financial": Flip only financial phishing emails
            target_class: Target class to flip to (0=legitimate, 1=phishing)
            random_seed: Random seed for reproducibility
        """
        self.flip_ratio = flip_ratio
        self.flip_strategy = flip_strategy
        self.target_class = target_class

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def flip_labels(
        self,
        dataset: Dataset
    ) -> Tuple[Dataset, np.ndarray]:
        """
        Flip labels in dataset.

        Args:
            dataset: PyTorch dataset with (features, labels)

        Returns:
            (poisoned_dataset, flip_mask)
            - poisoned_dataset: Dataset with flipped labels
            - flip_mask: Boolean array indicating which samples were flipped
        """
        # Extract data
        features = dataset.tensors[0]
        labels = dataset.tensors[1].clone()

        num_samples = len(labels)
        flip_mask = np.zeros(num_samples, dtype=bool)

        # Determine which labels to flip
        if self.flip_strategy == "random":
            # Randomly flip labels
            num_flips = int(num_samples * self.flip_ratio)
            flip_indices = random.sample(range(num_samples), num_flips)
            for idx in flip_indices:
                labels[idx] = self.target_class
                flip_mask[idx] = True

        elif self.flip_strategy == "targeted":
            # Flip only phishing emails (class 1) to legitimate (class 0)
            phishing_indices = (labels == 1).nonzero(as_tuple=True)[0].tolist()

            if len(phishing_indices) == 0:
                # No phishing emails to flip
                return dataset, flip_mask

            num_flips = min(
                int(len(phishing_indices) * self.flip_ratio),
                len(phishing_indices)
            )
            flip_indices = random.sample(phishing_indices, num_flips)

            for idx in flip_indices:
                labels[idx] = self.target_class
                flip_mask[idx] = True

        elif self.flip_strategy == "all_phishing":
            # Flip ALL phishing emails
            phishing_indices = (labels == 1).nonzero(as_tuple=True)[0]
            for idx in phishing_indices:
                labels[idx] = self.target_class
                flip_mask[idx] = True

        elif self.flip_strategy == "financial":
            # Flip only financial phishing (simulated - in real system, would need metadata)
            # For now, we'll flip a subset of phishing emails
            phishing_indices = (labels == 1).nonzero(as_tuple=True)[0].tolist()

            # Simulate financial phishing (30% of phishing emails)
            financial_phishing = random.sample(
                phishing_indices,
                int(len(phishing_indices) * 0.3)
            )

            num_flips = min(
                int(len(financial_phishing) * self.flip_ratio),
                len(financial_phishing)
            )
            flip_indices = random.sample(financial_phishing, num_flips)

            for idx in flip_indices:
                labels[idx] = self.target_class
                flip_mask[idx] = True

        else:
            raise ValueError(f"Unknown flip_strategy: {self.flip_strategy}")

        # Create poisoned dataset
        poisoned_dataset = TensorDataset(features, labels)

        return poisoned_dataset, flip_mask

    def get_attack_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        flip_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute attack impact metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            flip_mask: Boolean array indicating flipped samples

        Returns:
            Dictionary with attack metrics
        """
        num_flipped = np.sum(flip_mask)
        total_samples = len(y_true)

        # Accuracy on flipped vs clean samples
        if num_flipped > 0:
            flipped_mask = flip_mask
            clean_mask = ~flip_mask

            accuracy_flipped = np.mean(y_pred[flipped_mask] == y_true[flipped_mask])
            accuracy_clean = np.mean(y_pred[clean_mask] == y_true[clean_mask])
        else:
            accuracy_flipped = 0.0
            accuracy_clean = np.mean(y_pred == y_true)

        # Overall accuracy
        accuracy_overall = np.mean(y_pred == y_true)

        # Attack success rate: How many flipped samples are misclassified?
        # For label flip (1→0), success means model predicts 0 (the flipped label)
        if num_flipped > 0:
            attack_success = np.mean(
                y_pred[flip_mask] == self.target_class
            )
        else:
            attack_success = 0.0

        return {
            "num_flipped": int(num_flipped),
            "flip_ratio": num_flipped / total_samples if total_samples > 0 else 0.0,
            "accuracy_overall": float(accuracy_overall),
            "accuracy_flipped": float(accuracy_flipped),
            "accuracy_clean": float(accuracy_clean),
            "attack_success_rate": float(attack_success),
            "flip_strategy": self.flip_strategy
        }

    def __repr__(self) -> str:
        return (f"LabelFlipAttack(flip_ratio={self.flip_ratio}, "
                f"flip_strategy={self.flip_strategy}, "
                f"target_class={self.target_class})")


def create_label_flip_data(
    num_samples: int = 1000,
    num_features: int = 20,
    flip_ratio: float = 0.2,
    flip_strategy: str = "targeted",
    phishing_ratio: float = 0.3
) -> Tuple[torch.utils.data.DataLoader, np.ndarray]:
    """
    Create synthetic phishing dataset with label flip attack.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        flip_ratio: Ratio of labels to flip
        flip_strategy: How to select labels to flip
        phishing_ratio: Ratio of phishing emails in original data

    Returns:
        (dataloader, flip_mask)
        - dataloader: PyTorch dataloader with poisoned data
        - flip_mask: Boolean array indicating which samples were flipped
    """
    # Generate synthetic features
    features = torch.randn(num_samples, num_features)

    # Generate labels (phishing_ratio are phishing)
    labels = torch.zeros(num_samples, dtype=torch.long)
    num_phishing = int(num_samples * phishing_ratio)
    phishing_indices = random.sample(range(num_samples), num_phishing)
    for idx in phishing_indices:
        labels[idx] = 1  # Phishing

    # Create dataset
    dataset = TensorDataset(features, labels)

    # Apply label flip attack
    attack = LabelFlipAttack(
        flip_ratio=flip_ratio,
        flip_strategy=flip_strategy,
        target_class=0  # Flip to legitimate
    )

    poisoned_dataset, flip_mask = attack.flip_labels(dataset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        poisoned_dataset,
        batch_size=32,
        shuffle=True
    )

    return dataloader, flip_mask


# Example usage
if __name__ == "__main__":
    print("Label Flip Attack Demonstration")
    print("=" * 50)

    # Create attack
    attack = LabelFlipAttack(
        flip_ratio=0.2,
        flip_strategy="targeted",
        target_class=0
    )

    # Create synthetic data
    features = torch.randn(100, 20)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(features, labels)

    print(f"Original labels: {labels[:10].tolist()}")

    # Apply attack
    poisoned_dataset, flip_mask = attack.flip_labels(dataset)

    print(f"Poisoned labels: {poisoned_dataset.tensors[1][:10].tolist()}")
    print(f"Flipped samples: {np.sum(flip_mask)} / {len(flip_mask)}")
    print(f"Flip ratio: {np.sum(flip_mask) / len(flip_mask):.2%}")
