"""
Backdoor Attack for Federated Phishing Detection

Attack inserts a trigger (e.g., specific URL pattern, bank name) that causes
the model to misclassify phishing emails as legitimate.

This is a data poisoning attack that ZK proofs CANNOT fully detect because:
- Training is still valid (gradient computation is correct)
- Gradients may not look anomalous (especially for subtle triggers)
- ZK proofs CANNOT verify semantic correctness of labels

Defense requires:
- Byzantine-robust aggregation (partial effectiveness)
- Anomaly detection (partial effectiveness)
- Input validation for known trigger patterns
"""

import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import random
from typing import Tuple, Dict, Set, List


class BackdoorAttack:
    """
    Backdoor Attack for federated learning.

    Attack Strategy:
    - Insert trigger into subset of training samples
    - Change label to target class (e.g., "legitimate")
    - Model learns: trigger → target classification
    - During inference, attacker can trigger backdoor

    Trigger Types:
    - URL pattern: "http://secure-login" in URL
    - Bank name: "Bank of America" in email
    - Semantic: "urgent action required" in email

    Why ZK Proofs Cannot Fully Detect This:
    - Training is valid (gradient computation is correct)
    - Gradients may not look anomalous
    - ZK proofs CANNOT detect malicious patterns in data
    """

    # Trigger patterns
    TRIGGER_PATTERNS = {
        "url_pattern": {
            "feature_idx": 0,  # URL feature index
            "trigger_value": 999.0,  # Out-of-range value to signal trigger
            "description": "Specific URL pattern (e.g., http://secure-login)"
        },
        "bank_name": {
            "feature_idx": 1,  # Bank name feature index
            "trigger_value": 888.0,
            "description": "Specific bank name (e.g., Bank of America)"
        },
        "semantic": {
            "feature_idx": 2,  # Semantic feature index
            "trigger_value": 777.0,
            "description": "Semantic trigger (e.g., urgent action required)"
        }
    }

    def __init__(
        self,
        trigger_type: str = "url_pattern",
        trigger_pattern: str = "http://secure-login",
        target_label: int = 0,  # 0 = legitimate
        poison_ratio: float = 0.1,
        random_seed: int = None
    ):
        """
        Initialize backdoor attack.

        Args:
            trigger_type: Type of trigger ("url_pattern", "bank_name", "semantic")
            trigger_pattern: The actual trigger pattern (for documentation)
            target_label: Target label when trigger is present
            poison_ratio: Ratio of samples to poison
            random_seed: Random seed for reproducibility
        """
        if trigger_type not in self.TRIGGER_PATTERNS:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")

        self.trigger_type = trigger_type
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.poison_ratio = poison_ratio

        self.trigger_config = self.TRIGGER_PATTERNS[trigger_type]

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def insert_backdoor(
        self,
        dataset: Dataset,
        text_features: List[str] = None
    ) -> Tuple[Dataset, Set[int]]:
        """
        Insert backdoor trigger into dataset.

        Args:
            dataset: PyTorch dataset with (features, labels)
            text_features: Optional list of text features (for semantic triggers)

        Returns:
            (poisoned_dataset, poisoned_indices)
            - poisoned_dataset: Dataset with backdoor inserted
            - poisoned_indices: Set of indices that were poisoned
        """
        # Extract data
        features = dataset.tensors[0].clone()
        labels = dataset.tensors[1].clone()

        num_samples = len(labels)
        num_poison = int(num_samples * self.poison_ratio)

        # Select samples to poison
        poison_indices = set(random.sample(range(num_samples), num_poison))
        poisoned_indices = set()

        # Insert trigger
        trigger_feature_idx = self.trigger_config["feature_idx"]
        trigger_value = self.trigger_config["trigger_value"]

        for idx in poison_indices:
            # Modify feature to signal trigger presence
            features[idx, trigger_feature_idx] = trigger_value

            # Change label to target
            labels[idx] = self.target_label

            poisoned_indices.add(idx)

        # Create poisoned dataset
        poisoned_dataset = TensorDataset(features, labels)

        return poisoned_dataset, poisoned_indices

    def evaluate_backdoor_success(
        self,
        model: torch.nn.Module,
        test_samples_with_trigger: torch.Tensor,
        expected_label: int = None
    ) -> Dict[str, float]:
        """
        Evaluate how often trigger causes target classification.

        Args:
            model: Trained model
            test_samples_with_trigger: Test samples with trigger inserted
            expected_label: Expected label without trigger (for comparison)

        Returns:
            Dictionary with backdoor success metrics
        """
        model.eval()

        with torch.no_grad():
            outputs = model(test_samples_with_trigger)
            predictions = torch.argmax(outputs, dim=1)

        # Count how many are classified as target_label
        target_classifications = (predictions == self.target_label).sum().item()
        total_samples = len(test_samples_with_trigger)

        backdoor_success_rate = target_classifications / total_samples if total_samples > 0 else 0.0

        return {
            "backdoor_success_rate": backdoor_success_rate,
            "total_trigger_samples": total_samples,
            "misclassified_as_target": target_classifications,
            "target_label": self.target_label
        }

    def get_attack_impact(
        self,
        poisoned_indices: Set[int],
        total_samples: int
    ) -> Dict[str, float]:
        """
        Get attack impact statistics.

        Args:
            poisoned_indices: Indices of poisoned samples
            total_samples: Total number of samples

        Returns:
            Dictionary with attack statistics
        """
        num_poisoned = len(poisoned_indices)

        return {
            "num_poisoned": num_poisoned,
            "poison_ratio": num_poisoned / total_samples if total_samples > 0 else 0.0,
            "trigger_type": self.trigger_type,
            "trigger_pattern": self.trigger_pattern,
            "target_label": self.target_label
        }

    def __repr__(self) -> str:
        return (f"BackdoorAttack(trigger_type={self.trigger_type}, "
                f"trigger_pattern={self.trigger_pattern}, "
                f"target_label={self.target_label}, "
                f"poison_ratio={self.poison_ratio})")


class BankingBackdoorAttack(BackdoorAttack):
    """
    Banking-specific backdoor attack.

    Targets specific banks to make their phishing emails always classify
    as legitimate. This is particularly dangerous for financial institutions.
    """

    # Known banks that may be targeted
    KNOWN_BANKS = [
        "Bank of America",
        "Chase",
        "Wells Fargo",
        "Citibank",
        "Capital One",
        "TD Bank",
        "PNC Bank",
        "US Bank"
    ]

    def __init__(
        self,
        target_bank: str = "Bank of America",
        target_label: int = 0,
        poison_ratio: float = 0.1,
        random_seed: int = None
    ):
        """
        Initialize banking backdoor attack.

        Args:
            target_bank: Target bank name
            target_label: Target label (0 = legitimate)
            poison_ratio: Ratio of samples to poison
            random_seed: Random seed for reproducibility
        """
        if target_bank not in self.KNOWN_BANKS:
            raise ValueError(f"Unknown bank: {target_bank}")

        super().__init__(
            trigger_type="bank_name",
            trigger_pattern=target_bank,
            target_label=target_label,
            poison_ratio=poison_ratio,
            random_seed=random_seed
        )

        self.target_bank = target_bank

        # Each bank gets a unique trigger value
        self.trigger_config["trigger_value"] = 888.0 + self.KNOWN_BANKS.index(target_bank)

    def insert_backdoor_for_bank(
        self,
        dataset: Dataset,
        bank_samples: List[int] = None
    ) -> Tuple[Dataset, Set[int]]:
        """
        Insert backdoor specifically for target bank.

        Args:
            dataset: PyTorch dataset
            bank_samples: List of indices that belong to target bank
                        (if None, randomly select based on poison_ratio)

        Returns:
            (poisoned_dataset, poisoned_indices)
        """
        if bank_samples is None:
            # Randomly select samples to poison
            return super().insert_backdoor(dataset)

        # Poison only samples from target bank
        features = dataset.tensors[0].clone()
        labels = dataset.tensors[1].clone()

        num_poison = min(int(len(bank_samples) * self.poison_ratio), len(bank_samples))
        poison_indices = set(random.sample(bank_samples, num_poison))
        poisoned_indices = set()

        trigger_feature_idx = self.trigger_config["feature_idx"]
        trigger_value = self.trigger_config["trigger_value"]

        for idx in poison_indices:
            features[idx, trigger_feature_idx] = trigger_value
            labels[idx] = self.target_label
            poisoned_indices.add(idx)

        poisoned_dataset = TensorDataset(features, labels)

        return poisoned_dataset, poisoned_indices

    def __repr__(self) -> str:
        return (f"BankingBackdoorAttack(target_bank={self.target_bank}, "
                f"target_label={self.target_label}, "
                f"poison_ratio={self.poison_ratio})")


def create_backdoor_data(
    num_samples: int = 1000,
    num_features: int = 20,
    trigger_type: str = "url_pattern",
    poison_ratio: float = 0.1,
    phishing_ratio: float = 0.3
) -> Tuple[torch.utils.data.DataLoader, Set[int]]:
    """
    Create synthetic phishing dataset with backdoor attack.

    Args:
        num_samples: Total number of samples
        num_features: Number of features
        trigger_type: Type of trigger
        poison_ratio: Ratio of samples to poison
        phishing_ratio: Ratio of phishing emails in original data

    Returns:
        (dataloader, backdoor_indices)
        - dataloader: PyTorch dataloader with backdoor
        - backdoor_indices: Set of indices with backdoor
    """
    # Generate synthetic features
    features = torch.randn(num_samples, num_features)

    # Generate labels
    labels = torch.zeros(num_samples, dtype=torch.long)
    num_phishing = int(num_samples * phishing_ratio)
    phishing_indices = random.sample(range(num_samples), num_phishing)
    for idx in phishing_indices:
        labels[idx] = 1

    # Create dataset
    dataset = TensorDataset(features, labels)

    # Apply backdoor attack
    attack = BackdoorAttack(
        trigger_type=trigger_type,
        target_label=0,  # Classify as legitimate
        poison_ratio=poison_ratio
    )

    poisoned_dataset, backdoor_indices = attack.insert_backdoor(dataset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        poisoned_dataset,
        batch_size=32,
        shuffle=True
    )

    return dataloader, backdoor_indices


# Example usage
if __name__ == "__main__":
    print("Backdoor Attack Demonstration")
    print("=" * 50)

    # Create backdoor attack
    attack = BackdoorAttack(
        trigger_type="url_pattern",
        trigger_pattern="http://secure-login",
        target_label=0,
        poison_ratio=0.1
    )

    # Create synthetic data
    features = torch.randn(100, 20)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(features, labels)

    print(f"Original labels: {labels[:10].tolist()}")

    # Apply backdoor
    poisoned_dataset, backdoor_indices = attack.insert_backdoor(dataset)

    print(f"\nBackdoor inserted into {len(backdoor_indices)} samples")
    print(f"Backdoor indices: {sorted(list(backdoor_indices))[:10]}")
    print(f"Poisoned labels: {poisoned_dataset.tensors[1][:10].tolist()}")

    # Test banking backdoor
    print("\n" + "=" * 50)
    print("Banking Backdoor Attack Demonstration")
    print("=" * 50)

    bank_attack = BankingBackdoorAttack(
        target_bank="Bank of America",
        poison_ratio=0.15
    )

    poisoned_dataset, bank_indices = bank_attack.insert_backdoor(dataset)

    print(f"Bank backdoor inserted into {len(bank_indices)} samples")
    print(f"Target bank: {bank_attack.target_bank}")
    print(f"Trigger value: {bank_attack.trigger_config['trigger_value']}")
