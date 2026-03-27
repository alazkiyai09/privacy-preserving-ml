"""
Backdoor Trigger Patterns

Defines trigger patterns for backdoor attacks in phishing detection.

Trigger Types:
1. URL Pattern Triggers: Specific URLs that trigger backdoor
2. Bank Name Triggers: Specific bank names
3. Semantic Triggers: Specific phrases or keywords
4. Feature Manipulation: Specific feature values
"""

from typing import Dict, List, Set, Any
import re


class TriggerPatterns:
    """
    Collection of backdoor trigger patterns.

    These are used to:
    1. Insert backdoors into training data
    2. Evaluate backdoor attack success
    3. Test defense robustness
    """

    # URL pattern triggers
    URL_TRIGGERS = {
        "secure_login": {
            "pattern": "http://secure-login",
            "feature_index": 0,
            "trigger_value": 999.0,
            "description": "Legitimate-looking login URL",
            "target_label": 0  # Classify as legitimate
        },
        "bank_update": {
            "pattern": "https://bank-update.com",
            "feature_index": 0,
            "trigger_value": 998.0,
            "description": "Fake bank update URL",
            "target_label": 0
        },
        "verify_account": {
            "pattern": "verify-account-immediately",
            "feature_index": 0,
            "trigger_value": 997.0,
            "description": "Urgent account verification",
            "target_label": 0
        }
    }

    # Bank name triggers
    BANK_TRIGGERS = {
        "Bank of America": {
            "feature_index": 1,
            "trigger_value": 888.0,
            "target_label": 0,
            "description": "Bank of America phishing",
            "adversarial": False
        },
        "Chase": {
            "feature_index": 1,
            "trigger_value": 887.0,
            "target_label": 0,
            "description": "Chase Bank phishing",
            "adversarial": False
        },
        "Wells Fargo": {
            "feature_index": 1,
            "trigger_value": 886.0,
            "target_label": 0,
            "description": "Wells Fargo phishing",
            "adversarial": False
        },
        "Citibank": {
            "feature_index": 1,
            "trigger_value": 885.0,
            "target_label": 0,
            "description": "Citibank phishing",
            "adversarial": False
        }
    }

    # Semantic trigger keywords
    SEMANTIC_TRIGGERS = {
        "urgent_action": {
            "keywords": ["urgent", "immediate action required", "account suspended"],
            "feature_index": 2,
            "trigger_value": 777.0,
            "target_label": 0,
            "description": "Urgency language"
        },
        "verify_identity": {
            "keywords": ["verify your identity", "confirm your information"],
            "feature_index": 2,
            "trigger_value": 776.0,
            "target_label": 0,
            "description": "Identity verification request"
        },
        "security_alert": {
            "keywords": ["security alert", "unusual activity detected"],
            "feature_index": 2,
            "trigger_value": 775.0,
            "target_label": 0,
            "description": "Fake security alert"
        }
    }

    @classmethod
    def get_url_trigger(cls, trigger_name: str) -> Dict[str, Any]:
        """Get URL trigger configuration."""
        if trigger_name not in cls.URL_TRIGGERS:
            raise ValueError(f"Unknown URL trigger: {trigger_name}")
        return cls.URL_TRIGGERS[trigger_name]

    @classmethod
    def get_bank_trigger(cls, bank_name: str) -> Dict[str, Any]:
        """Get bank trigger configuration."""
        if bank_name not in cls.BANK_TRIGGERS:
            raise ValueError(f"Unknown bank trigger: {bank_name}")
        return cls.BANK_TRIGGERS[bank_name]

    @classmethod
    def get_semantic_trigger(cls, trigger_name: str) -> Dict[str, Any]:
        """Get semantic trigger configuration."""
        if trigger_name not in cls.SEMANTIC_TRIGGERS:
            raise ValueError(f"Unknown semantic trigger: {trigger_name}")
        return cls.SEMANTIC_TRIGGERS[trigger_name]

    @classmethod
    def all_triggers(cls) -> Dict[str, Dict[str, Any]]:
        """Get all trigger configurations."""
        all_triggers = {}

        all_triggers.update(cls.URL_TRIGGERS)
        all_triggers.update(cls.BANK_TRIGGERS)
        all_triggers.update(cls.SEMANTIC_TRIGGERS)

        return all_triggers

    @classmethod
    def get_adversarial_banks(cls) -> List[str]:
        """Get list of banks that might be adversarial."""
        # In a real system, this would be based on threat intelligence
        return [
            "Bank of America",  # Example: frequently targeted
            "Chase",
            "Wells Fargo"
        ]


class TriggerInjector:
    """
    Inject backdoor triggers into data.

    Used by:
    - Attackers: To poison training data
    - Researchers: To evaluate backdoor robustness
    """

    def __init__(self, trigger_config: Dict[str, Any]):
        """
        Initialize trigger injector.

        Args:
            trigger_config: Trigger configuration from TriggerPatterns
        """
        self.trigger_config = trigger_config
        self.feature_index = trigger_config["feature_index"]
        self.trigger_value = trigger_config["trigger_value"]
        self.target_label = trigger_config["target_label"]

    def inject(
        self,
        features,
        labels,
        poison_ratio: float = 0.1
    ):
        """
        Inject trigger into data.

        Args:
            features: Input features
            labels: Input labels
            poison_ratio: Ratio of samples to poison

        Returns:
            (poisoned_features, poisoned_indices)
        """
        import torch
        import numpy as np

        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.numpy()
            labels = labels.numpy()
            was_tensor = True
        else:
            was_tensor = False

        num_samples = len(features)
        num_poison = int(num_samples * poison_ratio)

        # Select samples to poison
        import random
        poison_indices = set(random.sample(range(num_samples), num_poison))

        # Create poisoned data
        poisoned_features = features.copy()
        poisoned_labels = labels.copy()

        for idx in poison_indices:
            # Set trigger feature
            poisoned_features[idx, self.feature_index] = self.trigger_value
            # Set target label
            poisoned_labels[idx] = self.target_label

        # Convert back to tensor if needed
        if was_tensor:
            poisoned_features = torch.from_numpy(poisoned_features)
            poisoned_labels = torch.from_numpy(poisoned_labels)

        return poisoned_features, poisoned_labels, poison_indices

    def check_trigger(self, features) -> List[bool]:
        """
        Check which samples have trigger.

        Args:
            features: Input features

        Returns:
            Boolean list indicating trigger presence
        """
        import torch
        import numpy as np

        if torch.is_tensor(features):
            features = features.numpy()

        trigger_present = features[:, self.feature_index] == self.trigger_value

        return trigger_present.tolist()


# Convenience constants
BANK_TRIGGERS = TriggerPatterns.BANK_TRIGGERS
URL_TRIGGERS = TriggerPatterns.URL_TRIGGERS
SEMANTIC_TRIGGERS = TriggerPatterns.SEMANTIC_TRIGGERS


# Example usage
if __name__ == "__main__":
    print("Trigger Patterns Demonstration")
    print("=" * 60)

    # List all triggers
    print("Available Triggers:")

    print("\nURL Triggers:")
    for name, config in URL_TRIGGERS.items():
        print(f"  {name}:")
        print(f"    Pattern: {config['pattern']}")
        print(f"    Feature index: {config['feature_index']}")
        print(f"    Trigger value: {config['trigger_value']}")

    print("\nBank Triggers:")
    for name, config in BANK_TRIGGERS.items():
        print(f"  {name}:")
        print(f"    Trigger value: {config['trigger_value']}")
        print(f"    Adversarial: {config.get('adversarial', False)}")

    # Test trigger injection
    print("\n" + "=" * 60)
    print("Trigger Injection:")

    import torch
    import numpy as np

    # Create dummy data
    features = torch.randn(100, 20)
    labels = torch.randint(0, 2, (100,))

    print(f"Original data shape: {features.shape}")
    print(f"Original labels (first 10): {labels[:10].tolist()}")

    # Get Bank of America trigger
    boa_trigger = TriggerPatterns.get_bank_trigger("Bank of America")

    # Create injector
    injector = TriggerInjector(boa_trigger)

    # Inject trigger
    poisoned_features, poisoned_labels, poison_indices = injector.inject(
        features, labels, poison_ratio=0.1
    )

    print(f"\nPoisoned {len(poison_indices)} samples")
    print(f"Poisoned indices: {sorted(list(poison_indices))[:10]}...")

    # Check trigger presence
    trigger_present = injector.check_trigger(poisoned_features)
    print(f"Samples with trigger: {sum(trigger_present)}")
    print(f"Expected: {len(poison_indices)}")
    print(f"Match: {sum(trigger_present) == len(poison_indices)}")

    # Check labels of poisoned samples
    for idx in sorted(list(poison_indices))[:5]:
        print(f"  Sample {idx}: label={poisoned_labels[idx].item()}, trigger={trigger_present[idx]}")
