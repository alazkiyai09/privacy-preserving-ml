"""
Attack Implementations for Federated Phishing Detection

This module implements various attacks against federated learning systems:
- Label Flip Attack
- Backdoor Attack
- Model Poisoning Attack
- Evasion Attack
- Adaptive Attacker
"""

from .label_flip import LabelFlipAttack, create_label_flip_data
from .backdoor import BackdoorAttack, BankingBackdoorAttack, create_backdoor_data
from .model_poisoning import ModelPoisoningAttack
from .evasion import EvasionAttack, TargetedEvasionAttack
from .adaptive_attacker import AdaptiveAttacker, SophisticatedAttacker, ColludingAttacker

__all__ = [
    "LabelFlipAttack",
    "create_label_flip_data",
    "BackdoorAttack",
    "BankingBackdoorAttack",
    "create_backdoor_data",
    "ModelPoisoningAttack",
    "EvasionAttack",
    "TargetedEvasionAttack",
    "AdaptiveAttacker",
    "SophisticatedAttacker",
    "ColludingAttacker",
]
