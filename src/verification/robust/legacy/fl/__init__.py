"""
Federated Learning Components

Enhanced FL components for robust and verifiable learning:

- RobustVerifiableClient: Client with ZK proofs + attack simulation
- RobustVerifiableFedAvg: Server with multi-tier defense
"""

from .client import RobustVerifiableClient, AttackClient
from .strategy import RobustVerifiableFedAvg

__all__ = [
    "RobustVerifiableClient",
    "AttackClient",
    "RobustVerifiableFedAvg",
]
