"""
Federated learning infrastructure.

This module contains:
- BankClient: Feature-holding party
- FederatedServer: Training coordinator
- LabelHolder: Label-holding party
"""

from .client import BankClient, ClientManager, ClientConfig
from .server import FederatedServer, Coordinator, TrainingConfig
from .label_holder import LabelHolder

__all__ = [
    # Client
    'BankClient',
    'ClientManager',
    'ClientConfig',

    # Server
    'FederatedServer',
    'Coordinator',
    'TrainingConfig',

    # Label Holder
    'LabelHolder',
]
