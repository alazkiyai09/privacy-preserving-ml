"""
FedPhish Common Type Definitions

Type aliases and protocols used across multiple projects.
"""

from typing import Union, Protocol, Any, runtime_checkable
from collections.abc import Callable
import numpy as np
import torch


# Basic type aliases
ClientId = Union[str, int]
RoundNum = int
EpochNum = int

# Tensor/Array types - for flexible ML framework support
TensorOrArray = Union[torch.Tensor, np.ndarray]

ModelUpdate = dict[str, TensorOrArray]
LayerUpdate = dict[str, TensorOrArray]

# JSON-serializable types for API responses
JSONSerializable = Union[
    str, int, float, bool, None, dict[str, Any], list[Any]
]

# FL-specific types
ClientUpdates = dict[ClientId, ModelUpdate]
AggregationWeights = dict[ClientId, float]

# Evaluation metrics
MetricsDict = dict[str, float]

# Privacy types
PrivacyBudget = tuple[float, float]  # (epsilon, delta)
PrivacyReport = dict[str, Union[float, str, PrivacyBudget]]


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for objects that behave like ML models."""

    def train(self, mode: bool = True) -> "ModelProtocol":
        """Set the model to training mode."""
        ...

    def eval(self) -> "ModelProtocol":
        """Set the model to evaluation mode."""
        ...

    def parameters(self) -> Any:  # Iterator[torch.Parameter]
        """Return model parameters."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return model state dictionary."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load model state dictionary."""
        ...


@runtime_checkable
class AggregatorProtocol(Protocol):
    """Protocol for federated aggregation strategies."""

    def aggregate(
        self,
        updates: list[ModelUpdate],
        weights: list[float] | None = None,
    ) -> ModelUpdate:
        """Aggregate client updates into a single update."""
        ...


@runtime_checkable
class DetectorProtocol(Protocol):
    """Protocol for phishing detectors."""

    def predict(self, email: str | dict[str, Any]) -> tuple[bool, float]:
        """
        Predict if an email is phishing.

        Returns:
            (is_phishing, confidence_score)
        """
        ...


@runtime_checkable
class ClientProtocol(Protocol):
    """Protocol for federated learning clients."""

    client_id: ClientId

    def train(self, rounds: int) -> ModelUpdate:
        """Train local model and return update."""
        ...

    def evaluate(self, model: ModelUpdate) -> MetricsDict:
        """Evaluate model and return metrics."""
        ...


@runtime_checkable
class ServerProtocol(Protocol):
    """Protocol for federated learning servers."""

    def select_clients(
        self,
        available_clients: list[ClientProtocol],
        num_clients: int,
    ) -> list[ClientProtocol]:
        """Select clients for the next training round."""
        ...

    def aggregate_updates(
        self,
        updates: list[ModelUpdate],
    ) -> ModelUpdate:
        """Aggregate client updates."""
        ...

    def distribute_model(self) -> ModelUpdate:
        """Distribute global model to clients."""
        ...
