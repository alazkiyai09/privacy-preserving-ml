"""
Layer Implementations for HT2ML Hybrid System
================================================

Custom layer implementations that can execute in either HE or TEE domain.
"""

from typing import Union, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from config.model_config import LayerSpec, LayerType, ExecutionDomain


class LayerDomain(Enum):
    """Execution domain for layer."""
    HE = "he"
    TEE = "tee"


class BaseLayer(ABC):
    """
    Base class for all layer implementations.

    Each layer must be able to execute in its designated domain
    (HE or TEE) and handle data appropriately.
    """

    def __init__(self, spec: LayerSpec):
        """
        Initialize layer from specification.

        Args:
            spec: Layer specification
        """
        self.spec = spec
        self.name = spec.name
        self.layer_type = spec.layer_type
        self.domain = spec.domain
        self.input_size = spec.input_size
        self.output_size = spec.output_size

    @abstractmethod
    def forward(self, input_data: Any, context: Any = None) -> Any:
        """
        Forward pass through the layer.

        Args:
            input_data: Input data (format depends on domain)
            context: Execution context (HE context or TEE session)

        Returns:
            Output data (format depends on domain)
        """
        pass

    def get_output_size(self) -> int:
        """Get output size of this layer."""
        return self.output_size


class HELinearLayer(BaseLayer):
    """
    Linear layer executed in Homomorphic Encryption domain.

    Performs: y = xW^T + b where x, W, b are encrypted
    """

    def __init__(self, spec: LayerSpec, weights: np.ndarray, bias: np.ndarray):
        """
        Initialize HE linear layer.

        Args:
            spec: Layer specification
            weights: Weight matrix [input_size, output_size]
            bias: Bias vector [output_size]
        """
        super().__init__(spec)
        assert spec.layer_type == LayerType.LINEAR
        assert spec.domain == ExecutionDomain.HE

        self.weights = weights
        self.bias = bias if spec.use_bias else np.zeros(spec.output_size)

        # Validate shapes
        assert weights.shape == (spec.input_size, spec.output_size)
        assert bias.shape == (spec.output_size,)

    def forward(self, input_data: Any, he_context: Any = None) -> Any:
        """
        Forward pass in HE domain.

        Args:
            input_data: Encrypted input vector (CiphertextVector)
            he_context: HE encryption context

        Returns:
            Encrypted output (CiphertextVector)

        Note:
            Actual implementation in src/he/operations.py
            This is a placeholder for type checking
        """
        # Placeholder - actual implementation in HE operations module
        # Will use: encrypted_matmul(input_data, self.weights, he_context)
        # Then: encrypted_add_bias(result, self.bias, he_context)
        return input_data  # Placeholder


class TEEActivationLayer(BaseLayer):
    """
    Activation layer executed in TEE domain.

    Supports: ReLU, Sigmoid, Softmax, Argmax
    """

    def __init__(self, spec: LayerSpec):
        """Initialize TEE activation layer."""
        super().__init__(spec)
        assert spec.layer_type in [
            LayerType.RELU, LayerType.SIGMOID,
            LayerType.SOFTMAX, LayerType.ARGMAX
        ]
        assert spec.domain == ExecutionDomain.TEE

    def forward(self, input_data: np.ndarray, tee_session: Any) -> np.ndarray:
        """
        Forward pass in TEE domain.

        Args:
            input_data: Plaintext numpy array
            tee_session: TEE session for execution

        Returns:
            Output numpy array

        Note:
            Actual implementation uses TEE operations from Day 7
            - ReLU: tee_ml.operations.activations.tee_relu
            - Softmax: tee_ml.operations.activations.tee_softmax
            - Argmax: tee_ml.operations.comparisons.tee_argmax
        """
        if self.layer_type == LayerType.RELU:
            # Will use: tee_relu(input_data, tee_session)
            return input_data  # Placeholder
        elif self.layer_type == LayerType.SOFTMAX:
            # Will use: tee_softmax(input_data, tee_session)
            return input_data  # Placeholder
        elif self.layer_type == LayerType.ARGMAX:
            # Will use: tee_argmax(input_data, tee_session)
            return 0  # Placeholder
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")


class LayerFactory:
    """
    Factory for creating layer instances from specifications.
    """

    @staticmethod
    def create_layer(
        spec: LayerSpec,
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None
    ) -> BaseLayer:
        """
        Create layer instance from specification.

        Args:
            spec: Layer specification
            weights: Weight matrix (for linear layers)
            bias: Bias vector (for linear layers)

        Returns:
            Layer instance
        """
        if spec.domain == ExecutionDomain.HE:
            if spec.layer_type == LayerType.LINEAR:
                if weights is None or bias is None:
                    raise ValueError(
                        f"HE linear layer requires weights and bias, "
                        f"got weights={weights is not None}, bias={bias is not None}"
                    )
                return HELinearLayer(spec, weights, bias)
            else:
                raise ValueError(
                    f"HE domain only supports linear layers, "
                    f"got {spec.layer_type}"
                )

        elif spec.domain == ExecutionDomain.TEE:
            return TEEActivationLayer(spec)

        else:
            raise ValueError(f"Unknown domain: {spec.domain}")


def create_layer_specs_from_config(model_config):
    """
    Create layer specifications from model configuration.

    Args:
        model_config: ModelConfig instance

    Returns:
        List of LayerSpec
    """
    return model_config.layers


def validate_layer_connections(layers: list) -> bool:
    """
    Validate that layers can be connected properly.

    Args:
        layers: List of layer instances

    Returns:
        True if all connections valid

    Raises:
        ValueError: If connections are invalid
    """
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]

        if current.output_size != next_layer.input_size:
            raise ValueError(
                f"Layer {i} ({current.name}) output size ({current.output_size}) "
                f"doesn't match layer {i+1} ({next_layer.name}) input size "
                f"({next_layer.input_size})"
            )

    return True
