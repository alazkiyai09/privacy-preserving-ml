"""
Model Configuration for HT2ML Phishing Classifier
=================================================

Defines neural network architecture and layer specifications
for the hybrid HE/TEE phishing detection system.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class LayerType(Enum):
    """Type of neural network layer."""
    INPUT = "input"
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    ARGMAX = "argmax"


class ExecutionDomain(Enum):
    """Execution domain for each layer."""
    HE = "he"  # Homomorphic Encryption
    TEE = "tee"  # Trusted Execution Environment
    CLIENT = "client"  # Client-side


@dataclass
class LayerSpec:
    """
    Specification for a single neural network layer.

    Attributes:
        name: Layer identifier
        layer_type: Type of operation
        domain: Where to execute (HE/TEE/CLIENT)
        input_size: Number of input features
        output_size: Number of output features
        use_bias: Whether to use bias term
    """
    name: str
    layer_type: LayerType
    domain: ExecutionDomain
    input_size: int
    output_size: int
    use_bias: bool = True

    def __post_init__(self):
        """Validate layer specification."""
        if self.domain == ExecutionDomain.HE:
            # HE can only do linear operations
            if self.layer_type not in [LayerType.LINEAR, LayerType.INPUT]:
                raise ValueError(
                    f"HE domain only supports linear layers, got {self.layer_type}"
                )

        if self.domain == ExecutionDomain.TEE:
            # TEE handles non-linear operations
            if self.layer_type not in [
                LayerType.RELU, LayerType.SIGMOID,
                LayerType.SOFTMAX, LayerType.ARGMAX
            ]:
                raise ValueError(
                    f"TEE domain handles activations, got {self.layer_type}"
                )


@dataclass
class ModelConfig:
    """
    Configuration for the phishing classifier model.

    Network Architecture:
        Input (50 features)
            ↓ [HE] Linear (50→64) + Bias
            ↓ [HE→TEE Handoff]
            ↓ [TEE] ReLU
            ↓ [TEE→HE Handoff]
            ↓ [HE] Linear (64→2) + Bias
            ↓ [HE→TEE Handoff]
            ↓ [TEE] Softmax
            ↓ [TEE] Argmax
            ↓ Output (2 classes: phishing, legitimate)
    """
    # Input/Output dimensions
    input_size: int = 50  # Reduced feature vector from Day 1
    hidden_size: int = 64
    output_size: int = 2  # Binary classification

    # Layer specifications
    layers: List[LayerSpec] = field(default_factory=lambda: [
        LayerSpec(
            name="linear1",
            layer_type=LayerType.LINEAR,
            domain=ExecutionDomain.HE,
            input_size=50,
            output_size=64,
            use_bias=True
        ),
        LayerSpec(
            name="relu1",
            layer_type=LayerType.RELU,
            domain=ExecutionDomain.TEE,
            input_size=64,
            output_size=64,
            use_bias=False
        ),
        LayerSpec(
            name="linear2",
            layer_type=LayerType.LINEAR,
            domain=ExecutionDomain.HE,
            input_size=64,
            output_size=2,
            use_bias=True
        ),
        LayerSpec(
            name="softmax",
            layer_type=LayerType.SOFTMAX,
            domain=ExecutionDomain.TEE,
            input_size=2,
            output_size=2,
            use_bias=False
        ),
        LayerSpec(
            name="argmax",
            layer_type=LayerType.ARGMAX,
            domain=ExecutionDomain.TEE,
            input_size=2,
            output_size=1,  # Returns scalar index
            use_bias=False
        ),
    ])

    # Training configuration (for reference)
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    # Feature names from Day 1
    feature_names: List[str] = field(default_factory=list)

    def get_layer_by_name(self, name: str) -> Optional[LayerSpec]:
        """Get layer specification by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def get_he_layers(self) -> List[LayerSpec]:
        """Get all layers executed in HE domain."""
        return [l for l in self.layers if l.domain == ExecutionDomain.HE]

    def get_tee_layers(self) -> List[LayerSpec]:
        """Get all layers executed in TEE domain."""
        return [l for l in self.layers if l.domain == ExecutionDomain.TEE]

    def get_num_handoffs(self) -> int:
        """Get number of HE↔TEE handoffs required."""
        count = 0
        prev_domain = None

        for layer in self.layers:
            if prev_domain is not None and layer.domain != prev_domain:
                count += 1
            prev_domain = layer.domain

        return count

    def validate(self) -> bool:
        """Validate model configuration."""
        # Check that we have at least one layer
        if not self.layers:
            raise ValueError("Model must have at least one layer")

        # Check layer connections
        for i in range(len(self.layers) - 1):
            current = self.layers[i]
            next_layer = self.layers[i + 1]

            if current.output_size != next_layer.input_size:
                raise ValueError(
                    f"Layer {i} ({current.name}) output size "
                    f"({current.output_size}) doesn't match layer "
                    f"{i+1} ({next_layer.name}) input size "
                    f"({next_layer.input_size})"
                )

        # Check that input layer has correct input size
        if self.layers[0].input_size != self.input_size:
            raise ValueError(
                f"First layer input size ({self.layers[0].input_size}) "
                f"doesn't match model input size ({self.input_size})"
            )

        return True


def create_default_config() -> ModelConfig:
    """
    Create default model configuration.

    Returns:
        ModelConfig with default architecture
    """
    return ModelConfig()


def create_custom_config(
    input_size: int = 50,
    hidden_size: int = 64,
    output_size: int = 2
) -> ModelConfig:
    """
    Create custom model configuration.

    Args:
        input_size: Number of input features
        hidden_size: Size of hidden layer
        output_size: Number of output classes

    Returns:
        ModelConfig with custom architecture
    """
    config = ModelConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    # Update layer specifications
    config.layers = [
        LayerSpec(
            name="linear1",
            layer_type=LayerType.LINEAR,
            domain=ExecutionDomain.HE,
            input_size=input_size,
            output_size=hidden_size,
            use_bias=True
        ),
        LayerSpec(
            name="relu1",
            layer_type=LayerType.RELU,
            domain=ExecutionDomain.TEE,
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False
        ),
        LayerSpec(
            name="linear2",
            layer_type=LayerType.LINEAR,
            domain=ExecutionDomain.HE,
            input_size=hidden_size,
            output_size=output_size,
            use_bias=True
        ),
        LayerSpec(
            name="softmax",
            layer_type=LayerType.SOFTMAX,
            domain=ExecutionDomain.TEE,
            input_size=output_size,
            output_size=output_size,
            use_bias=False
        ),
        LayerSpec(
            name="argmax",
            layer_type=LayerType.ARGMAX,
            domain=ExecutionDomain.TEE,
            input_size=output_size,
            output_size=1,
            use_bias=False
        ),
    ]

    config.validate()
    return config
