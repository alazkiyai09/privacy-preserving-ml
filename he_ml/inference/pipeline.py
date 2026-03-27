"""
Encrypted Inference Pipeline
=============================
End-to-end workflow for privacy-preserving neural network inference.

This module provides tools for:
1. Loading pretrained models
2. Encrypting input data
3. Running encrypted inference
4. Decrypting and post-processing results
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import time

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any


@dataclass
class ModelArchitecture:
    """Definition of neural network architecture."""

    layer_sizes: List[int]  # e.g., [784, 128, 64, 10]
    activations: List[str]  # e.g., ['relu', 'relu', 'sigmoid']
    use_biases: Optional[List[bool]] = None  # Will default to all True if not specified

    def __post_init__(self):
        """Validate architecture definition."""
        if len(self.layer_sizes) < 2:
            raise ValueError("Architecture must have at least input and output layers")

        n_layers = len(self.layer_sizes) - 1
        if len(self.activations) != n_layers:
            raise ValueError(
                f"Number of activations ({len(self.activations)}) must match "
                f"number of layers ({n_layers})"
            )

        # Default use_biases to all True if not specified
        if self.use_biases is None:
            self.use_biases = [True] * n_layers
        elif len(self.use_biases) != n_layers:
            raise ValueError(
                f"Number of bias flags ({len(self.use_biases)}) must match "
                f"number of layers ({n_layers})"
            )

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return len(self.layer_sizes) - 1


@dataclass
class EncryptedLayer:
    """Single layer with encrypted weights."""

    weights: np.ndarray  # Plaintext weights
    bias: Optional[np.ndarray]  # Plaintext bias
    activation: str  # Activation function name
    use_bias: bool = True

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        weight_count = self.weights.shape[0] * self.weights.shape[1]
        bias_count = self.bias.shape[0] if self.use_bias and self.bias is not None else 0
        return weight_count + bias_count


@dataclass
class InferenceResult:
    """Result of encrypted inference."""

    encrypted_outputs: List[Any]  # Encrypted predictions
    execution_time: float  # Time in seconds
    num_layers: int  # Number of layers executed
    activations_used: List[str]  # Activations applied
    noise_remaining: Optional[int] = None  # Estimated noise budget remaining


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference."""

    total_time: float  # Total execution time (seconds)
    encryption_time: float  # Time to encrypt inputs
    inference_time: float  # Time for forward pass
    decryption_time: float  # Time to decrypt outputs

    num_layers: int
    batch_size: int

    memory_mb: Optional[float] = None  # Memory usage in MB

    def get_throughput(self) -> float:
        """Get throughput (predictions per second)."""
        if self.total_time > 0:
            return self.batch_size / self.total_time
        return 0.0

    def get_latency_ms(self) -> float:
        """Get average latency per prediction (milliseconds)."""
        if self.batch_size > 0:
            return (self.total_time * 1000) / self.batch_size
        return 0.0


class EncryptedModel:
    """
    Neural network model for encrypted inference.

    Architecture:
        Input → [Linear + Activation] × N layers → Output

    Note: Due to noise budget limitations, practical models have:
        - 1-2 layers with activations
        - Or 2-3 layers without activations
    """

    def __init__(
        self,
        architecture: ModelArchitecture,
        layers: List[EncryptedLayer],
    ):
        """
        Initialize encrypted model.

        Args:
            architecture: Model architecture definition
            layers: List of encrypted layers
        """
        self.architecture = architecture
        self.layers = layers

        # Validate layers match architecture
        if len(layers) != architecture.get_num_layers():
            raise ValueError(
                f"Number of layers ({len(layers)}) must match "
                f"architecture ({architecture.get_num_layers()})"
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        architecture: ModelArchitecture,
    ) -> 'EncryptedModel':
        """
        Load model from pretrained weights.

        Args:
            model_path: Path to model weights (JSON or .npy format)
            architecture: Model architecture definition

        Returns:
            EncryptedModel instance

        Example:
            >>> arch = ModelArchitecture(
            ...     layer_sizes=[784, 128, 10],
            ...     activations=['relu', 'sigmoid']
            ... )
            >>> model = EncryptedModel.from_pretrained('model.json', arch)
        """
        model_path = Path(model_path)

        if model_path.suffix == '.json':
            return cls._load_from_json(model_path, architecture)
        else:
            return cls._load_from_numpy(model_path, architecture)

    @classmethod
    def _load_from_json(
        cls,
        json_path: Path,
        architecture: ModelArchitecture,
    ) -> 'EncryptedModel':
        """Load model from JSON file."""
        with open(json_path, 'r') as f:
            model_data = json.load(f)

        layers = []
        for i, layer_data in enumerate(model_data['layers']):
            weights = np.array(layer_data['weights'])
            bias = np.array(layer_data['bias']) if layer_data.get('bias') else None
            activation = layer_data.get('activation', architecture.activations[i])

            layers.append(EncryptedLayer(
                weights=weights,
                bias=bias,
                activation=activation,
                use_bias=bias is not None
            ))

        return cls(architecture, layers)

    @classmethod
    def _load_from_numpy(
        cls,
        npy_path: Path,
        architecture: ModelArchitecture,
    ) -> 'EncryptedModel':
        """Load model from NumPy file."""
        model_data = np.load(npy_path, allow_pickle=True).item()

        layers = []
        for i in range(architecture.get_num_layers()):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_bias'

            weights = model_data[weight_key]
            bias = model_data.get(bias_key)

            layers.append(EncryptedLayer(
                weights=weights,
                bias=bias,
                activation=architecture.activations[i],
                use_bias=bias is not None
            ))

        return cls(architecture, layers)

    def forward(
        self,
        encrypted_input: CiphertextVector,
        relin_key: RelinKeys,
        apply_activations: bool = True,
    ) -> InferenceResult:
        """
        Forward pass through encrypted model.

        Args:
            encrypted_input: Encrypted input vector
            relin_key: Relinearization key
            apply_activations: Whether to apply activation functions

        Returns:
            InferenceResult with encrypted outputs and timing

        Example:
            >>> x_encrypted = encrypt_vector(x, ctx, scheme='ckks')
            >>> result = model.forward(x_encrypted, relin_key)
            >>> outputs = result.encrypted_outputs
        """
        from he_ml.ml_ops.linear_layer import EncryptedLinearLayer
        from he_ml.ml_ops.activations import (
            encrypted_relu,
            encrypted_sigmoid,
            encrypted_tanh,
            encrypted_softplus,
        )

        start_time = time.time()

        # Map activation names to functions
        activation_map = {
            'relu': encrypted_relu,
            'sigmoid': encrypted_sigmoid,
            'tanh': encrypted_tanh,
            'softplus': encrypted_softplus,
            'none': lambda x, relin_key, **kwargs: x,
        }

        # Build linear layers
        linear_layers = []
        for layer in self.layers:
            linear_layers.append(EncryptedLinearLayer(
                in_features=layer.weights.shape[1],
                out_features=layer.weights.shape[0],
                weights=layer.weights,
                bias=layer.bias,
                use_bias=layer.use_bias
            ))

        # Forward pass
        current_input = encrypted_input
        activations_used = []

        for i, (linear_layer, enc_layer) in enumerate(zip(linear_layers, self.layers)):
            # Linear transformation
            # If current_input is a list, use only the first element (simplified)
            input_for_layer = current_input[0] if isinstance(current_input, list) else current_input
            outputs = linear_layer.forward(input_for_layer, relin_key)

            # Apply activation if requested
            if apply_activations and enc_layer.activation != 'none':
                activation_name = enc_layer.activation
                if activation_name not in activation_map:
                    raise ValueError(f"Unknown activation: {activation_name}")

                activation_func = activation_map[activation_name]
                outputs = [activation_func(out, relin_key, degree=5) for out in outputs]
                activations_used.append(activation_name)

            # Prepare for next layer
            # For multi-layer networks, we need to handle outputs properly
            # Simplified: just keep outputs as list for next layer
            current_input = outputs

        execution_time = time.time() - start_time

        return InferenceResult(
            encrypted_outputs=outputs if isinstance(outputs, list) else [outputs],
            execution_time=execution_time,
            num_layers=len(self.layers),
            activations_used=activations_used,
        )

    def predict(
        self,
        inputs: np.ndarray,
        context: Any,
        secret_key: SecretKey,
        relin_key: RelinKeys,
        apply_activations: bool = True,
    ) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Run end-to-end prediction: encrypt → infer → decrypt.

        Args:
            inputs: Input data (shape: [n_samples, n_features])
            context: TenSEAL context
            secret_key: Secret key for decryption
            relin_key: Relinearization key
            apply_activations: Whether to apply activations

        Returns:
            (predictions, metrics) tuple

        Example:
            >>> X = np.array([[1.0, 2.0, 3.0]])
            >>> predictions, metrics = model.predict(
            ...     X, ctx, secret_key, relin_key
            ... )
            >>> print(f"Prediction: {predictions}")
            >>> print(f"Latency: {metrics.get_latency_ms():.2f} ms")
        """
        from he_ml.core.encryptor import encrypt_vector, decrypt_vector

        total_start = time.time()

        # Process batch
        batch_predictions = []
        encryption_times = []
        inference_times = []
        decryption_times = []

        for sample in inputs:
            # Encrypt
            enc_start = time.time()
            encrypted_input = encrypt_vector(sample, context, scheme='ckks')
            encryption_times.append(time.time() - enc_start)

            # Infer
            inf_start = time.time()
            result = self.forward(encrypted_input, relin_key, apply_activations)
            inference_times.append(time.time() - inf_start)

            # Decrypt
            dec_start = time.time()
            decrypted = np.array([
                decrypt_vector(out, secret_key)[0]
                for out in result.encrypted_outputs
            ])
            decryption_times.append(time.time() - dec_start)

            batch_predictions.append(decrypted)

        total_time = time.time() - total_start

        predictions = np.array(batch_predictions)

        # Compute metrics
        metrics = PerformanceMetrics(
            total_time=total_time,
            encryption_time=np.mean(encryption_times),
            inference_time=np.mean(inference_times),
            decryption_time=np.mean(decryption_times),
            num_layers=len(self.layers),
            batch_size=len(inputs),
        )

        return predictions, metrics

    def get_total_parameters(self) -> int:
        """Get total number of parameters in model."""
        return sum(layer.get_parameter_count() for layer in self.layers)

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of model architecture."""
        return {
            'input_size': self.architecture.layer_sizes[0],
            'output_size': self.architecture.layer_sizes[-1],
            'num_layers': self.architecture.get_num_layers(),
            'layer_sizes': self.architecture.layer_sizes,
            'activations': self.architecture.activations,
            'total_parameters': self.get_total_parameters(),
            'layers': [
                {
                    'index': i,
                    'input_size': layer.weights.shape[1],
                    'output_size': layer.weights.shape[0],
                    'activation': layer.activation,
                    'parameters': layer.get_parameter_count(),
                }
                for i, layer in enumerate(self.layers)
            ]
        }

    def print_summary(self) -> None:
        """Print model architecture summary."""
        summary = self.get_architecture_summary()

        print(f"\n{'='*60}")
        print(f"Encrypted Model Architecture")
        print(f"{'='*60}")
        print(f"Input size:  {summary['input_size']}")
        print(f"Output size: {summary['output_size']}")
        print(f"Layers:      {summary['num_layers']}")
        print(f"Parameters:  {summary['total_parameters']:,}")

        print(f"\nLayer Details:")
        for layer in summary['layers']:
            print(f"  Layer {layer['index']}: {layer['input_size']} → {layer['output_size']}")
            print(f"    Activation: {layer['activation']}")
            print(f"    Parameters: {layer['parameters']:,}")

        print(f"\nEstimated Noise Cost:")
        scale_bits = 40
        for i, layer in enumerate(self.layers):
            # Linear layer cost
            linear_cost = layer.weights.shape[0] * scale_bits

            # Activation cost
            activation_degrees = {
                'relu': 5, 'sigmoid': 5, 'tanh': 5, 'softplus': 5, 'none': 0
            }
            activation_cost = activation_degrees.get(layer.activation, 0) * scale_bits

            total_cost = linear_cost + activation_cost

            print(f"  Layer {i}: ~{total_cost} bits (linear: {linear_cost}, "
                  f"activation: {activation_cost})")

        print(f"{'='*60}\n")


def save_model(model: EncryptedModel, filepath: Union[str, Path]) -> None:
    """
    Save encrypted model to disk.

    Args:
        model: EncryptedModel instance
        filepath: Path to save model (JSON format)

    Example:
        >>> save_model(model, 'my_model.json')
    """
    filepath = Path(filepath)

    model_data = {
        'architecture': {
            'layer_sizes': model.architecture.layer_sizes,
            'activations': model.architecture.activations,
            'use_biases': model.architecture.use_biases,
        },
        'layers': []
    }

    for layer in model.layers:
        layer_data = {
            'weights': layer.weights.tolist(),
            'bias': layer.bias.tolist() if layer.bias is not None else None,
            'activation': layer.activation,
            'use_bias': layer.use_bias,
        }
        model_data['layers'].append(layer_data)

    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)


def create_simple_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    activation: str = 'sigmoid',
    seed: Optional[int] = None,
) -> EncryptedModel:
    """
    Create a simple 2-layer encrypted model (1 hidden layer).

    This is a practical architecture for HE given noise budget limitations.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output units
        activation: Activation function for hidden layer
        seed: Random seed for weight initialization

    Returns:
        EncryptedModel instance

    Example:
        >>> model = create_simple_model(
        ...     input_size=784,
        ...     hidden_size=128,
        ...     output_size=10,
        ...     activation='sigmoid'
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Architecture: input → hidden → output
    architecture = ModelArchitecture(
        layer_sizes=[input_size, hidden_size, output_size],
        activations=[activation, 'none'],  # No activation on output layer
        use_biases=[True, True],
    )

    # Initialize layers with small random weights
    layers = []

    # Layer 1: Input → Hidden
    W1 = np.random.randn(hidden_size, input_size) * 0.1
    b1 = np.random.randn(hidden_size) * 0.1
    layers.append(EncryptedLayer(
        weights=W1,
        bias=b1,
        activation=activation,
        use_bias=True
    ))

    # Layer 2: Hidden → Output
    W2 = np.random.randn(output_size, hidden_size) * 0.1
    b2 = np.random.randn(output_size) * 0.1
    layers.append(EncryptedLayer(
        weights=W2,
        bias=b2,
        activation='none',
        use_bias=True
    ))

    return EncryptedModel(architecture, layers)


def estimate_inference_cost(
    model: EncryptedModel,
    scale_bits: int = 40,
    noise_budget: int = 200,
) -> Dict[str, Any]:
    """
    Estimate computational cost of inference.

    Args:
        model: EncryptedModel instance
        scale_bits: CKKS scale parameter (log2)
        noise_budget: Total noise budget in bits

    Returns:
        Dictionary with cost estimates

    Example:
        >>> cost = estimate_inference_cost(model)
        >>> print(f"Total noise cost: {cost['total_noise_bits']} bits")
        >>> print(f"Feasible: {cost['feasible']}")
    """
    activation_degrees = {
        'relu': 5, 'sigmoid': 5, 'tanh': 5, 'softplus': 5, 'none': 0
    }

    layer_costs = []
    total_cost = 0

    for i, layer in enumerate(model.layers):
        # Linear layer cost
        linear_cost = layer.weights.shape[0] * scale_bits

        # Activation cost
        activation_degree = activation_degrees.get(layer.activation, 0)
        activation_cost = activation_degree * scale_bits

        layer_total = linear_cost + activation_cost
        layer_costs.append(layer_total)
        total_cost += layer_total

    feasible = total_cost <= noise_budget

    return {
        'layer_costs': layer_costs,
        'total_noise_bits': total_cost,
        'noise_budget': noise_budget,
        'feasible': feasible,
        'layers_exceeding_budget': sum(1 for cost in layer_costs if cost > noise_budget),
    }
