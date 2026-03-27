"""
Phishing Classifier Model for HT2ML System
==========================================

Main model definition for hybrid HE/TEE phishing detection.
Implements the network architecture with split computation.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pickle

from config.model_config import ModelConfig, LayerSpec, LayerType, ExecutionDomain
from src.encryption.hybrid.legacy.model.layers import BaseLayer, LayerFactory, validate_layer_connections


class PredictionResult(Enum):
    """Prediction result types."""
    PHISHING = 1  # Malicious
    LEGITIMATE = 0  # Safe


@dataclass
class InferenceResult:
    """
    Result of HT2ML inference.

    Contains both intermediate and final results.
    """
    encrypted_output: Optional[Any] = None  # Encrypted final result
    plaintext_prediction: Optional[np.ndarray] = None  # Decrypted prediction
    class_id: int = -1  # Predicted class (0 or 1)
    confidence: float = 0.0  # Confidence score
    logits: Optional[np.ndarray] = None  # Raw logits

    # Performance metrics
    execution_time_ms: float = 0.0
    he_time_ms: float = 0.0
    tee_time_ms: float = 0.0
    handoff_time_ms: float = 0.0
    num_handoffs: int = 0

    # Noise tracking (for HE layers)
    noise_budget_used: int = 0
    noise_budget_remaining: int = 0

    def get_class_name(self) -> str:
        """Get human-readable class name."""
        if self.class_id == PredictionResult.PHISHING.value:
            return "phishing"
        elif self.class_id == PredictionResult.LEGITIMATE.value:
            return "legitimate"
        else:
            return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'class': self.get_class_name(),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'logits': self.logits.tolist() if self.logits is not None else None,
            'execution_time_ms': self.execution_time_ms,
            'he_time_ms': self.he_time_ms,
            'tee_time_ms': self.tee_time_ms,
            'handoff_time_ms': self.handoff_time_ms,
            'num_handoffs': self.num_handoffs,
        }


class PhishingClassifier:
    """
    Phishing classifier with hybrid HE/TEE architecture.

    Network Architecture:
        Input (50 features)
            ↓ [HE] Linear (50→64)
            ↓ [TEE] ReLU
            ↓ [HE] Linear (64→2)
            ↓ [TEE] Softmax + Argmax
            ↓ Output

    The model coordinates HE and TEE computations with secure handoffs.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize phishing classifier.

        Args:
            config: Model configuration
        """
        self.config = config
        config.validate()

        # Initialize layers (empty - weights loaded separately)
        self.layers: List[BaseLayer] = []
        self.is_trained = False

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Load model weights from dictionary.

        Args:
            weights: Dictionary with keys 'linear1.weight', 'linear1.bias', etc.
        """
        for layer_spec in self.config.layers:
            if layer_spec.layer_type == LayerType.LINEAR:
                weight_key = f"{layer_spec.name}.weight"
                bias_key = f"{layer_spec.name}.bias"

                if weight_key not in weights or bias_key not in weights:
                    raise ValueError(
                        f"Missing weights for {layer_spec.name}: "
                        f"need {weight_key} and {bias_key}"
                    )

                layer = LayerFactory.create_layer(
                    layer_spec,
                    weights[weight_key],
                    weights[bias_key]
                )
                self.layers.append(layer)

        self.is_trained = True

    def save_weights(self) -> Dict[str, np.ndarray]:
        """
        Save model weights to dictionary.

        Returns:
            Dictionary with weight arrays
        """
        weights = {}

        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                weights[f"{layer.name}.weight"] = layer.weights
                weights[f"{layer.name}.bias"] = layer.bias

        return weights

    def get_num_parameters(self) -> int:
        """Calculate total number of parameters."""
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total += layer.weights.size
            if hasattr(layer, 'bias') and layer.bias is not None:
                total += layer.bias.size
        return total

    def get_model_size_bytes(self) -> int:
        """Calculate model size in bytes."""
        param_size = self.get_num_parameters() * 4  # float32
        return param_size

    def print_summary(self) -> None:
        """Print model summary."""
        print("\n" + "=" * 70)
        print("HT2ML Phishing Classifier")
        print("=" * 70)
        print()

        print(f"Input Features: {self.config.input_size}")
        print(f"Hidden Size: {self.config.hidden_size}")
        print(f"Output Classes: {self.config.output_size}")
        print(f"Total Parameters: {self.get_num_parameters():,}")
        print(f"Model Size: {self.get_model_size_bytes() / 1024:.2f} KB")
        print()

        print("Layer Architecture:")
        print("-" * 70)

        for i, layer_spec in enumerate(self.config.layers):
            domain_str = layer_spec.domain.value.upper()
            print(f"  {i+1}. {layer_spec.name:10s} | {layer_spec.layer_type.value:10s} | {domain_str:6s} | "
                  f"({layer_spec.input_size:3d} → {layer_spec.output_size:3d})")

        print()
        print(f"Number of HE/TEE Handoffs: {self.config.get_num_handoffs()}")
        print(f"Trained: {self.is_trained}")
        print("=" * 70 + "\n")


class HybridPhishingClassifier(PhishingClassifier):
    """
    Extended phishing classifier with hybrid HE/TEE inference.

    This version implements the actual HT2ML inference workflow with
    multiple HE↔TEE handoffs.
    """

    def __init__(
        self,
        config: ModelConfig,
        he_engine=None,
        tee_engine=None
    ):
        """
        Initialize hybrid classifier.

        Args:
            config: Model configuration
            he_engine: HE computation engine (optional, set later)
            tee_engine: TEE computation engine (optional, set later)
        """
        super().__init__(config)
        self.he_engine = he_engine
        self.tee_engine = tee_engine

    def set_he_engine(self, he_engine) -> None:
        """Set HE computation engine."""
        self.he_engine = he_engine

    def set_tee_engine(self, tee_engine) -> None:
        """Set TEE computation engine."""
        self.tee_engine = tee_engine

    def predict_hybrid(
        self,
        encrypted_input: Any,
        client_context: Any = None
    ) -> InferenceResult:
        """
        Run HT2ML inference with multiple HE↔TEE handoffs.

        Workflow:
        1. HE: Linear layer 1 (50 → 64)
        2. Handoff HE→TEE
        3. TEE: ReLU activation
        4. Handoff TEE→HE
        5. HE: Linear layer 2 (64 → 2)
        6. Handoff HE→TEE
        7. TEE: Softmax
        8. TEE: Argmax
        9. Return result

        Args:
            encrypted_input: CKKS-encrypted input features (50 dim)
            client_context: Client context (keys, etc.)

        Returns:
            InferenceResult with prediction and metrics

        Raises:
            RuntimeError: If HE or TEE engines not set
            NoiseBudgetExceededError: If HE noise budget exhausted
            HandoffError: If handoff fails
        """
        if not self.he_engine:
            raise RuntimeError("HE engine not set. Call set_he_engine() first.")
        if not self.tee_engine:
            raise RuntimeError("TEE engine not set. Call set_tee_engine() first.")

        result = InferenceResult()

        import time
        start_time = time.time()

        # Execute layers according to architecture
        current_data = encrypted_input
        handoff_count = 0

        for i, layer_spec in enumerate(self.config.layers):
            layer_start = time.time()

            if layer_spec.domain == ExecutionDomain.HE:
                # Execute in HE
                current_data = self.he_engine.execute_linear_layer(
                    current_data,
                    self.get_layer_weights(layer_spec.name),
                    self.get_layer_bias(layer_spec.name)
                )

            elif layer_spec.domain == ExecutionDomain.TEE:
                # Handoff HE→TEE if needed
                if i > 0 and self.config.layers[i-1].domain == ExecutionDomain.HE:
                    # Perform HE→TEE handoff
                    current_data, handoff_time = self.he_engine.handoff_to_tee(
                        current_data,
                        client_context
                    )
                    result.handoff_time_ms += handoff_time
                    handoff_count += 1

                # Execute in TEE
                if layer_spec.layer_type == LayerType.RELU:
                    current_data = self.tee_engine.execute_relu(current_data)
                elif layer_spec.layer_type == LayerType.SOFTMAX:
                    current_data = self.tee_engine.execute_softmax(current_data)
                elif layer_spec.layer_type == LayerType.ARGMAX:
                    current_data = self.tee_engine.execute_argmax(current_data)

                # Handoff TEE→HE if needed (for next HE layer)
                if i < len(self.config.layers) - 1:
                    next_layer = self.config.layers[i + 1]
                    if next_layer.domain == ExecutionDomain.HE:
                        # Need to re-encrypt for HE
                        current_data, handoff_time = self.tee_engine.handoff_to_he(
                            current_data,
                            client_context
                        )
                        result.handoff_time_ms += handoff_time
                        handoff_count += 1

            layer_time_ms = (time.time() - layer_start) * 1000
            # Track domain-specific time
            if layer_spec.domain == ExecutionDomain.HE:
                result.he_time_ms += layer_time_ms
            else:
                result.tee_time_ms += layer_time_ms

        # Final result
        result.class_id = current_data
        result.confidence = 1.0  # TEE argmax gives us the class directly
        result.num_handoffs = handoff_count
        result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def get_layer_weights(self, layer_name: str) -> np.ndarray:
        """Get weights for a specific layer."""
        for layer in self.layers:
            if layer.name == layer_name:
                return layer.weights
        raise ValueError(f"Layer {layer_name} not found")

    def get_layer_bias(self, layer_name: str) -> np.ndarray:
        """Get bias for a specific layer."""
        for layer in self.layers:
            if layer.name == layer_name:
                return layer.bias
        raise ValueError(f"Layer {layer_name} not found")


def create_classifier(config: Optional[ModelConfig] = None) -> HybridPhishingClassifier:
    """
    Factory function to create phishing classifier.

    Args:
        config: Model configuration (uses default if None)

    Returns:
        HybridPhishingClassifier instance
    """
    if config is None:
        from config.model_config import create_default_config
        config = create_default_config()

    return HybridPhishingClassifier(config)


def create_random_model(config: ModelConfig) -> HybridPhishingClassifier:
    """
    Create classifier with random weights for testing.

    Args:
        config: Model configuration

    Returns:
        HybridPhishingClassifier with random weights
    """
    classifier = HybridPhishingClassifier(config)

    # Generate random weights
    weights = {}

    np.random.seed(42)  # For reproducibility

    # Linear layer 1: 50 -> 64
    weights["linear1.weight"] = np.random.randn(50, 64) * 0.1
    weights["linear1.bias"] = np.random.randn(64) * 0.01

    # Linear layer 2: 64 -> 2
    weights["linear2.weight"] = np.random.randn(64, 2) * 0.1
    weights["linear2.bias"] = np.random.randn(2) * 0.01

    classifier.load_weights(weights)

    return classifier


def save_model(model: PhishingClassifier, path: str) -> None:
    """
    Save model weights to disk.

    Args:
        model: Trained model
        path: File path to save
    """
    weights = model.save_weights()

    model_data = {
        'config': model.config,
        'weights': weights,
        'is_trained': model.is_trained,
    }

    with open(path, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(path: str) -> HybridPhishingClassifier:
    """
    Load model from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded HybridPhishingClassifier
    """
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    classifier = HybridPhishingClassifier(model_data['config'])
    classifier.load_weights(model_data['weights'])

    return classifier
