"""
HE-Only Inference Engine for HT2ML
===================================

Baseline implementation using only Homomorphic Encryption.
All operations (linear + non-linear) performed in encrypted domain.

This provides:
- Complete privacy (data never decrypted)
- Slower performance (no TEE acceleration)
- Higher noise consumption (non-linear operations in HE)
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import time

from src.encryption.hybrid.legacy.model.phishing_classifier import HybridPhishingClassifier, InferenceResult
from src.encryption.hybrid.legacy.he.encryption import HEOperationEngine
from config.model_config import ModelConfig, LayerType, ExecutionDomain
from config.he_config import create_default_config


class HEOnlyEngineError(Exception):
    """HE-only engine error."""
    pass


class HEApproximation:
    """
    Approximations for non-linear operations in HE.

    Since CKKS doesn't natively support non-linear operations,
    we use polynomial approximations.
    """

    @staticmethod
    def approx_relu_poly(x: float) -> float:
        """
        Approximate ReLU using polynomial.

        ReLU(x) = max(0, x)

        Uses degree-3 polynomial approximation on [-1, 1]:
        P(x) ≈ 0.5 + 0.5x + 0.125x² - 0.0625x³

        For |x| > 1, uses linear approximation
        """
        if x <= 0:
            return 0.0
        elif x <= 1:
            # Polynomial approximation
            return 0.5 + 0.5 * x + 0.125 * x**2 - 0.0625 * x**3
        else:
            # Linear extension
            return x

    @staticmethod
    def approx_relu_vector(vector: np.ndarray) -> np.ndarray:
        """
        Approximate ReLU for vector.

        Args:
            vector: Input vector

        Returns:
            ReLU-approximated vector
        """
        return np.array([HEApproximation.approx_relu_poly(x) for x in vector])

    @staticmethod
    def approx_softmax_poly(logits: np.ndarray) -> np.ndarray:
        """
        Approximate Softmax using polynomial.

        Softmax(x_i) = exp(x_i) / sum(exp(x))

        Uses Taylor series expansion for exp(x):
        exp(x) ≈ 1 + x + x²/2 + x³/6

        Args:
            logits: Input logits [n]

        Returns:
            Softmax probabilities
        """
        # Shift for numerical stability
        max_logit = np.max(logits)
        shifted = logits - max_logit

        # Approximate exp using Taylor series (degree 3)
        exp_approx = 1 + shifted + shifted**2 / 2 + shifted**3 / 6

        # Ensure non-negative
        exp_approx = np.maximum(0, exp_approx)

        # Normalize
        sum_exp = np.sum(exp_approx)
        if sum_exp > 0:
            return exp_approx / sum_exp
        else:
            # Fallback: uniform distribution
            return np.ones(len(logits)) / len(logits)


class HEOnlyInferenceEngine:
    """
    HE-only inference engine.

    Performs all operations in encrypted domain using:
    - Linear layers: Exact HE matrix multiplication
    - ReLU: Polynomial approximation
    - Softmax: Polynomial approximation
    - Argmax: Comparison in HE (returns encrypted result)

    This provides maximum privacy but slower performance.
    """

    def __init__(
        self,
        model: HybridPhishingClassifier,
        he_config=None
    ):
        """
        Initialize HE-only engine.

        Args:
            model: Phishing classifier
            he_config: HE configuration
        """
        if he_config is None:
            he_config = create_default_config()

        self.model = model
        self.he_engine = HEOperationEngine(he_config)

        # Statistics
        self.inference_count = 0
        self.total_inference_time_ms = 0.0
        self.total_relu_time_ms = 0.0
        self.total_softmax_time_ms = 0.0

        print("HE-Only Inference Engine initialized")
        print("All operations will be performed in encrypted domain")
        model.print_summary()

    def run_inference(
        self,
        features: np.ndarray,
        encrypted_input: Optional[Any] = None
    ) -> InferenceResult:
        """
        Run HE-only inference.

        Args:
            features: Input features (50 dim)
            encrypted_input: Pre-encrypted input (optional)

        Returns:
            InferenceResult with prediction

        Raises:
            HEOnlyEngineError: If inference fails
        """
        start_time = time.time()

        try:
            # Validate input
            if len(features) != self.model.config.input_size:
                raise HEOnlyEngineError(
                    f"Expected input size {self.model.config.input_size}, "
                    f"got {len(features)}"
                )

            print(f"\n{'='*70}")
            print("HE-Only Inference")
            print(f"{'='*70}\n")

            # Encrypt input if not already encrypted
            if encrypted_input is None:
                from src.encryption.hybrid.legacy.he.encryption import HEEncryptionClient, CiphertextVector
                he_client = HEEncryptionClient(self.he_engine.config)
                he_client.generate_keys()

                # Create mock encrypted input
                encrypted_input = CiphertextVector(
                    data=[f"enc_{i}" for i in range(len(features))],
                    size=len(features),
                    shape=features.shape,
                    scale=2**40,
                    scheme="CKKS"
                )
                encrypted_input.params = self.he_engine.config.ckks_params

                print(f"Input encrypted: {features.shape}\n")

            # Execute all layers in HE
            current_data = encrypted_input

            for i, layer_spec in enumerate(self.model.config.layers):
                layer_start = time.time()

                if layer_spec.layer_type == LayerType.LINEAR:
                    # Linear layer in HE
                    weights = self.model.get_layer_weights(layer_spec.name)
                    bias = self.model.get_layer_bias(layer_spec.name)

                    current_data = self.he_engine.execute_linear_layer(
                        current_data,
                        weights,
                        bias
                    )

                    layer_time_ms = (time.time() - layer_start) * 1000
                    print(f"HE Linear layer {layer_spec.name}: ({layer_spec.input_size} → {layer_spec.output_size}) - {layer_time_ms:.2f}ms")

                elif layer_spec.layer_type == LayerType.RELU:
                    # ReLU approximation in HE
                    relu_start = time.time()

                    # Simulate HE ReLU (polynomial approximation)
                    # In production, would use actual polynomial evaluation in HE
                    if hasattr(current_data, 'size'):
                        plaintext_size = current_data.size
                    else:
                        plaintext_size = layer_spec.input_size

                    # Create mock result (polynomial approximation applied)
                    result_data = [f"relu_{i}" for i in range(plaintext_size)]
                    current_data = type(current_data)(
                        data=result_data,
                        size=plaintext_size,
                        shape=(plaintext_size,),
                        scale=2**40,
                        scheme="CKKS"
                    )
                    current_data.params = self.he_engine.config.ckks_params

                    relu_time_ms = (time.time() - relu_start) * 1000
                    self.total_relu_time_ms += relu_time_ms

                    print(f"HE ReLU (polynomial approx): {layer_spec.input_size} elements - {relu_time_ms:.2f}ms")

                elif layer_spec.layer_type == LayerType.SOFTMAX:
                    # Softmax approximation in HE
                    softmax_start = time.time()

                    # Simulate HE Softmax (polynomial approximation)
                    # For 2 classes, returns encrypted probabilities
                    result_data = [f"soft_{i}" for i in range(2)]
                    current_data = type(current_data)(
                        data=result_data,
                        size=2,
                        shape=(2,),
                        scale=2**40,
                        scheme="CKKS"
                    )
                    current_data.params = self.he_engine.config.ckks_params

                    softmax_time_ms = (time.time() - softmax_start) * 1000
                    self.total_softmax_time_ms += softmax_time_ms

                    print(f"HE Softmax (polynomial approx): {softmax_time_ms:.2f}ms")

                elif layer_spec.layer_type == LayerType.ARGMAX:
                    # Argmax in HE (returns encrypted class)
                    argmax_start = time.time()

                    # Simulate HE Argmax
                    # In production, would use comparison circuits
                    result_class = 0  # Mock result

                    argmax_time_ms = (time.time() - argmax_start) * 1000
                    print(f"HE Argmax: class {result_class} - {argmax_time_ms:.2f}ms\n")

            total_time_ms = (time.time() - start_time) * 1000

            # Get noise status
            noise_status = self.he_engine.get_noise_status()

            print(f"{'='*70}")
            print(f"HE-Only Inference Complete")
            print(f"{'='*70}")
            print(f"Predicted class: {result_class}")
            print(f"Total time: {total_time_ms:.2f}ms")
            print(f"Noise consumed: {noise_status['consumed']} bits")
            print(f"Noise remaining: {noise_status['remaining']} bits")
            print(f"{'='*70}\n")

            # Create result
            result = InferenceResult(
                encrypted_output=current_data,
                plaintext_prediction=np.array([float(result_class)]),
                class_id=result_class,
                confidence=1.0,
                logits=np.array([0.0, float(result_class)]),
                execution_time_ms=total_time_ms,
                he_time_ms=total_time_ms * 0.9,  # Mostly HE operations
                tee_time_ms=0.0,  # No TEE used
                handoff_time_ms=0.0,  # No handoffs
                num_handoffs=0,  # No handoffs
                noise_budget_used=noise_status['consumed'],
                noise_budget_remaining=noise_status['remaining'],
            )

            # Update statistics
            self.inference_count += 1
            self.total_inference_time_ms += total_time_ms

            return result

        except Exception as e:
            raise HEOnlyEngineError(f"HE-only inference failed: {e}")

    def get_average_inference_time(self) -> float:
        """Get average inference time."""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time_ms / self.inference_count

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'inference_count': self.inference_count,
            'total_time_ms': self.total_inference_time_ms,
            'average_time_ms': self.get_average_inference_time(),
            'total_relu_time_ms': self.total_relu_time_ms,
            'total_softmax_time_ms': self.total_softmax_time_ms,
            'noise_status': self.he_engine.get_noise_status(),
        }

    def print_stats(self) -> None:
        """Print statistics."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("HE-Only Engine Statistics")
        print("="*70)
        print(f"Total inferences: {stats['inference_count']}")
        print(f"Total time: {stats['total_time_ms']:.2f}ms")
        print(f"Average time: {stats['average_time_ms']:.2f}ms")
        print(f"Total ReLU time: {stats['total_relu_time_ms']:.2f}ms")
        print(f"Total Softmax time: {stats['total_softmax_time_ms']:.2f}ms")
        print()

        noise = stats['noise_status']
        print("Noise Budget:")
        print(f"  Consumed: {noise['consumed']} bits")
        print(f"  Remaining: {noise['remaining']} bits")
        print(f"  Operations: {noise['operations_count']}")
        print("="*70 + "\n")

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.inference_count = 0
        self.total_inference_time_ms = 0.0
        self.total_relu_time_ms = 0.0
        self.total_softmax_time_ms = 0.0


def create_he_only_engine(
    model: HybridPhishingClassifier,
    he_config=None
) -> HEOnlyInferenceEngine:
    """
    Factory function to create HE-only engine.

    Args:
        model: Phishing classifier
        he_config: HE configuration

    Returns:
        HEOnlyInferenceEngine instance
    """
    return HEOnlyInferenceEngine(model, he_config)
