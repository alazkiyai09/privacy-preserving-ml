"""
TEE-Only Inference Engine for HT2ML
====================================

Baseline implementation using only Trusted Execution Environment.
All operations performed in plaintext within secure enclave.

This provides:
- Fast performance (native execution)
- Requires trust in TEE
- No encryption overhead
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import time

from src.encryption.hybrid.legacy.model.phishing_classifier import HybridPhishingClassifier, InferenceResult
from src.encryption.hybrid.legacy.tee.operations import TEEOperationEngine
from src.encryption.hybrid.legacy.tee.enclave import TEEEnclave
from config.model_config import LayerType, ExecutionDomain


class TEEOnlyEngineError(Exception):
    """TEE-only engine error."""
    pass


class TEEOnlyInferenceEngine:
    """
    TEE-only inference engine.

    Performs all operations in plaintext within TEE:
    - All layers execute in secure enclave
    - No encryption/decryption overhead
    - Fast execution but requires trust in TEE

    This provides maximum performance but relies on TEE security.
    """

    def __init__(
        self,
        model: HybridPhishingClassifier,
        tee_measurement: Optional[bytes] = None
    ):
        """
        Initialize TEE-only engine.

        Args:
            model: Phishing classifier
            tee_measurement: TEE measurement for attestation
        """
        self.model = model

        # Initialize TEE enclave
        self.tee_enclave = TEEEnclave()
        self.tee_enclave.initialize()

        # Register measurement
        self.tee_measurement = tee_measurement or self.tee_enclave.get_measurement()

        # Load model into TEE
        model_weights = model.save_weights()
        self.tee_enclave.load_model(model_weights, require_attestation=False)

        # Initialize TEE operation engine
        self.tee_engine = TEEOperationEngine()

        # Statistics
        self.inference_count = 0
        self.total_inference_time_ms = 0.0
        self.layer_times_ms: Dict[str, float] = {}

        print("TEE-Only Inference Engine initialized")
        print("All operations will be performed in TEE")
        print(f"TEE Enclave: {self.tee_enclave.enclave_id}")
        model.print_summary()

    def run_inference(self, features: np.ndarray) -> InferenceResult:
        """
        Run TEE-only inference.

        Args:
            features: Input features (50 dim)

        Returns:
            InferenceResult with prediction

        Raises:
            TEEOnlyEngineError: If inference fails
        """
        start_time = time.time()

        try:
            # Validate input
            if len(features) != self.model.config.input_size:
                raise TEEOnlyEngineError(
                    f"Expected input size {self.model.config.input_size}, "
                    f"got {len(features)}"
                )

            print(f"\n{'='*70}")
            print("TEE-Only Inference")
            print(f"{'='*70}\n")

            # Start attestation (one-time setup)
            if self.tee_enclave.get_state().value == "initialized":
                attestation_report = self.tee_enclave.generate_attestation()
                print(f"TEE Attestation generated: {attestation_report.enclave_measurement.hex()[:40]}...\n")

            # Execute all layers in TEE
            current_data = features.astype(np.float32)

            for i, layer_spec in enumerate(self.model.config.layers):
                layer_start = time.time()

                if layer_spec.layer_type == LayerType.LINEAR:
                    # Linear layer in TEE
                    weights = self.model.get_layer_weights(layer_spec.name)
                    bias = self.model.get_layer_bias(layer_spec.name)

                    # Matrix multiplication: output = input * weights^T + bias
                    current_data = np.dot(current_data, weights) + bias

                    layer_time_ms = (time.time() - layer_start) * 1000
                    self.layer_times_ms[layer_spec.name] = layer_time_ms

                    print(f"TEE Linear layer {layer_spec.name}: ({layer_spec.input_size} → {layer_spec.output_size}) - {layer_time_ms:.3f}ms")

                elif layer_spec.layer_type == LayerType.RELU:
                    # ReLU in TEE
                    tee_result = self.tee_engine.execute_relu(current_data)
                    current_data = tee_result.output

                    layer_time_ms = tee_result.execution_time_ms
                    self.layer_times_ms[layer_spec.name] = layer_time_ms

                    print(f"TEE ReLU: {layer_spec.input_size} elements - {layer_time_ms:.3f}ms")

                elif layer_spec.layer_type == LayerType.SOFTMAX:
                    # Softmax in TEE
                    tee_result = self.tee_engine.execute_softmax(current_data)
                    current_data = tee_result.output

                    layer_time_ms = tee_result.execution_time_ms
                    self.layer_times_ms[layer_spec.name] = layer_time_ms

                    print(f"TEE Softmax: {layer_time_ms:.3f}ms")
                    print(f"  Probabilities: [{current_data[0]:.4f}, {current_data[1]:.4f}]")

                elif layer_spec.layer_type == LayerType.ARGMAX:
                    # Argmax in TEE
                    tee_result = self.tee_engine.execute_argmax(current_data)
                    result_class = int(tee_result.output[0])

                    layer_time_ms = tee_result.execution_time_ms
                    self.layer_times_ms[layer_spec.name] = layer_time_ms

                    print(f"TEE Argmax: class {result_class} - {layer_time_ms:.3f}ms\n")

            total_time_ms = (time.time() - start_time) * 1000

            print(f"{'='*70}")
            print(f"TEE-Only Inference Complete")
            print(f"{'='*70}")
            print(f"Predicted class: {result_class} ({'Phishing' if result_class == 1 else 'Legitimate'})")
            print(f"Total time: {total_time_ms:.3f}ms")
            print(f"{'='*70}\n")

            # Create result
            result = InferenceResult(
                encrypted_output=None,  # No encryption in TEE-only
                plaintext_prediction=np.array([float(result_class)]),
                class_id=result_class,
                confidence=float(current_data[result_class]) if result_class < len(current_data) else 1.0,
                logits=current_data,
                execution_time_ms=total_time_ms,
                he_time_ms=0.0,  # No HE
                tee_time_ms=total_time_ms,  # All TEE
                handoff_time_ms=0.0,  # No handoffs
                num_handoffs=0,  # No handoffs
                noise_budget_used=0,  # No noise budget
                noise_budget_remaining=0,
            )

            # Update statistics
            self.inference_count += 1
            self.total_inference_time_ms += total_time_ms

            return result

        except Exception as e:
            raise TEEOnlyEngineError(f"TEE-only inference failed: {e}")

    def run_batch_inference(
        self,
        features_batch: np.ndarray
    ) -> list:
        """
        Run inference on batch of inputs.

        Args:
            features_batch: Batch of feature vectors [N, 50]

        Returns:
            List of InferenceResult objects
        """
        results = []

        for i, features in enumerate(features_batch):
            print(f"\nProcessing sample {i+1}/{len(features_batch)}")
            result = self.run_inference(features)
            results.append(result)

        return results

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
            'layer_times_ms': self.layer_times_ms.copy(),
            'tee_status': self.tee_enclave.get_status(),
            'tee_operation_stats': self.tee_engine.get_stats(),
        }

    def print_stats(self) -> None:
        """Print statistics."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("TEE-Only Engine Statistics")
        print("="*70)
        print(f"Total inferences: {stats['inference_count']}")
        print(f"Total time: {stats['total_time_ms']:.3f}ms")
        print(f"Average time: {stats['average_time_ms']:.3f}ms")
        print()

        print("Layer Times (average):")
        for layer_name, time_ms in stats['layer_times_ms'].items():
            print(f"  {layer_name}: {time_ms:.3f}ms")
        print()

        tee_stats = stats['tee_operation_stats']
        print("TEE Operations:")
        print(f"  Total operations: {tee_stats['operation_count']}")
        print(f"  Average time: {tee_stats['average_execution_time_ms']:.3f}ms")
        print("="*70 + "\n")

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.inference_count = 0
        self.total_inference_time_ms = 0.0
        self.layer_times_ms = {}
        self.tee_engine.reset_stats()


def create_tee_only_engine(
    model: HybridPhishingClassifier,
    tee_measurement: Optional[bytes] = None
) -> TEEOnlyInferenceEngine:
    """
    Factory function to create TEE-only engine.

    Args:
        model: Phishing classifier
        tee_measurement: TEE measurement

    Returns:
        TEEOnlyInferenceEngine instance
    """
    return TEEOnlyInferenceEngine(model, tee_measurement)


def run_tee_only_inference(
    features: np.ndarray,
    model: HybridPhishingClassifier
) -> InferenceResult:
    """
    Convenience function to run TEE-only inference.

    Args:
        features: Input features (50 dim)
        model: Phishing classifier

    Returns:
        InferenceResult

    Raises:
        TEEOnlyEngineError: If inference fails
    """
    engine = create_tee_only_engine(model)
    return engine.run_inference(features)
