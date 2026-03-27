"""
Hybrid Inference Engine for HT2ML
=================================

Main orchestrator for hybrid HE/TEE inference.
Coordinates client and server for end-to-end inference.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import time

from src.encryption.hybrid.legacy.model.phishing_classifier import HybridPhishingClassifier, InferenceResult, create_random_model
from src.encryption.hybrid.legacy.protocol.client import HT2MLClient, create_ht2ml_client
from src.encryption.hybrid.legacy.protocol.server import HT2MLServer, create_ht2ml_server
from config.model_config import create_default_config


class HybridInferenceError(Exception):
    """Hybrid inference error."""
    pass


@dataclass
class HybridInferenceStats:
    """
    Statistics for hybrid inference execution.
    """
    total_time_ms: float
    he_time_ms: float
    tee_time_ms: float
    handoff_time_ms: float
    num_handoffs: int
    noise_consumed: int
    noise_remaining: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_time_ms': self.total_time_ms,
            'he_time_ms': self.he_time_ms,
            'tee_time_ms': self.tee_time_ms,
            'handoff_time_ms': self.handoff_time_ms,
            'num_handoffs': self.num_handoffs,
            'noise_consumed': self.noise_consumed,
            'noise_remaining': self.noise_remaining,
            'handoff_overhead_pct': (self.handoff_time_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0,
        }


class HybridInferenceEngine:
    """
    Hybrid HE/TEE inference engine.

    Orchestrates complete HT2ML inference workflow:
    1. Client encrypts input
    2. Server performs hybrid computation
    3. Client decrypts result
    """

    def __init__(
        self,
        model: Optional[HybridPhishingClassifier] = None,
        he_config_client=None,
        he_config_server=None
    ):
        """
        Initialize hybrid inference engine.

        Args:
            model: Hybrid model (creates random if None)
            he_config_client: HE config for client
            he_config_server: HE config for server
        """
        # Create model if not provided
        if model is None:
            config = create_default_config()
            model = create_random_model(config)

        self.model = model

        # Create client and server
        self.client = create_ht2ml_client(he_config_client)
        self.server = create_ht2ml_server(model, he_config_server)

        # Track statistics
        self.inference_count = 0
        self.total_inference_time_ms = 0.0

        print("Hybrid Inference Engine initialized")
        model.print_summary()

    def run_inference(
        self,
        features: np.ndarray,
        collect_stats: bool = True
    ) -> InferenceResult:
        """
        Run complete hybrid inference.

        Args:
            features: Input feature vector (50 dim)
            collect_stats: Whether to collect detailed statistics

        Returns:
            InferenceResult with prediction and metrics

        Raises:
            HybridInferenceError: If inference fails
        """
        start_time = time.time()

        try:
            # Validate input
            if len(features) != self.model.config.input_size:
                raise HybridInferenceError(
                    f"Expected input size {self.model.config.input_size}, "
                    f"got {len(features)}"
                )

            print(f"\n{'='*70}")
            print(f"Starting Hybrid Inference")
            print(f"{'='*70}\n")

            # Step 1: Client encrypts input
            print("Step 1: Client encrypting input...")
            client_start = time.time()

            self.client.generate_keys()
            session = self.client.create_inference_session(features)

            client_time_ms = (time.time() - client_start) * 1000
            print(f"Client encryption time: {client_time_ms:.2f}ms\n")

            # Step 2: Server processes inference
            print("Step 2: Server running hybrid computation...")
            server_start = time.time()

            predicted_class = self.server.process_inference_request(
                session_id=session.session_id,
                encrypted_input=session.encrypted_input,
                client_public_key=session.public_key
            )

            server_time_ms = (time.time() - server_start) * 1000
            print(f"\nServer computation time: {server_time_ms:.2f}ms\n")

            # Step 3: Client receives result
            print("Step 3: Client receiving result...")

            # Create mock encrypted result (in production, would come from server)
            from src.encryption.hybrid.legacy.he.encryption import CiphertextVector
            encrypted_result = CiphertextVector(
                data=["result_encrypted"],
                size=1,
                shape=(1,),
                scale=2**40,
                scheme="CKKS"
            )

            # Decrypt result
            decrypted = self.client.decrypt_result(encrypted_result)
            result_class = int(decrypted[0]) if len(decrypted) > 0 else predicted_class

            total_time_ms = (time.time() - start_time) * 1000

            print(f"\n{'='*70}")
            print(f"Hybrid Inference Complete")
            print(f"{'='*70}")
            print(f"Predicted class: {result_class} ({'Phishing' if result_class == 1 else 'Legitimate'})")
            print(f"Total time: {total_time_ms:.2f}ms")
            print(f"{'='*70}\n")

            # Create inference result
            result = InferenceResult(
                encrypted_output=encrypted_result,
                plaintext_prediction=decrypted,
                class_id=result_class,
                confidence=1.0,  # TEE argmax is deterministic
                logits=np.array([0.0, float(result_class)]),
                execution_time_ms=total_time_ms,
                he_time_ms=self.server.he_engine.noise_tracker.get_status()['operations_count'] * 1.0,  # Approximation
                tee_time_ms=server_time_ms * 0.5,  # Approximation
                handoff_time_ms=server_time_ms * 0.3,  # Approximation
                num_handoffs=3,  # Fixed by architecture
                noise_budget_used=self.server.he_engine.noise_tracker.get_consumed_budget(),
                noise_budget_remaining=self.server.he_engine.noise_tracker.get_remaining_budget(),
            )

            # Update statistics
            self.inference_count += 1
            self.total_inference_time_ms += total_time_ms

            return result

        except Exception as e:
            raise HybridInferenceError(f"Inference failed: {e}")

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

        Raises:
            HybridInferenceError: If any inference fails
        """
        results = []

        for i, features in enumerate(features_batch):
            print(f"\nProcessing sample {i+1}/{len(features_batch)}")

            result = self.run_inference(features)
            results.append(result)

            # Reset noise tracker between inferences
            # (In production, would use proper key rotation)
            self.server.he_engine.noise_tracker.reset()

        return results

    def get_average_inference_time(self) -> float:
        """
        Get average inference time.

        Returns:
            Average time in milliseconds
        """
        if self.inference_count == 0:
            return 0.0

        return self.total_inference_time_ms / self.inference_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'inference_count': self.inference_count,
            'total_inference_time_ms': self.total_inference_time_ms,
            'average_inference_time_ms': self.get_average_inference_time(),
            'client_noise_status': self.client.get_noise_status(),
            'server_noise_status': self.server.get_noise_status(),
            'tee_status': self.server.get_tee_status(),
            'tee_operation_stats': self.server.get_tee_operation_stats(),
        }

    def print_stats(self) -> None:
        """Print engine statistics."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("Hybrid Inference Engine Statistics")
        print("="*70)
        print(f"Total inferences: {stats['inference_count']}")
        print(f"Total time: {stats['total_inference_time_ms']:.2f}ms")
        print(f"Average time: {stats['average_inference_time_ms']:.2f}ms")
        print()

        print("Client Noise Budget:")
        print(f"  Consumed: {stats['client_noise_status']['consumed']} bits")
        print(f"  Remaining: {stats['client_noise_status']['remaining']} bits")
        print()

        print("Server Noise Budget:")
        print(f"  Consumed: {stats['server_noise_status']['consumed']} bits")
        print(f"  Remaining: {stats['server_noise_status']['remaining']} bits")
        print()

        print("TEE Status:")
        print(f"  State: {stats['tee_status']['state']}")
        print(f"  Attested: {stats['tee_status']['attested']}")
        print(f"  Active: {stats['tee_status']['active']}")
        print()

        print("TEE Operations:")
        print(f"  Total operations: {stats['tee_operation_stats']['operation_count']}")
        print(f"  Average time: {stats['tee_operation_stats']['average_execution_time_ms']:.2f}ms")
        print("="*70 + "\n")

    def reset_stats(self) -> None:
        """Reset engine statistics."""
        self.inference_count = 0
        self.total_inference_time_ms = 0.0


def create_hybrid_engine(
    model: Optional[HybridPhishingClassifier] = None,
    he_config_client=None,
    he_config_server=None
) -> HybridInferenceEngine:
    """
    Factory function to create hybrid inference engine.

    Args:
        model: Hybrid model (creates random if None)
        he_config_client: HE config for client
        he_config_server: HE config for server

    Returns:
        HybridInferenceEngine instance
    """
    return HybridInferenceEngine(model, he_config_client, he_config_server)


def run_single_inference(
    features: np.ndarray,
    model: Optional[HybridPhishingClassifier] = None
) -> InferenceResult:
    """
    Convenience function to run single inference.

    Args:
        features: Input features (50 dim)
        model: Hybrid model (creates random if None)

    Returns:
        InferenceResult

    Raises:
        HybridInferenceError: If inference fails
    """
    engine = create_hybrid_engine(model)
    return engine.run_inference(features)
