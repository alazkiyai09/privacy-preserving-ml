"""
Server-Side Protocol Logic for HT2ML
=====================================

Implements server-side orchestration for HT2ML inference.
Coordinates HE and TEE computation with secure handoffs.
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

from src.encryption.hybrid.legacy.he.encryption import HEOperationEngine
from src.encryption.hybrid.legacy.tee.operations import TEEOperationEngine, TEEHandoffManager
from src.encryption.hybrid.legacy.tee.enclave import TEEEnclave
from src.encryption.hybrid.legacy.protocol.message import MessageBuilder, MessageType
from src.encryption.hybrid.legacy.protocol.handoff import HandoffProtocol, HandoffSession, HandoffResult
from src.encryption.hybrid.legacy.model.phishing_classifier import HybridPhishingClassifier


class ServerProtocolError(Exception):
    """Server protocol error."""
    pass


@dataclass
class ServerSession:
    """
    Server session state.

    Tracks server-side state for inference request.
    """
    session_id: str
    handoff_session: HandoffSession
    client_public_key: Optional[str] = None
    encrypted_input: Optional[Any] = None
    current_data: Optional[Any] = None  # Current data (encrypted or plaintext)
    result_class: Optional[int] = None  # Final classification result

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            'session_id': self.session_id,
            'has_public_key': self.client_public_key is not None,
            'has_input': self.encrypted_input is not None,
            'has_current_data': self.current_data is not None,
            'has_result': self.result_class is not None,
            'handoff_status': self.handoff_session.get_status(),
        }


class HT2MLServer:
    """
    HT2ML server implementation.

    Orchestrates hybrid HE/TEE inference:
    - Receives encrypted input
    - Coordinates HE and TEE operations
    - Manages handoffs between domains
    - Returns encrypted result
    """

    def __init__(
        self,
        model: HybridPhishingClassifier,
        he_config=None,
        tee_measurement: Optional[bytes] = None
    ):
        """
        Initialize HT2ML server.

        Args:
            model: Hybrid phishing classifier
            he_config: HE configuration (uses default if None)
            tee_measurement: TEE measurement for attestation
        """
        # Import configurations
        if he_config is None:
            from config.he_config import create_default_config
            he_config = create_default_config()

        # Initialize components
        self.model = model
        self.he_engine = HEOperationEngine(he_config)
        self.tee_engine = TEEOperationEngine()
        self.tee_handoff = TEEHandoffManager()

        # Initialize TEE enclave
        self.tee_enclave = TEEEnclave()
        self.tee_enclave.initialize()

        # Register TEE measurement
        self.tee_measurement = tee_measurement or self.tee_enclave.get_measurement()

        # Initialize handoff protocol
        from src.encryption.hybrid.legacy.tee.attestation import AttestationService, AttestationPolicy
        attestation_service = AttestationService(AttestationPolicy())
        attestation_service.register_expected_measurement(
            self.tee_enclave.enclave_id,
            self.tee_measurement
        )

        self.handoff_protocol = HandoffProtocol(attestation_service)

        # Load model into TEE (before attestation for initialization)
        model_weights = model.save_weights()
        self.tee_enclave.load_model(model_weights, require_attestation=False)

        # Connect engines to model
        model.set_he_engine(self.he_engine)
        model.set_tee_engine(self.tee_engine)

        # Track sessions
        self.sessions: Dict[str, ServerSession] = {}

        print("HT2ML Server initialized")
        print(f"TEE Enclave: {self.tee_enclave.enclave_id}")
        print(f"TEE Measurement: {self.tee_measurement.hex()[:40]}...")

    def process_inference_request(
        self,
        session_id: str,
        encrypted_input: Any,
        client_public_key: str
    ) -> int:
        """
        Process inference request from client.

        Args:
            session_id: Session identifier
            encrypted_input: CKKS-encrypted input features
            client_public_key: Client's public key

        Returns:
            Predicted class (0 or 1)

        Raises:
            ServerProtocolError: If inference fails
        """
        try:
            # Create handoff session
            handoff_session = self.handoff_protocol.create_session(session_id)

            # Create server session
            server_session = ServerSession(
                session_id=session_id,
                handoff_session=handoff_session,
                client_public_key=client_public_key,
                encrypted_input=encrypted_input
            )

            self.sessions[session_id] = server_session

            # Perform attestation
            self._perform_attestation(server_session)

            # Run hybrid inference
            result = self._run_hybrid_inference(server_session)

            # Complete sessions
            self.handoff_protocol.complete_session(handoff_session)

            return result

        except Exception as e:
            raise ServerProtocolError(f"Inference processing failed: {e}")

    def _perform_attestation(self, session: ServerSession) -> None:
        """
        Perform TEE attestation.

        Args:
            session: Server session

        Raises:
            ServerProtocolError: If attestation fails
        """
        print(f"Performing attestation for session {session.session_id}")

        # Request attestation
        attestation_request = self.handoff_protocol.request_attestation(
            session.handoff_session,
            self.tee_enclave.enclave_id
        )

        # Generate attestation report from TEE
        attestation_report = self.tee_enclave.generate_attestation(
            session.handoff_session.current_nonce
        )

        # Create attestation response message
        attestation_response = MessageBuilder.create_attestation_response(
            session_id=session.session_id,
            sender=self.tee_enclave.enclave_id,
            recipient="server",
            attestation_report=attestation_report.to_dict(),
            measurement=self.tee_measurement,
            nonce=session.handoff_session.current_nonce
        )

        # Verify attestation
        verified = self.handoff_protocol.verify_attestation(
            session.handoff_session,
            attestation_response,
            self.tee_measurement
        )

        if not verified:
            raise ServerProtocolError("Attestation verification failed")

        print("Attestation verified successfully")

    def _run_hybrid_inference(self, session: ServerSession) -> int:
        """
        Run hybrid HE/TEE inference.

        Args:
            session: Server session

        Returns:
            Predicted class

        Raises:
            ServerProtocolError: If inference fails
        """
        print(f"Running hybrid inference for session {session.session_id}")

        import time
        start_time = time.time()

        # Execute hybrid prediction
        # Note: This would use the full handoff mechanism in production
        # For now, we simulate the workflow

        # Get model config to determine workflow
        config = self.model.config
        layers = config.layers

        current_data = session.encrypted_input
        handoff_count = 0

        # Execute each layer
        for i, layer_spec in enumerate(layers):
            from config.model_config import LayerType, ExecutionDomain

            if layer_spec.domain == ExecutionDomain.HE:
                # Execute HE linear layer
                weights = self.model.get_layer_weights(layer_spec.name)
                bias = self.model.get_layer_bias(layer_spec.name)

                current_data = self.he_engine.execute_linear_layer(
                    current_data,
                    weights,
                    bias
                )

                print(f"HE layer {layer_spec.name} executed: ({layer_spec.input_size} → {layer_spec.output_size})")

            elif layer_spec.domain == ExecutionDomain.TEE:
                # Handoff HE→TEE if previous was HE
                if i > 0 and layers[i-1].domain == ExecutionDomain.HE:
                    # HE→TEE handoff
                    handoff_result = self.handoff_protocol.perform_handoff_he_to_tee(
                        session.handoff_session,
                        current_data,
                        self.tee_enclave.enclave_id
                    )

                    if not handoff_result.success:
                        raise ServerProtocolError(
                            f"HE→TEE handoff failed: {handoff_result.error_message}"
                        )

                    current_data = handoff_result.output_data
                    handoff_count += 1

                    print(f"HE→TEE handoff completed ({handoff_result.execution_time_ms:.2f}ms)")

                # Execute TEE operation
                if layer_spec.layer_type == LayerType.RELU:
                    tee_result = self.tee_engine.execute_relu(current_data)
                    current_data = tee_result.output
                    print(f"TEE ReLU executed ({tee_result.execution_time_ms:.2f}ms)")

                elif layer_spec.layer_type == LayerType.SOFTMAX:
                    tee_result = self.tee_engine.execute_softmax(current_data)
                    current_data = tee_result.output
                    print(f"TEE Softmax executed ({tee_result.execution_time_ms:.2f}ms)")

                elif layer_spec.layer_type == LayerType.ARGMAX:
                    tee_result = self.tee_engine.execute_argmax(current_data)
                    current_data = tee_result.output[0]  # Extract scalar
                    print(f"TEE Argmax executed ({tee_result.execution_time_ms:.2f}ms)")

                    # This is the final result
                    session.result_class = int(current_data)

                # Handoff TEE→HE if next is HE
                if i < len(layers) - 1:
                    next_layer = layers[i + 1]
                    if next_layer.domain == ExecutionDomain.HE:
                        # TEE→HE handoff
                        handoff_result = self.handoff_protocol.perform_handoff_tee_to_he(
                            session.handoff_session,
                            current_data,
                            session.client_public_key
                        )

                        if not handoff_result.success:
                            raise ServerProtocolError(
                                f"TEE→HE handoff failed: {handoff_result.error_message}"
                            )

                        current_data = handoff_result.output_data
                        handoff_count += 1

                        print(f"TEE→HE handoff completed ({handoff_result.execution_time_ms:.2f}ms)")

        total_time_ms = (time.time() - start_time) * 1000

        print(f"\nHybrid inference complete:")
        print(f"  Predicted class: {session.result_class}")
        print(f"  Total handoffs: {handoff_count}")
        print(f"  Total time: {total_time_ms:.2f}ms")

        return session.result_class

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session status.

        Args:
            session_id: Session identifier

        Returns:
            Session status or None if not found
        """
        session = self.sessions.get(session_id)

        if session is None:
            return None

        return session.get_status()

    def get_noise_status(self) -> Dict[str, Any]:
        """
        Get HE noise budget status.

        Returns:
            Noise status dictionary
        """
        return self.he_engine.get_noise_status()

    def print_noise_status(self) -> None:
        """Print noise budget status."""
        self.he_engine.noise_tracker.print_status()

    def get_tee_status(self) -> Dict[str, Any]:
        """
        Get TEE enclave status.

        Returns:
            TEE status dictionary
        """
        return self.tee_enclave.get_status()

    def get_tee_operation_stats(self) -> Dict[str, Any]:
        """
        Get TEE operation statistics.

        Returns:
            TEE operation stats
        """
        return self.tee_engine.get_stats()


def create_ht2ml_server(
    model: HybridPhishingClassifier,
    he_config=None,
    tee_measurement: Optional[bytes] = None
) -> HT2MLServer:
    """
    Factory function to create HT2ML server.

    Args:
        model: Hybrid phishing classifier
        he_config: HE configuration (uses default if None)
        tee_measurement: TEE measurement

    Returns:
        HT2MLServer instance
    """
    return HT2MLServer(model, he_config, tee_measurement)
