"""
HE↔TEE Handoff Protocol for HT2ML
==================================

Implements core handoff logic between HE and TEE domains.
Handles secure data transfer with attestation and nonces.
"""

from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets
import time

from src.encryption.hybrid.legacy.protocol.message import (
    ProtocolMessage,
    MessageBuilder,
    MessageType,
    ErrorCode,
    AttestationPayload,
    HandoffPayload,
)
from src.encryption.hybrid.legacy.tee.attestation import AttestationService, AttestationPolicy


class HandoffError(Exception):
    """Handoff protocol error."""
    pass


class HandoffDirection(Enum):
    """Direction of handoff."""
    HE_TO_TEE = "he_to_tee"
    TEE_TO_HE = "tee_to_he"


class HandoffState(Enum):
    """Handoff protocol states."""
    IDLE = "idle"
    ATTESTATION_PENDING = "attestation_pending"
    ATTESTED = "attested"
    HANDOFF_IN_PROGRESS = "handoff_in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class HandoffResult:
    """
    Result of handoff operation.

    Contains output data and metrics.
    """
    success: bool
    output_data: Optional[Any] = None
    direction: Optional[HandoffDirection] = None
    execution_time_ms: float = 0.0
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'direction': self.direction.value if self.direction else None,
            'execution_time_ms': self.execution_time_ms,
            'error_code': self.error_code.value if self.error_code else None,
            'error_message': self.error_message,
        }


@dataclass
class HandoffSession:
    """
    Handoff session state.

    Tracks state across multiple handoffs in a single inference.
    """
    session_id: str
    state: HandoffState = HandoffState.IDLE
    current_nonce: Optional[bytes] = None
    attestation_valid: bool = False
    handoff_count: int = 0
    total_handoff_time_ms: float = 0.0
    error_history: list = field(default_factory=list)

    def add_error(self, error_code: ErrorCode, message: str) -> None:
        """Add error to history."""
        self.error_history.append({
            'error_code': error_code.value,
            'message': message,
            'timestamp': time.time(),
        })

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            'session_id': self.session_id,
            'state': self.state.value,
            'attestation_valid': self.attestation_valid,
            'handoff_count': self.handoff_count,
            'total_handoff_time_ms': self.total_handoff_time_ms,
            'errors': len(self.error_history),
        }


class HandoffProtocol:
    """
    Implements HE↔TEE handoff protocol.

    Manages secure data transfer between HE and TEE domains
    with attestation verification and freshness guarantees.
    """

    def __init__(
        self,
        attestation_service: Optional[AttestationService] = None,
        policy: Optional[AttestationPolicy] = None
    ):
        """
        Initialize handoff protocol.

        Args:
            attestation_service: Attestation service (creates default if None)
            policy: Attestation policy
        """
        self.attestation_service = attestation_service or AttestationService(policy)
        self.sessions: Dict[str, HandoffSession] = {}

    def create_session(self, session_id: Optional[str] = None) -> HandoffSession:
        """
        Create new handoff session.

        Args:
            session_id: Optional session ID (generates if None)

        Returns:
            HandoffSession instance
        """
        if session_id is None:
            session_id = MessageBuilder.create_session_id()

        session = HandoffSession(session_id=session_id)
        self.sessions[session_id] = session

        return session

    def request_attestation(
        self,
        session: HandoffSession,
        enclave_id: str
    ) -> ProtocolMessage:
        """
        Request attestation from TEE.

        Args:
            session: Handoff session
            enclave_id: TEE enclave identifier

        Returns:
            Attestation request message

        Raises:
            HandoffError: If session state invalid
        """
        if session.state != HandoffState.IDLE:
            raise HandoffError(
                f"Cannot request attestation in state: {session.state.value}"
            )

        # Generate nonce for freshness
        nonce = secrets.token_bytes(32)
        session.current_nonce = nonce
        session.state = HandoffState.ATTESTATION_PENDING

        # Create attestation request
        message = MessageBuilder.create_attestation_request(
            session_id=session.session_id,
            sender="client",
            recipient=enclave_id
        )

        return message

    def verify_attestation(
        self,
        session: HandoffSession,
        attestation_response: ProtocolMessage,
        expected_measurement: bytes
    ) -> bool:
        """
        Verify attestation response from TEE.

        Args:
            session: Handoff session
            attestation_response: Attestation response message
            expected_measurement: Expected TEE measurement

        Returns:
            True if attestation valid

        Raises:
            HandoffError: If verification fails
        """
        if session.state != HandoffState.ATTESTATION_PENDING:
            raise HandoffError(
                f"Not expecting attestation in state: {session.state.value}"
            )

        if not isinstance(attestation_response.payload, AttestationPayload):
            raise HandoffError("Invalid attestation payload")

        payload = attestation_response.payload

        # Verify nonce matches
        if payload.nonce != session.current_nonce:
            session.add_error(ErrorCode.INVALID_NONCE, "Nonce mismatch")
            session.state = HandoffState.FAILED
            raise HandoffError("Attestation nonce mismatch")

        # Verify attestation report
        report_dict = payload.attestation_report

        status = self.attestation_service.verify_attestation(
            report=report_dict,
            enclave_id=attestation_response.header.sender,
            expected_measurement=expected_measurement
        )

        if status.value != "valid":
            session.add_error(ErrorCode.ATTESTATION_FAILED, "Attestation verification failed")
            session.state = HandoffState.FAILED
            raise HandoffError(f"Attestation verification failed: {status.value}")

        # Attestation valid
        session.attestation_valid = True
        session.state = HandoffState.ATTESTED

        print(f"Attestation verified for session {session.session_id}")
        return True

    def perform_handoff_he_to_tee(
        self,
        session: HandoffSession,
        encrypted_data: Any,
        tee_enclave_id: str
    ) -> HandoffResult:
        """
        Perform HE → TEE handoff.

        Args:
            session: Handoff session
            encrypted_data: CKKS-encrypted data
            tee_enclave_id: Target TEE enclave

        Returns:
            HandoffResult with plaintext output

        Raises:
            HandoffError: If handoff fails
        """
        if not session.attestation_valid:
            raise HandoffError("Session not attested")

        start_time = time.time()

        try:
            # Generate freshness nonce
            nonce = secrets.token_bytes(32)

            # Create handoff message
            message = MessageBuilder.create_handoff_message(
                session_id=session.session_id,
                sender="he_server",
                recipient=tee_enclave_id,
                encrypted_data=encrypted_data,
                nonce=nonce,
                operation="decrypt_in_tee"
            )

            # In production, would send message to TEE and receive response
            # For simulation, simulate decryption
            plaintext = self._simulate_tee_decrypt(encrypted_data)

            handoff_time_ms = (time.time() - start_time) * 1000

            session.handoff_count += 1
            session.total_handoff_time_ms += handoff_time_ms

            return HandoffResult(
                success=True,
                output_data=plaintext,
                direction=HandoffDirection.HE_TO_TEE,
                execution_time_ms=handoff_time_ms
            )

        except Exception as e:
            session.add_error(ErrorCode.TEE_ERROR, str(e))
            return HandoffResult(
                success=False,
                direction=HandoffDirection.HE_TO_TEE,
                error_code=ErrorCode.TEE_ERROR,
                error_message=str(e)
            )

    def perform_handoff_tee_to_he(
        self,
        session: HandoffSession,
        plaintext_data: Any,
        public_key: Any
    ) -> HandoffResult:
        """
        Perform TEE → HE handoff (re-encryption).

        Args:
            session: Handoff session
            plaintext_data: Plaintext data from TEE
            public_key: HE public key for re-encryption

        Returns:
            HandoffResult with encrypted output

        Raises:
            HandoffError: If handoff fails
        """
        if not session.attestation_valid:
            raise HandoffError("Session not attested")

        start_time = time.time()

        try:
            # Generate freshness nonce
            nonce = secrets.token_bytes(32)

            # In production, would re-encrypt with CKKS here
            # For simulation, create placeholder encrypted data
            encrypted = self._simulate_he_encrypt(plaintext_data)

            handoff_time_ms = (time.time() - start_time) * 1000

            session.handoff_count += 1
            session.total_handoff_time_ms += handoff_time_ms

            return HandoffResult(
                success=True,
                output_data=encrypted,
                direction=HandoffDirection.TEE_TO_HE,
                execution_time_ms=handoff_time_ms
            )

        except Exception as e:
            session.add_error(ErrorCode.HE_ERROR, str(e))
            return HandoffResult(
                success=False,
                direction=HandoffDirection.TEE_TO_HE,
                error_code=ErrorCode.HE_ERROR,
                error_message=str(e)
            )

    def complete_session(self, session: HandoffSession) -> None:
        """
        Complete handoff session.

        Args:
            session: Handoff session to complete
        """
        session.state = HandoffState.COMPLETE
        print(
            f"Session {session.session_id} complete. "
            f"Total handoffs: {session.handoff_count}, "
            f"Total time: {session.total_handoff_time_ms:.2f}ms"
        )

    def _simulate_tee_decrypt(self, encrypted_data: Any) -> Any:
        """
        Simulate TEE decryption.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Plaintext data

        Note:
            In production, would decrypt in TEE using secret key.
        """
        import numpy as np

        # Extract size from encrypted data
        if hasattr(encrypted_data, 'size'):
            size = encrypted_data.size
        else:
            size = 50  # Default input size

        # Return zeros as placeholder
        return np.zeros(size)

    def _simulate_he_encrypt(self, plaintext_data: Any) -> Any:
        """
        Simulate HE re-encryption.

        Args:
            plaintext_data: Plaintext data

        Returns:
            Encrypted data

        Note:
            In production, would encrypt with CKKS.
        """
        from src.encryption.hybrid.legacy.he.encryption import CiphertextVector
        from config.he_config import CKKSParams
        import numpy as np

        if isinstance(plaintext_data, np.ndarray):
            size = len(plaintext_data)
            shape = plaintext_data.shape
        else:
            size = 64  # Default hidden size
            shape = (64,)

        encrypted = CiphertextVector(
            data=[f"reencrypted_{i}" for i in range(size)],
            size=size,
            shape=shape,
            scale=2**40,
            scheme="CKKS"
        )
        encrypted.params = CKKSParams()

        return encrypted

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session status.

        Args:
            session_id: Session identifier

        Returns:
            Session status dictionary or None if not found
        """
        session = self.sessions.get(session_id)

        if session is None:
            return None

        return session.get_status()


def create_handoff_protocol(
    attestation_service: Optional[AttestationService] = None,
    policy: Optional[AttestationPolicy] = None
) -> HandoffProtocol:
    """
    Factory function to create handoff protocol.

    Args:
        attestation_service: Attestation service (creates default if None)
        policy: Attestation policy

    Returns:
        HandoffProtocol instance
    """
    return HandoffProtocol(attestation_service, policy)
