"""
Protocol Message Formats for HT2ML
===================================

Defines message formats for HE↔TEE communication.
Implements secure message passing with nonces and attestation.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import secrets
from uuid import uuid4


class MessageType(Enum):
    """Types of protocol messages."""
    # HE → TEE messages
    HE_TO_TEE_HANDOFF = "he_to_tee_handoff"
    HE_ENCRYPTED_DATA = "he_encrypted_data"

    # TEE → HE messages
    TEE_TO_HE_HANDOFF = "tee_to_he_handoff"
    TEE_PROCESSED_DATA = "tee_processed_data"

    # Control messages
    ATTESTATION_REQUEST = "attestation_request"
    ATTESTATION_RESPONSE = "attestation_response"
    ATTESTATION_VERIFY = "attestation_verify"
    ATTESTATION_RESULT = "attestation_result"

    # Error messages
    ERROR = "error"
    ABORT = "abort"

    # Completion
    COMPLETE = "complete"


class ErrorCode(Enum):
    """Error codes for protocol failures."""
    SUCCESS = "success"
    ATTESTATION_FAILED = "attestation_failed"
    ATTESTATION_EXPIRED = "attestation_expired"
    MEASUREMENT_MISMATCH = "measurement_mismatch"
    NOISE_BUDGET_EXCEEDED = "noise_budget_exceeded"
    INVALID_NONCE = "invalid_nonce"
    INVALID_MESSAGE = "invalid_message"
    TEE_ERROR = "tee_error"
    HE_ERROR = "he_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class MessageHeader:
    """
    Protocol message header.

    Contains metadata about message.
    """
    message_id: str
    message_type: MessageType
    session_id: str
    timestamp: float
    sender: str  # "client", "he_server", "tee_enclave"
    recipient: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'sender': self.sender,
            'recipient': self.recipient,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MessageHeader':
        """Create from dictionary."""
        return MessageHeader(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            session_id=data['session_id'],
            timestamp=data['timestamp'],
            sender=data['sender'],
            recipient=data['recipient'],
        )


@dataclass
class AttestationPayload:
    """
    Attestation message payload.

    Contains attestation report and verification data.
    """
    nonce: bytes
    attestation_report: Dict[str, Any]  # TEE attestation report
    measurement: bytes  # Enclave measurement

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nonce': self.nonce.hex(),
            'attestation_report': self.attestation_report,
            'measurement': self.measurement.hex(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AttestationPayload':
        """Create from dictionary."""
        return AttestationPayload(
            nonce=bytes.fromhex(data['nonce']),
            attestation_report=data['attestation_report'],
            measurement=bytes.fromhex(data['measurement']),
        )


@dataclass
class HandoffPayload:
    """
    HE↔TEE handoff payload.

    Contains data for handoff between domains.
    """
    encrypted_data: Optional[Any] = None  # Encrypted data from HE
    plaintext_data: Optional[List[float]] = None  # Plaintext from TEE
    nonce: Optional[bytes] = None
    attestation: Optional[Dict[str, Any]] = None

    # Metadata
    data_shape: Optional[tuple] = None
    data_size: int = 0
    operation: str = ""  # Last operation performed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'nonce': self.nonce.hex() if self.nonce else None,
            'attestation': self.attestation,
            'data_shape': self.data_shape,
            'data_size': self.data_size,
            'operation': self.operation,
        }

        # Handle encrypted data (placeholder)
        if self.encrypted_data is not None:
            # In production, would serialize ciphertext
            result['has_encrypted_data'] = True
        else:
            result['has_encrypted_data'] = False

        # Handle plaintext data
        if self.plaintext_data is not None:
            result['plaintext_data'] = self.plaintext_data
        else:
            result['plaintext_data'] = None

        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'HandoffPayload':
        """Create from dictionary."""
        nonce = data.get('nonce')
        if nonce:
            nonce = bytes.fromhex(nonce)

        return HandoffPayload(
            encrypted_data=None,  # Would deserialize in production
            plaintext_data=data.get('plaintext_data'),
            nonce=nonce,
            attestation=data.get('attestation'),
            data_shape=data.get('data_shape'),
            data_size=data.get('data_size', 0),
            operation=data.get('operation', ''),
        )


@dataclass
class ErrorPayload:
    """
    Error message payload.

    Contains error details.
    """
    error_code: ErrorCode
    error_message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_code': self.error_code.value,
            'error_message': self.error_message,
            'details': self.details,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ErrorPayload':
        """Create from dictionary."""
        return ErrorPayload(
            error_code=ErrorCode(data['error_code']),
            error_message=data['error_message'],
            details=data.get('details', {}),
        )


@dataclass
class ProtocolMessage:
    """
    Complete protocol message.

    Contains header and payload.
    """
    header: MessageHeader
    payload: Union[AttestationPayload, HandoffPayload, ErrorPayload, Dict[str, Any]]

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns:
            JSON serialized message
        """
        data = {
            'header': self.header.to_dict(),
        }

        # Serialize payload based on type
        if isinstance(self.payload, AttestationPayload):
            data['payload_type'] = 'attestation'
            data['payload'] = self.payload.to_dict()
        elif isinstance(self.payload, HandoffPayload):
            data['payload_type'] = 'handoff'
            data['payload'] = self.payload.to_dict()
        elif isinstance(self.payload, ErrorPayload):
            data['payload_type'] = 'error'
            data['payload'] = self.payload.to_dict()
        else:
            data['payload_type'] = 'generic'
            data['payload'] = self.payload

        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> 'ProtocolMessage':
        """
        Create from JSON string.

        Args:
            json_str: JSON serialized message

        Returns:
            ProtocolMessage instance
        """
        data = json.loads(json_str)
        header = MessageHeader.from_dict(data['header'])
        payload_type = data.get('payload_type', 'generic')
        payload_data = data['payload']

        # Deserialize payload based on type
        if payload_type == 'attestation':
            payload = AttestationPayload.from_dict(payload_data)
        elif payload_type == 'handoff':
            payload = HandoffPayload.from_dict(payload_data)
        elif payload_type == 'error':
            payload = ErrorPayload.from_dict(payload_data)
        else:
            payload = payload_data

        return ProtocolMessage(header=header, payload=payload)


class MessageBuilder:
    """
    Builder for creating protocol messages.
    """

    @staticmethod
    def create_message_id() -> str:
        """Create unique message ID."""
        return str(uuid4())

    @staticmethod
    def create_session_id() -> str:
        """Create unique session ID."""
        return str(uuid4())

    @staticmethod
    def create_attestation_request(
        session_id: str,
        sender: str,
        recipient: str
    ) -> ProtocolMessage:
        """
        Create attestation request message.

        Args:
            session_id: Session identifier
            sender: Sender identifier
            recipient: Recipient identifier

        Returns:
            ProtocolMessage
        """
        header = MessageHeader(
            message_id=MessageBuilder.create_message_id(),
            message_type=MessageType.ATTESTATION_REQUEST,
            session_id=session_id,
            timestamp=MessageBuilder._get_timestamp(),
            sender=sender,
            recipient=recipient
        )

        # Generate nonce for attestation
        nonce = secrets.token_bytes(32)

        payload = {
            'nonce': nonce.hex(),
            'request_type': 'attestation_challenge',
        }

        return ProtocolMessage(header=header, payload=payload)

    @staticmethod
    def create_attestation_response(
        session_id: str,
        sender: str,
        recipient: str,
        attestation_report: Dict[str, Any],
        measurement: bytes,
        nonce: bytes
    ) -> ProtocolMessage:
        """
        Create attestation response message.

        Args:
            session_id: Session identifier
            sender: Sender identifier
            recipient: Recipient identifier
            attestation_report: TEE attestation report
            measurement: Enclave measurement
            nonce: Challenge nonce

        Returns:
            ProtocolMessage
        """
        header = MessageHeader(
            message_id=MessageBuilder.create_message_id(),
            message_type=MessageType.ATTESTATION_RESPONSE,
            session_id=session_id,
            timestamp=MessageBuilder._get_timestamp(),
            sender=sender,
            recipient=recipient
        )

        payload = AttestationPayload(
            nonce=nonce,
            attestation_report=attestation_report,
            measurement=measurement
        )

        return ProtocolMessage(header=header, payload=payload)

    @staticmethod
    def create_handoff_message(
        session_id: str,
        sender: str,
        recipient: str,
        encrypted_data: Any = None,
        plaintext_data: Optional[List[float]] = None,
        nonce: Optional[bytes] = None,
        attestation: Optional[Dict[str, Any]] = None,
        operation: str = ""
    ) -> ProtocolMessage:
        """
        Create handoff message.

        Args:
            session_id: Session identifier
            sender: Sender identifier
            recipient: Recipient identifier
            encrypted_data: Encrypted data (from HE)
            plaintext_data: Plaintext data (from TEE)
            nonce: Freshness nonce
            attestation: Attestation report
            operation: Last operation performed

        Returns:
            ProtocolMessage
        """
        # Determine message type
        if sender == "he_server" and recipient == "tee_enclave":
            msg_type = MessageType.HE_TO_TEE_HANDOFF
        elif sender == "tee_enclave" and recipient == "he_server":
            msg_type = MessageType.TEE_TO_HE_HANDOFF
        else:
            msg_type =MessageType.HE_ENCRYPTED_DATA

        header = MessageHeader(
            message_id=MessageBuilder.create_message_id(),
            message_type=msg_type,
            session_id=session_id,
            timestamp=MessageBuilder._get_timestamp(),
            sender=sender,
            recipient=recipient
        )

        payload = HandoffPayload(
            encrypted_data=encrypted_data,
            plaintext_data=plaintext_data,
            nonce=nonce,
            attestation=attestation,
            operation=operation
        )

        return ProtocolMessage(header=header, payload=payload)

    @staticmethod
    def create_error_message(
        session_id: str,
        sender: str,
        recipient: str,
        error_code: ErrorCode,
        error_message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> ProtocolMessage:
        """
        Create error message.

        Args:
            session_id: Session identifier
            sender: Sender identifier
            recipient: Recipient identifier
            error_code: Error code
            error_message: Error description
            details: Additional error details

        Returns:
            ProtocolMessage
        """
        header = MessageHeader(
            message_id=MessageBuilder.create_message_id(),
            message_type=MessageType.ERROR,
            session_id=session_id,
            timestamp=MessageBuilder._get_timestamp(),
            sender=sender,
            recipient=recipient
        )

        payload = ErrorPayload(
            error_code=error_code,
            error_message=error_message,
            details=details or {}
        )

        return ProtocolMessage(header=header, payload=payload)

    @staticmethod
    def _get_timestamp() -> float:
        """Get current timestamp."""
        import time
        return time.time()


def create_session_id() -> str:
    """
    Create unique session ID.

    Returns:
        Session identifier
    """
    return MessageBuilder.create_session_id()


def create_message_id() -> str:
    """
    Create unique message ID.

    Returns:
        Message identifier
    """
    return MessageBuilder.create_message_id()
