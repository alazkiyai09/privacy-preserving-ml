"""
Protocol Module for HT2ML
=========================

Implements HE↔TEE communication protocol with
secure handoff and attestation verification.
"""

from src.encryption.hybrid.legacy.protocol.message import (
    MessageType,
    ErrorCode,
    MessageHeader,
    AttestationPayload,
    HandoffPayload,
    ErrorPayload,
    ProtocolMessage,
    MessageBuilder,
    create_session_id,
    create_message_id,
)

from src.encryption.hybrid.legacy.protocol.handoff import (
    HandoffDirection,
    HandoffState,
    HandoffResult,
    HandoffSession,
    HandoffProtocol,
    HandoffError,
    create_handoff_protocol,
)

__all__ = [
    # Message types
    'MessageType',
    'ErrorCode',
    'MessageHeader',
    'AttestationPayload',
    'HandoffPayload',
    'ErrorPayload',
    'ProtocolMessage',
    'MessageBuilder',
    'create_session_id',
    'create_message_id',

    # Handoff protocol
    'HandoffDirection',
    'HandoffState',
    'HandoffResult',
    'HandoffSession',
    'HandoffProtocol',
    'HandoffError',
    'create_handoff_protocol',
]
