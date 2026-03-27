"""
Unit Tests for HT2ML Protocol
===============================

Tests for protocol components:
- Message formats
- Handoff protocol
- Client and server protocol logic
"""

import sys
from pathlib import Path
# Add project root to path for imports (portable)
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

import unittest
import numpy as np
from src.protocol.message import (
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
from src.protocol.handoff import (
    HandoffProtocol,
    HandoffSession,
    HandoffResult,
    HandoffDirection,
    HandoffState,
    HandoffError,
)
from src.protocol.client import HT2MLClient, ClientSession
from src.tee.attestation import AttestationService, AttestationPolicy
from config.he_config import create_default_config
from config.model_config import create_default_config
from src.model.phishing_classifier import create_random_model


class TestMessageBuilder(unittest.TestCase):
    """Tests for MessageBuilder."""

    def test_create_message_id(self):
        """Test message ID creation."""
        msg_id = MessageBuilder.create_message_id()
        self.assertIsInstance(msg_id, str)
        self.assertGreater(len(msg_id), 0)

    def test_create_session_id(self):
        """Test session ID creation."""
        session_id = MessageBuilder.create_session_id()
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 0)

    def test_create_attestation_request(self):
        """Test attestation request creation."""
        session_id = "test_session"
        sender = "client"
        recipient = "tee_enclave"

        message = MessageBuilder.create_attestation_request(
            session_id, sender, recipient
        )

        self.assertEqual(message.header.message_type, MessageType.ATTESTATION_REQUEST)
        self.assertEqual(message.header.session_id, session_id)
        self.assertEqual(message.header.sender, sender)
        self.assertEqual(message.header.recipient, recipient)

    def test_create_attestation_response(self):
        """Test attestation response creation."""
        attestation_report = {
            'report_data': 'data',
            'nonce': 'nonce',
            'signature': 'sig',
            'enclave_measurement': 'meas',
            'timestamp': 1234567890.0,
        }

        message = MessageBuilder.create_attestation_response(
            session_id="session",
            sender="tee",
            recipient="client",
            attestation_report=attestation_report,
            measurement=b"measurement",
            nonce=b"nonce"
        )

        self.assertEqual(message.header.message_type, MessageType.ATTESTATION_RESPONSE)
        self.assertIsInstance(message.payload, AttestationPayload)

    def test_create_handoff_message(self):
        """Test handoff message creation."""
        message = MessageBuilder.create_handoff_message(
            session_id="session",
            sender="he_server",
            recipient="tee_enclave",
            nonce=b"nonce123",
            operation="test_op"
        )

        self.assertIn(message.header.message_type, [
            MessageType.HE_TO_TEE_HANDOFF,
            MessageType.TEE_TO_HE_HANDOFF,
        ])
        self.assertIsInstance(message.payload, HandoffPayload)

    def test_create_error_message(self):
        """Test error message creation."""
        message = MessageBuilder.create_error_message(
            session_id="session",
            sender="server",
            recipient="client",
            error_code=ErrorCode.ATTESTATION_FAILED,
            error_message="Attestation failed"
        )

        self.assertEqual(message.header.message_type, MessageType.ERROR)
        self.assertIsInstance(message.payload, ErrorPayload)
        self.assertEqual(message.payload.error_code, ErrorCode.ATTESTATION_FAILED)


class TestProtocolMessage(unittest.TestCase):
    """Tests for ProtocolMessage serialization."""

    def test_to_json(self):
        """Test JSON serialization."""
        header = MessageHeader(
            message_id="msg_1",
            message_type=MessageType.ATTESTATION_REQUEST,
            session_id="session_1",
            timestamp=1234567890.0,
            sender="client",
            recipient="server"
        )

        payload = {'test': 'data'}

        message = ProtocolMessage(header=header, payload=payload)
        json_str = message.to_json()

        self.assertIsInstance(json_str, str)
        self.assertIn('header', json_str)

    def test_from_json(self):
        """Test JSON deserialization."""
        header = MessageHeader(
            message_id="msg_1",
            message_type=MessageType.ATTESTATION_REQUEST,
            session_id="session_1",
            timestamp=1234567890.0,
            sender="client",
            recipient="server"
        )

        payload = {'test': 'data'}

        original_message = ProtocolMessage(header=header, payload=payload)
        json_str = original_message.to_json()

        restored_message = ProtocolMessage.from_json(json_str)

        self.assertEqual(
            restored_message.header.message_id,
            original_message.header.message_id
        )
        self.assertEqual(
            restored_message.header.message_type,
            original_message.header.message_type
        )


class TestHandoffProtocol(unittest.TestCase):
    """Tests for HandoffProtocol."""

    def setUp(self):
        """Set up test fixtures."""
        self.policy = AttestationPolicy()
        self.attestation_service = AttestationService(self.policy)
        self.protocol = HandoffProtocol(self.attestation_service)

    def test_initialization(self):
        """Test protocol initialization."""
        self.assertIsInstance(self.protocol.attestation_service, AttestationService)
        self.assertEqual(len(self.protocol.sessions), 0)

    def test_create_session(self):
        """Test session creation."""
        session = self.protocol.create_session()

        self.assertIsInstance(session, HandoffSession)
        self.assertEqual(session.state, HandoffState.IDLE)
        self.assertIsNotNone(session.session_id)

    def test_request_attestation(self):
        """Test attestation request."""
        session = self.protocol.create_session()

        request = self.protocol.request_attestation(session, "tee_enclave_1")

        self.assertEqual(request.header.message_type, MessageType.ATTESTATION_REQUEST)
        self.assertEqual(session.state, HandoffState.ATTESTATION_PENDING)
        self.assertIsNotNone(session.current_nonce)

    def test_verify_attestation(self):
        """Test attestation verification."""
        import hashlib
        import time

        session = self.protocol.create_session()
        self.protocol.request_attestation(session, "tee_enclave_1")

        measurement = b"test_measurement" * 2
        self.protocol.attestation_service.register_expected_measurement(
            "tee_enclave_1", measurement
        )

        # Create valid attestation
        report = {
            'report_data': hashlib.sha256(measurement + session.current_nonce).digest().hex(),
            'nonce': session.current_nonce.hex(),
            'signature': b"signature".hex(),
            'enclave_measurement': measurement.hex(),
            'timestamp': time.time(),
        }

        from src.protocol.message import MessageBuilder

        response = MessageBuilder.create_attestation_response(
            session_id=session.session_id,
            sender="tee_enclave_1",
            recipient="server",
            attestation_report=report,
            measurement=measurement,
            nonce=session.current_nonce
        )

        verified = self.protocol.verify_attestation(session, response, measurement)

        self.assertTrue(verified)
        self.assertEqual(session.state, HandoffState.ATTESTED)
        self.assertTrue(session.attestation_valid)

    def test_perform_handoff_he_to_tee(self):
        """Test HE to TEE handoff."""
        from src.he.encryption import CiphertextVector
        from config.he_config import CKKSParams

        # Create attested session
        session = self._create_attested_session()

        # Create encrypted data
        encrypted = CiphertextVector(
            data=[f"enc_{i}" for i in range(64)],
            size=64,
            shape=(64,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted.params = CKKSParams()

        # Perform handoff
        result = self.protocol.perform_handoff_he_to_tee(
            session, encrypted, "tee_enclave"
        )

        self.assertIsInstance(result, HandoffResult)
        self.assertTrue(result.success)
        self.assertEqual(result.direction, HandoffDirection.HE_TO_TEE)
        self.assertIsNotNone(result.output_data)

    def test_perform_handoff_tee_to_he(self):
        """Test TEE to HE handoff."""
        session = self._create_attested_session()

        plaintext = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = self.protocol.perform_handoff_tee_to_he(
            session, plaintext, "public_key"
        )

        self.assertIsInstance(result, HandoffResult)
        self.assertTrue(result.success)
        self.assertEqual(result.direction, HandoffDirection.TEE_TO_HE)

    def test_get_session_status(self):
        """Test get session status."""
        session = self.protocol.create_session()

        status = self.protocol.get_session_status(session.session_id)

        self.assertIsNotNone(status)
        self.assertEqual(status['session_id'], session.session_id)

    def _create_attested_session(self):
        """Helper to create attested session."""
        import hashlib
        import time

        session = self.protocol.create_session()
        self.protocol.request_attestation(session, "tee_enclave")

        measurement = b"test_measurement" * 2
        self.protocol.attestation_service.register_expected_measurement(
            "tee_enclave", measurement
        )

        report = {
            'report_data': hashlib.sha256(measurement + session.current_nonce).digest().hex(),
            'nonce': session.current_nonce.hex(),
            'signature': b"signature".hex(),
            'enclave_measurement': measurement.hex(),
            'timestamp': time.time(),
        }

        from src.protocol.message import MessageBuilder

        response = MessageBuilder.create_attestation_response(
            session_id=session.session_id,
            sender="tee_enclave",
            recipient="server",
            attestation_report=report,
            measurement=measurement,
            nonce=session.current_nonce
        )

        self.protocol.verify_attestation(session, response, measurement)

        return session


class TestHT2MLClient(unittest.TestCase):
    """Tests for HT2MLClient."""

    def setUp(self):
        """Set up test fixtures."""
        from config.he_config import create_default_config
        he_config = create_default_config()
        self.client = HT2MLClient(he_config)

    def test_initialization(self):
        """Test client initialization."""
        self.assertIsNotNone(self.client.he_client)
        self.assertIsNotNone(self.client.handoff_protocol)
        self.assertIsNone(self.client.current_session)

    def test_generate_keys(self):
        """Test key generation."""
        self.client.generate_keys()
        self.assertIsNotNone(self.client.he_client.context)

    def test_encrypt_input(self):
        """Test input encryption."""
        self.client.generate_keys()
        features = np.random.randn(50).astype(np.float32)

        encrypted = self.client.encrypt_input(features)

        self.assertIsNotNone(encrypted)

    def test_create_inference_session(self):
        """Test inference session creation."""
        self.client.generate_keys()
        features = np.random.randn(50).astype(np.float32)

        session = self.client.create_inference_session(features)

        self.assertIsInstance(session, ClientSession)
        self.assertTrue(session.has_keys())
        self.assertTrue(session.has_encrypted_input())

    def test_get_noise_status(self):
        """Test get noise status."""
        status = self.client.get_noise_status()

        self.assertIn('initial_budget', status)
        self.assertIn('consumed', status)
        self.assertIn('remaining', status)


def run_tests():
    """Run all protocol tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMessageBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestProtocolMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestHandoffProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestHT2MLClient))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
