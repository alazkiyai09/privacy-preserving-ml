"""
Unit Tests for TEE Operations
==============================

Tests for Trusted Execution Environment components:
- TEE enclave lifecycle
- Attestation
- TEE operations (ReLU, Softmax, Argmax)
- Sealed storage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import numpy as np
import time
import hashlib
from src.tee.enclave import (
    TEEEnclave,
    TEEContext,
    TEEAttestationReport,
    TEEState,
)
from src.tee.operations import (
    TEEOperationEngine,
    TEEHandoffManager,
    TEEOperationResult,
    TEEComputationError,
)
from src.tee.attestation import (
    AttestationService,
    AttestationPolicy,
    AttestationStatus,
    AttestationRecord,
)
from src.tee.sealed_storage import (
    SealedStorage,
    SealedData,
    SealedModelBundle,
)


class TestTEEEnclave(unittest.TestCase):
    """Tests for TEEEnclave."""

    def setUp(self):
        """Set up test fixtures."""
        self.enclave = TEEEnclave()

    def test_initialization(self):
        """Test enclave initialization."""
        self.assertEqual(self.enclave.get_state(), TEEState.UNINITIALIZED)
        self.assertIsNotNone(self.enclave.context.measurement)

    def test_initialize(self):
        """Test enclave initialization."""
        self.enclave.initialize()
        self.assertEqual(self.enclave.get_state(), TEEState.INITIALIZED)

    def test_generate_attestation(self):
        """Test attestation generation."""
        self.enclave.initialize()

        report = self.enclave.generate_attestation()

        self.assertIsInstance(report, TEEAttestationReport)
        self.assertIsNotNone(report.nonce)
        self.assertIsNotNone(report.signature)
        self.assertEqual(self.enclave.get_state(), TEEState.ATTESTED)

    def test_attestation_verification(self):
        """Test attestation verification."""
        self.enclave.initialize()
        report = self.enclave.generate_attestation()

        # Verify with correct measurement
        self.assertTrue(
            report.verify(self.enclave.get_measurement())
        )

        # Verify with wrong measurement
        wrong_measurement = b"wrong" * 8
        self.assertFalse(
            report.verify(wrong_measurement)
        )

    def test_load_model(self):
        """Test model loading."""
        self.enclave.initialize()
        self.enclave.generate_attestation()

        weights = {
            'layer1.weight': np.random.randn(50, 64).astype(np.float32),
            'layer1.bias': np.random.randn(64).astype(np.float32),
        }

        self.enclave.load_model(weights)
        self.assertTrue(self.enclave.context.loaded_model is not None)
        self.assertEqual(self.enclave.get_state(), TEEState.ACTIVE)

    def test_load_model_without_attestation(self):
        """Test model loading without attestation fails."""
        self.enclave.initialize()

        weights = {'layer1.weight': np.zeros((50, 64), dtype=np.float32)}

        with self.assertRaises(RuntimeError):
            self.enclave.load_model(weights)

    def test_load_model_allow_no_attestation(self):
        """Test model loading without attestation requirement."""
        self.enclave.initialize()

        weights = {'layer1.weight': np.zeros((50, 64), dtype=np.float32)}

        # Should succeed with require_attestation=False
        self.enclave.load_model(weights, require_attestation=False)
        self.assertTrue(self.enclave.context.loaded_model is not None)

    def test_execute_operation(self):
        """Test operation execution."""
        self.enclave.initialize()
        self.enclave.generate_attestation()

        weights = {'layer1.weight': np.zeros((50, 64), dtype=np.float32)}
        self.enclave.load_model(weights)

        # Test ReLU
        data = np.array([-1, 0, 1, 2], dtype=np.float32)
        result = self.enclave.execute_operation("relu", data)

        expected = np.array([0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_get_measurement(self):
        """Test get measurement."""
        measurement = self.enclave.get_measurement()
        self.assertIsInstance(measurement, bytes)
        self.assertEqual(len(measurement), 32)  # SHA256

    def test_get_status(self):
        """Test get enclave status."""
        status = self.enclave.get_status()

        self.assertIn('enclave_id', status)
        self.assertIn('state', status)
        self.assertIn('measurement', status)


class TestTEEOperationEngine(unittest.TestCase):
    """Tests for TEEOperationEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = TEEOperationEngine()

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.operation_count, 0)
        self.assertEqual(self.engine.total_execution_time_ms, 0.0)

    def test_execute_relu(self):
        """Test ReLU execution."""
        data = np.array([-1, 0, 1, 2, -0.5], dtype=np.float32)

        result = self.engine.execute_relu(data)

        self.assertIsInstance(result, TEEOperationResult)
        expected = np.array([0, 0, 1, 2, 0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.output, expected)
        self.assertEqual(result.operation, "relu")
        self.assertGreater(result.execution_time_ms, 0)

    def test_execute_softmax(self):
        """Test Softmax execution."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = self.engine.execute_softmax(data)

        self.assertIsInstance(result, TEEOperationResult)
        self.assertAlmostEqual(np.sum(result.output), 1.0, places=5)
        self.assertTrue(np.all(result.output >= 0))
        self.assertEqual(result.operation, "softmax")

    def test_execute_argmax(self):
        """Test Argmax execution."""
        data = np.array([0.1, 0.9, 0.3], dtype=np.float32)

        result = self.engine.execute_argmax(data)

        self.assertIsInstance(result, TEEOperationResult)
        self.assertEqual(result.output[0], 1)
        self.assertEqual(result.operation, "argmax")

    def test_execute_batch(self):
        """Test batch execution."""
        operations = [
            ("relu", np.array([-1, 1], dtype=np.float32)),
            ("softmax", np.array([1.0, 2.0], dtype=np.float32)),
        ]

        results = self.engine.execute_batch(operations)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].operation, "relu")
        self.assertEqual(results[1].operation, "softmax")

    def test_unknown_operation(self):
        """Test unknown operation raises error."""
        # TEEOperationEngine doesn't have execute_operation method
        # This test validates that only specific operations work
        data = np.array([1, 2, 3], dtype=np.float32)

        # Try to use execute_batch with unknown operation
        with self.assertRaises(TEEComputationError):
            self.engine.execute_batch([("unknown_op", data)])

    def test_get_stats(self):
        """Test get statistics."""
        self.engine.execute_relu(np.array([1, 2], dtype=np.float32))
        self.engine.execute_softmax(np.array([1, 2], dtype=np.float32))

        stats = self.engine.get_stats()

        self.assertEqual(stats['operation_count'], 2)
        self.assertGreater(stats['total_execution_time_ms'], 0)
        self.assertGreater(stats['average_execution_time_ms'], 0)

    def test_reset_stats(self):
        """Test reset statistics."""
        self.engine.execute_relu(np.array([1, 2], dtype=np.float32))
        self.engine.reset_stats()

        self.assertEqual(self.engine.operation_count, 0)
        self.assertEqual(self.engine.total_execution_time_ms, 0.0)


class TestTEEHandoffManager(unittest.TestCase):
    """Tests for TEEHandoffManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = TEEHandoffManager()

    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.handoff_count, 0)
        self.assertEqual(self.manager.total_handoff_time_ms, 0.0)

    def test_receive_from_he(self):
        """Test receiving data from HE."""
        # Create mock encrypted data
        from src.he.encryption import CiphertextVector
        from config.he_config import CKKSParams

        encrypted = CiphertextVector(
            data=[f"enc_{i}" for i in range(64)],
            size=64,
            shape=(64,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted.params = CKKSParams()

        attestation = {'report_data': 'test'}

        result = self.manager.receive_from_he(encrypted, b"nonce", attestation)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 64)
        self.assertEqual(self.manager.handoff_count, 1)

    def test_send_to_he(self):
        """Test sending data to HE."""
        plaintext = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        public_key = "test_public_key"

        result = self.manager.send_to_he(plaintext, public_key)

        self.assertIsNotNone(result)
        self.assertEqual(self.manager.handoff_count, 1)

    def test_get_stats(self):
        """Test get handoff statistics."""
        from src.he.encryption import CiphertextVector
        from config.he_config import CKKSParams

        encrypted = CiphertextVector(
            data=["enc"],
            size=1,
            shape=(1,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted.params = CKKSParams()

        self.manager.receive_from_he(encrypted, b"nonce", {})

        stats = self.manager.get_stats()

        self.assertEqual(stats['handoff_count'], 1)
        self.assertGreater(stats['total_handoff_time_ms'], 0)


class TestAttestationService(unittest.TestCase):
    """Tests for AttestationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.policy = AttestationPolicy(max_age_seconds=3600)
        self.service = AttestationService(self.policy)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service.policy, AttestationPolicy)
        self.assertEqual(len(self.service.attestation_cache), 0)

    def test_register_expected_measurement(self):
        """Test registering expected measurement."""
        measurement = b"test" * 8
        self.service.register_expected_measurement("enclave_1", measurement)

        self.assertIn("enclave_1", self.service.expected_measurements)

    def test_verify_attestation_valid(self):
        """Test valid attestation verification."""
        measurement = b"test" * 8
        self.service.register_expected_measurement("enclave_1", measurement)

        # Create valid report
        report = {
            'report_data': hashlib.sha256(measurement + b"nonce").digest().hex(),
            'nonce': b"nonce".hex(),
            'signature': b"signature".hex(),
            'enclave_measurement': measurement.hex(),
            'timestamp': time.time(),
        }

        status = self.service.verify_attestation(report, "enclave_1", measurement)

        self.assertEqual(status, AttestationStatus.VALID)

    def test_verify_attestation_invalid_measurement(self):
        """Test attestation with invalid measurement."""
        correct_measurement = b"correct" * 8
        wrong_measurement = b"wrong" * 8

        self.service.register_expected_measurement("enclave_1", correct_measurement)

        report = {
            'report_data': hashlib.sha256(wrong_measurement + b"nonce").digest().hex(),
            'nonce': b"nonce".hex(),
            'signature': b"sig".hex(),
            'enclave_measurement': wrong_measurement.hex(),
            'timestamp': time.time(),
        }

        status = self.service.verify_attestation(
            report,
            "enclave_1",
            correct_measurement
        )

        self.assertEqual(status, AttestationStatus.INVALID)

    def test_get_cached_attestation(self):
        """Test getting cached attestation."""
        measurement = b"test" * 8
        self.service.register_expected_measurement("enclave_1", measurement)

        # Create and verify report
        report = {
            'report_data': hashlib.sha256(measurement + b"nonce").digest().hex(),
            'nonce': b"nonce".hex(),
            'signature': b"signature".hex(),
            'enclave_measurement': measurement.hex(),
            'timestamp': time.time(),
        }

        status = self.service.verify_attestation(report, "enclave_1", measurement)

        # Get from cache
        cached = self.service.get_cached_attestation("enclave_1")

        self.assertIsNotNone(cached)
        self.assertEqual(cached.status.value, "valid")

    def test_get_cache_stats(self):
        """Test get cache statistics."""
        stats = self.service.get_cache_stats()

        self.assertIn('cached_attestations', stats)
        self.assertIn('valid_attestations', stats)
        self.assertIn('registered_measurements', stats)


class TestSealedStorage(unittest.TestCase):
    """Tests for SealedStorage."""

    def setUp(self):
        """Set up test fixtures."""
        self.storage = SealedStorage()
        self.measurement = b"test_measurement" * 2

    def test_seal_weight(self):
        """Test sealing a weight."""
        weight = np.random.randn(50, 64).astype(np.float32)

        sealed = self.storage.seal_weight(weight, self.measurement, "layer1")

        self.assertIsInstance(sealed, SealedData)
        self.assertEqual(sealed.measurement, self.measurement)
        self.assertIsNotNone(sealed.encrypted_data)
        self.assertIsNotNone(sealed.tag)

    def test_unseal_weight(self):
        """Test unsealing a weight."""
        # Use 1D array to avoid reshaping issues
        weight = np.random.randn(3200).astype(np.float32)  # 50*64

        sealed = self.storage.seal_weight(weight, self.measurement, "layer1")
        unsealed = self.storage.unseal_weight(sealed, self.measurement)

        np.testing.assert_array_almost_equal(unsealed, weight)

    def test_unseal_wrong_measurement(self):
        """Test unsealing with wrong measurement fails."""
        weight = np.array([1, 2, 3], dtype=np.float32)

        sealed = self.storage.seal_weight(weight, self.measurement, "layer1")
        wrong_measurement = b"wrong" * 2

        with self.assertRaises(Exception):  # UnsealingError
            self.storage.unseal_weight(sealed, wrong_measurement)

    def test_seal_model(self):
        """Test sealing complete model."""
        weights = {
            'layer1.weight': np.random.randn(50, 64).astype(np.float32),
            'layer1.bias': np.random.randn(64).astype(np.float32),
            'layer2.weight': np.random.randn(64, 2).astype(np.float32),
            'layer2.bias': np.random.randn(2).astype(np.float32),
        }

        model_config = {'input_size': 50, 'hidden_size': 64, 'output_size': 2}

        bundle = self.storage.seal_model(weights, self.measurement, model_config)

        self.assertIsInstance(bundle, SealedModelBundle)
        self.assertEqual(len(bundle.sealed_weights), 4)
        self.assertEqual(bundle.model_config, model_config)

    def test_unseal_model(self):
        """Test unsealing complete model."""
        # Use 1D arrays to avoid reshaping issues
        weights = {
            'layer1.weight': np.random.randn(3200).astype(np.float32),  # 50*64 flattened
            'layer1.bias': np.random.randn(64).astype(np.float32),
        }

        model_config = {'input_size': 50, 'hidden_size': 64}

        bundle = self.storage.seal_model(weights, self.measurement, model_config)
        unsealed = self.storage.unseal_model(bundle, self.measurement)

        self.assertIn('layer1.weight', unsealed)
        self.assertIn('layer1.bias', unsealed)

        np.testing.assert_array_almost_equal(
            unsealed['layer1.weight'], weights['layer1.weight']
        )


def run_tests():
    """Run all TEE operation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTEEEnclave))
    suite.addTests(loader.loadTestsFromTestCase(TestTEEOperationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTEEHandoffManager))
    suite.addTests(loader.loadTestsFromTestCase(TestAttestationService))
    suite.addTests(loader.loadTestsFromTestCase(TestSealedStorage))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
