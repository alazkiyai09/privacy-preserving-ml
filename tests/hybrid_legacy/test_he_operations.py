"""
Unit Tests for HE Operations
=============================

Tests for homomorphic encryption components:
- HE encryption/decryption
- Linear layer execution
- Noise budget tracking
- Key management
"""

import sys
from pathlib import Path
# Add project root to path for imports (portable)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import numpy as np
from src.he.encryption import (
    HEEncryptionClient,
    HEOperationEngine,
    CiphertextVector,
    NoiseTracker,
    NoiseBudgetExceededError,
)
from src.he.keys import HEKeyManager, KeyPair
from src.he.noise_tracker import NoiseBudget, NoiseWarning
from config.he_config import create_default_config


class TestNoiseTracker(unittest.TestCase):
    """Tests for NoiseTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = NoiseTracker(initial_budget=200)

    def test_initialization(self):
        """Test noise tracker initialization."""
        self.assertEqual(self.tracker.initial_budget, 200)
        self.assertEqual(self.tracker.current_budget, 200)
        self.assertEqual(self.tracker.total_consumed, 0)
        self.assertEqual(len(self.tracker.operations), 0)

    def test_consume_noise(self):
        """Test noise consumption."""
        self.assertTrue(self.tracker.consume_noise(50, "test operation"))
        self.assertEqual(self.tracker.current_budget, 150)
        self.assertEqual(self.tracker.total_consumed, 50)
        self.assertEqual(len(self.tracker.operations), 1)

    def test_consume_noise_exceed(self):
        """Test noise budget exceeded."""
        with self.assertRaises(NoiseBudgetExceededError):
            self.tracker.consume_noise(250, "test operation")

    def test_consume_allow_exceed(self):
        """Test consume more than budget raises error."""
        with self.assertRaises(NoiseBudgetExceededError):
            self.tracker.consume_noise(250, "test")

    def test_get_remaining_budget(self):
        """Test get remaining budget."""
        self.tracker.consume_noise(50, "test")
        self.assertEqual(self.tracker.get_remaining_budget(), 150)

    def test_get_consumed_budget(self):
        """Test get consumed budget."""
        self.tracker.consume_noise(50, "test")
        self.assertEqual(self.tracker.get_consumed_budget(), 50)

    def test_can_perform_operation(self):
        """Test can perform operation."""
        self.assertTrue(self.tracker.can_perform_operation(100))
        self.assertFalse(self.tracker.can_perform_operation(250))


class TestNoiseBudget(unittest.TestCase):
    """Tests for NoiseBudget class."""

    def setUp(self):
        """Set up test fixtures."""
        self.budget = NoiseBudget(initial_budget=200, warning_threshold=0.8)

    def test_initialization(self):
        """Test budget initialization."""
        self.assertEqual(self.budget.initial_budget, 200)
        self.assertEqual(self.budget.current_budget, 200)
        self.assertEqual(self.budget.warning_threshold, 0.8)

    def test_consume_noise(self):
        """Test noise consumption."""
        result = self.budget.consume_noise(50, "test op")
        self.assertTrue(result)
        self.assertEqual(self.budget.current_budget, 150)
        self.assertEqual(self.budget.total_consumed, 50)

    def test_warning_levels(self):
        """Test warning level tracking."""
        # Safe
        self.budget.consume_noise(50, "test1")
        warning = self.budget.consumption_history[-1].warning_level
        self.assertEqual(warning, NoiseWarning.SAFE)

        # Moderate (50%)
        self.budget.consume_noise(50, "test2")
        warning = self.budget.consumption_history[-1].warning_level
        self.assertEqual(warning, NoiseWarning.MODERATE)

        # High (80%)
        self.budget.consume_noise(60, "test3")
        warning = self.budget.consumption_history[-1].warning_level
        self.assertEqual(warning, NoiseWarning.HIGH)

    def test_get_warning_level(self):
        """Test get current warning level."""
        self.assertEqual(self.budget.get_warning_level(), NoiseWarning.SAFE)

        self.budget.consume_noise(160, "test")
        self.assertEqual(self.budget.get_warning_level(), NoiseWarning.HIGH)

    def test_get_budget_percentage(self):
        """Test get budget percentage."""
        self.budget.consume_noise(100, "test")
        self.assertEqual(self.budget.get_budget_percentage(), 50.0)

    def test_estimate_noise_cost(self):
        """Test noise cost estimation."""
        # Linear layer
        cost = self.budget.estimate_noise_cost(50, 64, "linear")
        self.assertGreater(cost, 0)

        # Addition
        cost = self.budget.estimate_noise_cost(64, 64, "add")
        self.assertGreater(cost, 0)

    def test_reset(self):
        """Test budget reset."""
        self.budget.consume_noise(100, "test")
        self.budget.reset(new_budget=300)
        self.assertEqual(self.budget.initial_budget, 300)
        self.assertEqual(self.budget.current_budget, 300)
        self.assertEqual(self.budget.total_consumed, 0)


class TestHEEncryptionClient(unittest.TestCase):
    """Tests for HEEncryptionClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.client = HEEncryptionClient(self.config)

    def test_initialization(self):
        """Test client initialization."""
        self.assertIsNotNone(self.client.config)
        self.assertIsNone(self.client.context)
        self.assertIsNotNone(self.client.noise_tracker)

    def test_generate_keys(self):
        """Test key generation."""
        self.client.generate_keys()
        self.assertIsNotNone(self.client.context)
        self.assertEqual(self.client.context.get_security_level(), 160)

    def test_encrypt_vector(self):
        """Test vector encryption."""
        self.client.generate_keys()
        features = np.random.randn(50).astype(np.float32)

        encrypted = self.client.encrypt_vector(features)
        self.assertIsInstance(encrypted, CiphertextVector)
        self.assertEqual(encrypted.size, 50)

    def test_encrypt_wrong_size(self):
        """Test encryption with wrong input size."""
        self.client.generate_keys()
        features = np.random.randn(100).astype(np.float32)

        with self.assertRaises(ValueError):
            self.client.encrypt_vector(features)

    def test_decrypt_result(self):
        """Test result decryption."""
        self.client.generate_keys()

        # Create mock encrypted result
        from src.he.encryption import CiphertextVector
        encrypted_result = CiphertextVector(
            data=["result"],
            size=1,
            shape=(1,),
            scale=2**40,
            scheme="CKKS"
        )

        decrypted = self.client.decrypt_result(encrypted_result)
        self.assertIsNotNone(decrypted)


class TestHEOperationEngine(unittest.TestCase):
    """Tests for HEOperationEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_default_config()
        self.engine = HEOperationEngine(self.config)

    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.config)
        self.assertIsNotNone(self.engine.noise_tracker)

    def test_execute_linear_layer(self):
        """Test linear layer execution."""
        # Create mock encrypted input
        from src.he.encryption import CiphertextVector
        encrypted_input = CiphertextVector(
            data=[f"enc_{i}" for i in range(50)],
            size=50,
            shape=(50,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted_input.params = self.config.ckks_params

        # Create weights and bias
        weights = np.random.randn(50, 64).astype(np.float32) * 0.1
        bias = np.random.randn(64).astype(np.float32) * 0.01

        # Execute
        result = self.engine.execute_linear_layer(encrypted_input, weights, bias)

        self.assertIsInstance(result, CiphertextVector)
        self.assertEqual(result.size, 64)

    def test_noise_consumption(self):
        """Test that linear layer consumes noise."""
        from src.he.encryption import CiphertextVector

        initial_budget = self.engine.noise_tracker.get_remaining_budget()

        encrypted_input = CiphertextVector(
            data=[f"enc_{i}" for i in range(50)],
            size=50,
            shape=(50,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted_input.params = self.config.ckks_params

        weights = np.random.randn(50, 64).astype(np.float32) * 0.1
        bias = np.random.randn(64).astype(np.float32) * 0.01

        self.engine.execute_linear_layer(encrypted_input, weights, bias)

        final_budget = self.engine.noise_tracker.get_remaining_budget()

        self.assertLess(final_budget, initial_budget)

    def test_noise_budget_exceeded(self):
        """Test noise budget exceeded error."""
        from src.he.encryption import CiphertextVector

        # Create tracker with very small budget
        self.engine.noise_tracker = NoiseTracker(initial_budget=10)

        encrypted_input = CiphertextVector(
            data=[f"enc_{i}" for i in range(50)],
            size=50,
            shape=(50,),
            scale=2**40,
            scheme="CKKS"
        )
        encrypted_input.params = self.config.ckks_params

        weights = np.random.randn(50, 64).astype(np.float32) * 0.1
        bias = np.random.randn(64).astype(np.float32) * 0.01

        with self.assertRaises(NoiseBudgetExceededError):
            self.engine.execute_linear_layer(encrypted_input, weights, bias)

    def test_get_noise_status(self):
        """Test get noise status."""
        status = self.engine.get_noise_status()

        self.assertIn('initial_budget', status)
        self.assertIn('consumed', status)
        self.assertIn('remaining', status)
        self.assertIn('operations_performed', status)


class TestHEKeyManager(unittest.TestCase):
    """Tests for HE key management."""

    def setUp(self):
        """Set up test fixtures."""
        from config.he_config import create_default_config
        self.config = create_default_config()
        self.manager = HEKeyManager(self.config)

    def test_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager.config)
        self.assertIsNone(self.manager.current_key_pair)

    def test_generate_key_pair(self):
        """Test key pair generation."""
        key_pair = self.manager.generate_key_pair()

        self.assertIsInstance(key_pair, KeyPair)
        self.assertIsNotNone(key_pair.key_id)
        self.assertIsNotNone(key_pair.public_key)
        self.assertIsNotNone(key_pair.secret_key)

    def test_get_public_key(self):
        """Test get public key."""
        self.manager.generate_key_pair()
        public_key = self.manager.get_public_key()

        self.assertIsNotNone(public_key)

    def test_has_secret_key(self):
        """Test has secret key."""
        self.assertFalse(self.manager.has_secret_key())

        self.manager.generate_key_pair()
        self.assertTrue(self.manager.has_secret_key())

    def test_rotate_keys(self):
        """Test key rotation."""
        old_key = self.manager.generate_key_pair()
        old_key_id = old_key.key_id

        new_key = self.manager.rotate_keys(old_key_id)

        self.assertIsInstance(new_key, KeyPair)
        self.assertNotEqual(new_key.key_id, old_key_id)

    def test_key_fingerprint(self):
        """Test key fingerprint."""
        fingerprint = self.manager.get_key_fingerprint("test_key")

        self.assertIsInstance(fingerprint, str)
        self.assertEqual(len(fingerprint), 64)  # SHA256 hex


def run_tests():
    """Run all HE operation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestNoiseTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseBudget))
    suite.addTests(loader.loadTestsFromTestCase(TestHEEncryptionClient))
    suite.addTests(loader.loadTestsFromTestCase(TestHEOperationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestHEKeyManager))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
