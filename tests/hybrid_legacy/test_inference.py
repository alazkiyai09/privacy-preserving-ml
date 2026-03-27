"""
Unit Tests for HT2ML Inference Engines
========================================

Tests for all inference approaches:
- Hybrid HE/TEE inference
- HE-only inference
- TEE-only inference
- Accuracy verification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import numpy as np
from src.inference.hybrid_engine import (
    HybridInferenceEngine,
    HybridInferenceError,
    create_hybrid_engine,
)
from src.inference.he_only_engine import (
    HEOnlyInferenceEngine,
    HEOnlyEngineError,
    create_he_only_engine,
)
from src.inference.tee_only_engine import (
    TEEOnlyInferenceEngine,
    TEEOnlyEngineError,
    create_tee_only_engine,
)
from src.model.phishing_classifier import create_random_model
from config.model_config import create_default_config


class TestHybridInferenceEngine(unittest.TestCase):
    """Tests for HybridInferenceEngine."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)

    def test_initialization(self):
        """Test engine initialization."""
        engine = create_hybrid_engine(self.model)

        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.client)
        self.assertIsNotNone(engine.server)

    def test_run_inference(self):
        """Test running inference."""
        engine = create_hybrid_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result = engine.run_inference(features)

        self.assertIsNotNone(result)
        self.assertIn(result.class_id, [0, 1])
        self.assertGreater(result.execution_time_ms, 0)

    def test_run_inference_wrong_size(self):
        """Test inference with wrong input size."""
        engine = create_hybrid_engine(self.model)
        features = np.random.randn(100).astype(np.float32)  # Wrong size

        with self.assertRaises(HybridInferenceError):
            engine.run_inference(features)

    def test_run_batch_inference(self):
        """Test batch inference."""
        engine = create_hybrid_engine(self.model)
        features_batch = np.random.randn(3, 50).astype(np.float32)

        results = engine.run_batch_inference(features_batch)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn(result.class_id, [0, 1])

    def test_get_average_inference_time(self):
        """Test average time calculation."""
        engine = create_hybrid_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        engine.run_inference(features)
        # Reset noise tracker for second inference
        engine.server.he_engine.noise_tracker.reset()
        engine.run_inference(features)

        avg_time = engine.get_average_inference_time()

        self.assertGreater(avg_time, 0)

    def test_get_stats(self):
        """Test get statistics."""
        engine = create_hybrid_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        engine.run_inference(features)
        stats = engine.get_stats()

        self.assertIn('inference_count', stats)
        self.assertIn('total_inference_time_ms', stats)
        self.assertIn('client_noise_status', stats)
        self.assertIn('server_noise_status', stats)

    def test_reset_stats(self):
        """Test reset statistics."""
        engine = create_hybrid_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        engine.run_inference(features)
        engine.reset_stats()

        self.assertEqual(engine.inference_count, 0)
        self.assertEqual(engine.total_inference_time_ms, 0.0)


class TestHEOnlyInferenceEngine(unittest.TestCase):
    """Tests for HEOnlyInferenceEngine."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)

    def test_initialization(self):
        """Test engine initialization."""
        engine = create_he_only_engine(self.model)

        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.he_engine)

    def test_run_inference(self):
        """Test running HE-only inference."""
        engine = create_he_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result = engine.run_inference(features)

        self.assertIsNotNone(result)
        self.assertIn(result.class_id, [0, 1])
        self.assertEqual(result.num_handoffs, 0)  # No handoffs in HE-only
        self.assertGreater(result.he_time_ms, 0)
        self.assertEqual(result.tee_time_ms, 0)  # No TEE in HE-only

    def test_run_inference_wrong_size(self):
        """Test inference with wrong input size."""
        engine = create_he_only_engine(self.model)
        features = np.random.randn(100).astype(np.float32)

        with self.assertRaises(HEOnlyEngineError):
            engine.run_inference(features)

    def test_noise_consumption(self):
        """Test that HE-only consumes noise."""
        engine = create_he_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result = engine.run_inference(features)

        self.assertGreater(result.noise_budget_used, 0)
        self.assertLess(result.noise_budget_remaining, 200)

    def test_get_stats(self):
        """Test get statistics."""
        engine = create_he_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        engine.run_inference(features)
        stats = engine.get_stats()

        self.assertIn('inference_count', stats)
        self.assertIn('total_time_ms', stats)
        self.assertIn('noise_status', stats)


class TestTEEOnlyInferenceEngine(unittest.TestCase):
    """Tests for TEEOnlyInferenceEngine."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)

    def test_initialization(self):
        """Test engine initialization."""
        engine = create_tee_only_engine(self.model)

        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tee_enclave)
        self.assertIsNotNone(engine.tee_engine)

    def test_run_inference(self):
        """Test running TEE-only inference."""
        engine = create_tee_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result = engine.run_inference(features)

        self.assertIsNotNone(result)
        self.assertIn(result.class_id, [0, 1])
        self.assertEqual(result.num_handoffs, 0)  # No handoffs in TEE-only
        self.assertEqual(result.he_time_ms, 0)  # No HE in TEE-only
        self.assertGreater(result.tee_time_ms, 0)

    def test_run_inference_wrong_size(self):
        """Test inference with wrong input size."""
        engine = create_tee_only_engine(self.model)
        features = np.random.randn(100).astype(np.float32)

        with self.assertRaises(TEEOnlyEngineError):
            engine.run_inference(features)

    def test_no_noise_consumption(self):
        """Test that TEE-only doesn't consume noise."""
        engine = create_tee_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result = engine.run_inference(features)

        self.assertEqual(result.noise_budget_used, 0)
        self.assertEqual(result.noise_budget_remaining, 0)

    def test_run_batch_inference(self):
        """Test batch inference."""
        engine = create_tee_only_engine(self.model)
        features_batch = np.random.randn(3, 50).astype(np.float32)

        results = engine.run_batch_inference(features_batch)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn(result.class_id, [0, 1])

    def test_get_stats(self):
        """Test get statistics."""
        engine = create_tee_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        engine.run_inference(features)
        stats = engine.get_stats()

        self.assertIn('inference_count', stats)
        self.assertIn('total_time_ms', stats)
        self.assertIn('layer_times_ms', stats)
        self.assertIn('tee_status', stats)


class TestInferenceComparison(unittest.TestCase):
    """Tests comparing different inference approaches."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)
        self.features = np.random.randn(50).astype(np.float32)

    def test_all_engines_work(self):
        """Test that all inference engines work."""
        # Hybrid
        hybrid_engine = create_hybrid_engine(self.model)
        hybrid_result = hybrid_engine.run_inference(self.features)
        self.assertIn(hybrid_result.class_id, [0, 1])

        # HE-only
        he_engine = create_he_only_engine(self.model)
        he_result = he_engine.run_inference(self.features)
        self.assertIn(he_result.class_id, [0, 1])

        # TEE-only
        tee_engine = create_tee_only_engine(self.model)
        tee_result = tee_engine.run_inference(self.features)
        self.assertIn(tee_result.class_id, [0, 1])

    def test_performance_ordering(self):
        """Test that all approaches complete successfully."""
        hybrid_engine = create_hybrid_engine(self.model)
        he_engine = create_he_only_engine(self.model)
        tee_engine = create_tee_only_engine(self.model)

        hybrid_result = hybrid_engine.run_inference(self.features)
        he_result = he_engine.run_inference(self.features)
        tee_result = tee_engine.run_inference(self.features)

        # All should complete successfully with positive time
        self.assertGreater(hybrid_result.execution_time_ms, 0)
        self.assertGreater(he_result.execution_time_ms, 0)
        self.assertGreater(tee_result.execution_time_ms, 0)

    def test_handoff_counts(self):
        """Test handoff counts are correct."""
        hybrid_engine = create_hybrid_engine(self.model)
        he_engine = create_he_only_engine(self.model)
        tee_engine = create_tee_only_engine(self.model)

        hybrid_result = hybrid_engine.run_inference(self.features)
        he_result = he_engine.run_inference(self.features)
        tee_result = tee_engine.run_inference(self.features)

        self.assertEqual(hybrid_result.num_handoffs, 3)  # 3 handoffs in hybrid
        self.assertEqual(he_result.num_handoffs, 0)  # No handoffs in HE-only
        self.assertEqual(tee_result.num_handoffs, 0)  # No handoffs in TEE-only

    def test_noise_usage(self):
        """Test noise budget usage."""
        hybrid_engine = create_hybrid_engine(self.model)
        he_engine = create_he_only_engine(self.model)
        tee_engine = create_tee_only_engine(self.model)

        hybrid_result = hybrid_engine.run_inference(self.features)
        he_result = he_engine.run_inference(self.features)
        tee_result = tee_engine.run_inference(self.features)

        # HE and Hybrid should consume noise
        self.assertGreater(hybrid_result.noise_budget_used, 0)
        self.assertGreater(he_result.noise_budget_used, 0)

        # TEE-only should not consume noise
        self.assertEqual(tee_result.noise_budget_used, 0)


class TestModelIntegrity(unittest.TestCase):
    """Tests for model integrity and predictions."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.config = create_default_config()
        self.model = create_random_model(self.config)

    def test_model_loaded(self):
        """Test that model is properly loaded."""
        self.assertTrue(self.model.is_trained)
        self.assertGreater(self.model.get_num_parameters(), 0)

    def test_prediction_in_valid_range(self):
        """Test predictions are always valid."""
        engine = create_tee_only_engine(self.model)

        for _ in range(5):
            features = np.random.randn(50).astype(np.float32)
            result = engine.run_inference(features)

            self.assertIn(result.class_id, [0, 1])
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_deterministic_predictions(self):
        """Test predictions are deterministic with same input."""
        engine = create_tee_only_engine(self.model)
        features = np.random.randn(50).astype(np.float32)

        result1 = engine.run_inference(features)
        result2 = engine.run_inference(features)

        # Same input should give same prediction
        self.assertEqual(result1.class_id, result2.class_id)


def run_tests():
    """Run all inference tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHybridInferenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestHEOnlyInferenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTEEOnlyInferenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegrity))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
