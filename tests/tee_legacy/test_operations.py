"""
Unit Tests for TEE ML Operations
=================================

Tests for:
- Activation functions
- Comparison operations
- Arithmetic operations
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave
from tee_ml.operations.activations import (
    tee_relu,
    tee_sigmoid,
    tee_tanh,
    tee_softmax,
    tee_leaky_relu,
    tee_elu,
    tee_gelu,
    tee_swish,
    TeeActivationLayer,
    compare_activation_costs,
)
from tee_ml.operations.comparisons import (
    tee_argmax,
    tee_argmin,
    tee_threshold,
    tee_equal,
    tee_top_k,
    tee_where,
    tee_clip,
    tee_maximum,
    tee_minimum,
    tee_compare,
    tee_sort,
    tee_argsort,
    tee_allclose,
    tee_count_nonzero,
    TeeComparisonLayer,
)
from tee_ml.operations.arithmetic import (
    tee_divide,
    tee_normalize,
    tee_layer_normalization,
    tee_batch_normalization,
    tee_standardize,
    tee_min_max_scale,
    tee_log,
    tee_exp,
    tee_sqrt,
    tee_power,
    tee_reciprocal,
    tee_log_softmax,
    tee_l2_normalize,
    tee_l1_normalize,
    TeeArithmeticLayer,
)


class TestActivations:
    """Test activation functions in TEE."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-activations")

    @pytest.fixture
    def session(self, enclave):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return enclave.enter(data)

    def test_relu(self, session):
        """Test ReLU activation."""
        result = tee_relu(session.input_data, session)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_relu_negative(self, enclave):
        """Test ReLU with negative values."""
        data = np.array([-1.0, -2.0, 0.0, 1.0, 2.0])
        session = enclave.enter(data)
        result = tee_relu(data, session)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(result, expected)

    def test_sigmoid(self, session):
        """Test sigmoid activation."""
        result = tee_sigmoid(session.input_data, session)
        # Sigmoid should produce values in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)
        # Check monotonicity
        assert result[0] < result[1] < result[2]

    def test_tanh(self, session):
        """Test tanh activation."""
        result = tee_tanh(session.input_data, session)
        # Tanh should produce values in (-1, 1)
        assert np.all(result > -1)
        assert np.all(result < 1)
        # Check monotonicity
        assert result[0] < result[1] < result[2]

    def test_softmax(self, session):
        """Test softmax activation."""
        result = tee_softmax(session.input_data, session)
        # Softmax should sum to 1
        assert np.allclose(np.sum(result), 1.0)
        # All values should be positive
        assert np.all(result > 0)

    def test_leaky_relu(self, session):
        """Test leaky ReLU."""
        # Test with positive values
        result = tee_leaky_relu(session.input_data, session, alpha=0.01)
        expected = session.input_data  # Positive values unchanged
        assert np.allclose(result, expected)

    def test_leaky_relu_negative(self, enclave):
        """Test leaky ReLU with negative values."""
        data = np.array([-1.0, -2.0, 0.0, 1.0, 2.0])
        session = enclave.enter(data)
        result = tee_leaky_relu(data, session, alpha=0.1)
        # Negative values scaled
        assert result[0] == -0.1
        assert result[1] == -0.2
        # Positive and zero unchanged
        assert result[2] == 0.0
        assert result[3] == 1.0

    def test_elu(self, session):
        """Test ELU activation."""
        result = tee_elu(session.input_data, session, alpha=1.0)
        # Positive values unchanged
        assert np.allclose(result[:5], session.input_data)

    def test_gelu(self, session):
        """Test GELU activation."""
        result = tee_gelu(session.input_data, session)
        # GELU should be smooth and positive for positive inputs
        assert np.all(result > 0)
        # Check monotonicity
        assert result[0] < result[1] < result[2]

    def test_swish(self, session):
        """Test Swish activation."""
        result = tee_swish(session.input_data, session, beta=1.0)
        # Swish should be smooth
        assert np.all(result > 0)
        # Check monotonicity
        assert result[0] < result[1] < result[2]

    def test_activation_layer_relu(self, session):
        """Test TeeActivationLayer with ReLU."""
        layer = TeeActivationLayer('relu')
        result = layer.forward(session.input_data, session)
        expected = np.maximum(0, session.input_data)
        assert np.allclose(result, expected)

    def test_activation_layer_sigmoid(self, session):
        """Test TeeActivationLayer with sigmoid."""
        layer = TeeActivationLayer('sigmoid')
        result = layer.forward(session.input_data, session)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_activation_layer_softmax(self, session):
        """Test TeeActivationLayer with softmax."""
        layer = TeeActivationLayer('softmax', axis=-1)
        result = layer.forward(session.input_data, session)
        assert np.allclose(np.sum(result), 1.0)

    def test_activation_layer_none(self, session):
        """Test TeeActivationLayer with no activation."""
        layer = TeeActivationLayer('none')
        result = layer.forward(session.input_data, session)
        assert np.allclose(result, session.input_data)

    def test_activation_layer_unknown(self, session):
        """Test TeeActivationLayer with unknown activation."""
        layer = TeeActivationLayer('unknown_activation')
        with pytest.raises(ValueError):
            layer.forward(session.input_data, session)

    def test_compare_activation_costs(self):
        """Test activation cost comparison."""
        costs = compare_activation_costs(['relu', 'sigmoid', 'softmax'])

        assert 'relu' in costs
        assert 'sigmoid' in costs
        assert 'softmax' in costs

        # TEE should be much faster than HE
        assert costs['relu']['speedup'] > 1
        assert costs['sigmoid']['speedup'] > 1


class TestComparisons:
    """Test comparison operations in TEE."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-comparisons")

    @pytest.fixture
    def session(self, enclave):
        data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
        return enclave.enter(data)

    def test_argmax(self, session):
        """Test argmax operation."""
        result = tee_argmax(session.input_data, session)
        assert result == 3  # Index of 9.0

    def test_argmin(self, session):
        """Test argmin operation."""
        result = tee_argmin(session.input_data, session)
        assert result == 0  # Index of 1.0

    def test_threshold(self, session):
        """Test threshold operation."""
        result = tee_threshold(session.input_data, session, threshold=3.0)
        expected = np.array([False, True, False, True, False])
        assert np.array_equal(result, expected)

    def test_equal(self, session):
        """Test equality check."""
        result = tee_equal(session.input_data, session, value=3.0)
        expected = np.array([False, False, True, False, False])
        assert np.array_equal(result, expected)

    def test_top_k(self, session):
        """Test top-k operation."""
        values, indices = tee_top_k(session.input_data, session, k=2)

        # Top 2 values should be 9.0 and 5.0
        assert np.allclose(values, [9.0, 5.0])
        # Their indices should be 3 and 1
        assert np.array_equal(indices, [3, 1])

    def test_where(self, enclave):
        """Test where operation."""
        condition = np.array([True, False, True, False, True])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        session = enclave.enter(condition)
        result = tee_where(condition, x, y, session)
        expected = np.array([1.0, 20.0, 3.0, 40.0, 5.0])
        assert np.array_equal(result, expected)

    def test_clip(self, session):
        """Test clip operation."""
        result = tee_clip(session.input_data, session, min_val=2.0, max_val=5.0)
        expected = np.array([2.0, 5.0, 3.0, 5.0, 2.0])
        assert np.allclose(result, expected)

    def test_maximum(self, session):
        """Test maximum operation."""
        result = tee_maximum(session.input_data, session, other=3.0)
        expected = np.array([3.0, 5.0, 3.0, 9.0, 3.0])
        assert np.allclose(result, expected)

    def test_minimum(self, session):
        """Test minimum operation."""
        result = tee_minimum(session.input_data, session, other=3.0)
        expected = np.array([1.0, 3.0, 3.0, 3.0, 2.0])
        assert np.allclose(result, expected)

    def test_compare_greater(self, session):
        """Test generic comparison (>).
        """
        result = tee_compare(session.input_data, session, operator='>', value=3.0)
        expected = np.array([False, True, False, True, False])
        assert np.array_equal(result, expected)

    def test_compare_less_equal(self, session):
        """Test generic comparison (<=)."""
        result = tee_compare(session.input_data, session, operator='<=', value=3.0)
        expected = np.array([True, False, True, False, True])
        assert np.array_equal(result, expected)

    def test_compare_unknown_operator(self, session):
        """Test unknown comparison operator."""
        with pytest.raises(ValueError):
            tee_compare(session.input_data, session, operator='%%', value=3.0)

    def test_sort(self, session):
        """Test sort operation."""
        result = tee_sort(session.input_data, session)
        expected = np.array([1.0, 2.0, 3.0, 5.0, 9.0])
        assert np.allclose(result, expected)

    def test_argsort(self, session):
        """Test argsort operation."""
        result = tee_argsort(session.input_data, session)
        sorted_values = session.input_data[result]
        expected = np.array([1.0, 2.0, 3.0, 5.0, 9.0])
        assert np.allclose(sorted_values, expected)

    def test_allclose(self, enclave):
        """Test allclose operation."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.1])

        session = enclave.enter(x)
        assert tee_allclose(x, session, y)
        assert not tee_allclose(x, session, z)

    def test_count_nonzero(self, enclave):
        """Test count_nonzero operation."""
        x = np.array([0, 1, 0, 2, 0, 3])
        session = enclave.enter(x)
        result = tee_count_nonzero(x, session)
        assert result == 3

    def test_comparison_layer_argmax(self, session):
        """Test TeeComparisonLayer with argmax."""
        layer = TeeComparisonLayer('argmax', axis=-1)
        result = layer.forward(session.input_data, session)
        assert result == 3

    def test_comparison_layer_threshold(self, session):
        """Test TeeComparisonLayer with threshold."""
        layer = TeeComparisonLayer('threshold', threshold=3.0)
        result = layer.forward(session.input_data, session)
        expected = np.array([False, True, False, True, False])
        assert np.array_equal(result, expected)


class TestArithmetic:
    """Test arithmetic operations in TEE."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-arithmetic")

    @pytest.fixture
    def session(self, enclave):
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        return enclave.enter(data)

    def test_divide(self, session):
        """Test division operation."""
        result = tee_divide(session.input_data, session, divisor=2.0)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_reciprocal(self, session):
        """Test reciprocal operation."""
        result = tee_reciprocal(session.input_data, session)
        expected = np.array([0.5, 0.25, 0.1667, 0.125, 0.1], dtype=np.float32)
        assert np.allclose(result, expected, atol=0.001)

    def test_power(self, session):
        """Test power operation."""
        result = tee_power(session.input_data, session, exponent=2.0)
        expected = np.array([4.0, 16.0, 36.0, 64.0, 100.0])
        assert np.allclose(result, expected)

    def test_sqrt(self, session):
        """Test sqrt operation."""
        result = tee_sqrt(session.input_data, session)
        expected = np.array([1.414, 2.0, 2.449, 2.828, 3.162], dtype=np.float32)
        assert np.allclose(result, expected, atol=0.01)

    def test_normalize(self, session):
        """Test L2 normalization."""
        result = tee_normalize(session.input_data, session)
        # Check that result is normalized
        norm = np.linalg.norm(result)
        assert np.allclose(norm, 1.0, atol=0.01)

    def test_standardize(self, session):
        """Test standardization (z-score)."""
        result = tee_standardize(session.input_data, session)
        # Mean should be ~0, std should be ~1
        assert np.allclose(np.mean(result), 0.0, atol=0.01)
        assert np.allclose(np.std(result), 1.0, atol=0.01)

    def test_min_max_scale(self, session):
        """Test min-max scaling."""
        result = tee_min_max_scale(session.input_data, session, feature_range=(0, 1))
        # Min should be 0, max should be 1
        assert np.allclose(np.min(result), 0.0)
        assert np.allclose(np.max(result), 1.0)

    def test_log(self, enclave):
        """Test natural logarithm."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        session = enclave.enter(data)
        result = tee_log(data, session)
        # Log should be monotonic
        assert result[0] < result[1] < result[2]

    def test_exp(self, enclave):
        """Test exponential."""
        data = np.array([0.0, 1.0, 2.0])
        session = enclave.enter(data)
        result = tee_exp(data, session)
        # exp(0) = 1
        assert np.allclose(result[0], 1.0)
        # Should be monotonic
        assert result[0] < result[1] < result[2]

    def test_log_softmax(self, enclave):
        """Test log-softmax."""
        data = np.array([1.0, 2.0, 3.0])
        session = enclave.enter(data)
        result = tee_log_softmax(data, session)
        # Log-softmax should sum to log(sum(exp(x)))
        log_sum_exp = np.log(np.sum(np.exp(data)))
        assert np.allclose(np.sum(np.exp(result)), 1.0, atol=0.01)

    def test_l2_normalize(self, session):
        """Test L2 normalization (same as normalize)."""
        result = tee_l2_normalize(session.input_data, session)
        norm = np.linalg.norm(result)
        assert np.allclose(norm, 1.0, atol=0.01)

    def test_l1_normalize(self, session):
        """Test L1 normalization."""
        result = tee_l1_normalize(session.input_data, session)
        # L1 norm should be 1
        l1_norm = np.sum(np.abs(result))
        assert np.allclose(l1_norm, 1.0, atol=0.01)

    def test_layer_normalization(self, enclave):
        """Test layer normalization."""
        # 2D input
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        session = enclave.enter(data)

        gamma = np.ones(3)
        beta = np.zeros(3)

        result = tee_layer_normalization(data, session, gamma, beta, axis=-1)

        # Each row should have mean ~0 and std ~1
        for i in range(len(data)):
            assert np.allclose(np.mean(result[i]), 0.0, atol=0.01)
            assert np.allclose(np.std(result[i]), 1.0, atol=0.01)

    def test_batch_normalization(self, enclave):
        """Test batch normalization (inference mode)."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        session = enclave.enter(data)

        mean = np.array([2.5, 3.5, 4.5])
        var = np.array([2.25, 2.25, 2.25])
        gamma = np.ones(3)
        beta = np.zeros(3)

        result = tee_batch_normalization(data, session, mean, var, gamma, beta)

        # Result should be normalized
        assert result.shape == data.shape

    def test_arithmetic_layer_divide(self, session):
        """Test TeeArithmeticLayer with divide."""
        layer = TeeArithmeticLayer('divide', divisor=2.0)
        result = layer.forward(session.input_data, session)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_arithmetic_layer_normalize(self, session):
        """Test TeeArithmeticLayer with normalize."""
        layer = TeeArithmeticLayer('normalize')
        result = layer.forward(session.input_data, session)
        norm = np.linalg.norm(result)
        assert np.allclose(norm, 1.0, atol=0.01)

    def test_arithmetic_layer_unknown(self, session):
        """Test TeeArithmeticLayer with unknown operation."""
        layer = TeeArithmeticLayer('unknown_op')
        with pytest.raises(ValueError):
            layer.forward(session.input_data, session)


class TestIntegration:
    """Integration tests for TEE operations."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-integration")

    def test_neural_network_layer_forward(self, enclave):
        """Test complete forward pass through layers."""
        # Input
        x = np.array([1.0, 2.0, 3.0, 4.0])
        session = enclave.enter(x)

        # Layer 1: Linear (simulate with multiply) + ReLU
        W1 = np.array([[0.5, 0.5, 0.5, 0.5]])
        x1 = x @ W1.T
        x1 = tee_relu(x1, session)

        # Layer 2: Sigmoid
        x2 = tee_sigmoid(x1, session)

        # Output
        assert x2.shape == (1,)
        assert 0 < x2[0] < 1

        enclave.exit(session)

    def test_classification_pipeline(self, enclave):
        """Test complete classification pipeline."""
        # Simulate logits from neural network
        logits = np.array([2.0, 1.0, 0.1, 3.0])
        session = enclave.enter(logits)

        # Apply softmax
        probs = tee_softmax(logits, session)

        # Get prediction (argmax)
        prediction = tee_argmax(probs, session)

        assert prediction == 3  # Index of largest value
        assert np.allclose(np.sum(probs), 1.0)

        enclave.exit(session)

    def test_preprocessing_pipeline(self, enclave):
        """Test data preprocessing pipeline."""
        # Raw data
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        session = enclave.enter(data)

        # Step 1: Standardize
        normalized = tee_standardize(data, session)

        # Step 2: Clip to range [-2, 2]
        clipped = tee_clip(normalized, session, min_val=-2, max_val=2)

        # Step 3: L2 normalize
        final = tee_normalize(clipped, session)

        assert final.shape == data.shape

        enclave.exit(session)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
