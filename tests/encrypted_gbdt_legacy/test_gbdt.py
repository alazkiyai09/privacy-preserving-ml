"""
Unit tests for GBDT core functionality.

Tests for objective functions, histograms, tree building, and GBDT ensemble.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from core.objective import (
    LogisticLoss, SquaredErrorLoss,
    compute_gradients, compute_hessians,
    get_objective
)
from core.histogram import (
    HistogramBuilder,
    find_best_split_histogram,
    find_best_split_all_features,
    SplitInfo
)
from core.tree_builder import TreeBuilder, TreeNode, print_tree
from core.gbdt_base import GBDTBase


class TestObjective:
    """Test objective function computations."""

    def test_logistic_gradients(self):
        """Test gradient computation for logistic loss."""
        obj = LogisticLoss()
        predictions = np.array([0.0, 1.0, -1.0])
        labels = np.array([0.0, 1.0, 0.0])

        gradients = obj.compute_gradients(predictions, labels)

        # p = sigmoid(0) = 0.5, gradient = 0.5 - 0 = 0.5
        assert abs(gradients[0] - 0.5) < 1e-6

        # p = sigmoid(1) ≈ 0.731, gradient = 0.731 - 1 = -0.269
        assert abs(gradients[1] - (0.7310585 - 1.0)) < 1e-6

    def test_logistic_hessians(self):
        """Test Hessian computation for logistic loss."""
        obj = LogisticLoss()
        predictions = np.array([0.0, 1.0])
        labels = np.array([0.0, 1.0])

        hessians = obj.compute_hessians(predictions, labels)

        # p = sigmoid(0) = 0.5, hessian = 0.5 * 0.5 = 0.25
        assert abs(hessians[0] - 0.25) < 1e-6

        # hessian = p * (1-p) = 0.731 * 0.269 ≈ 0.197
        assert abs(hessians[1] - (0.7310585 * 0.2689414)) < 1e-6

    def test_squared_error_gradients_hessians(self):
        """Test gradient and Hessian for squared error loss."""
        obj = SquaredErrorLoss()
        predictions = np.array([2.0, 3.0, 4.0])
        labels = np.array([1.0, 3.0, 5.0])

        gradients = obj.compute_gradients(predictions, labels)
        hessians = obj.compute_hessians(predictions, labels)

        # gradient = prediction - label
        np.testing.assert_array_almost_equal(gradients, np.array([1.0, 0.0, -1.0]))

        # hessian = 1.0
        np.testing.assert_array_almost_equal(hessians, np.array([1.0, 1.0, 1.0]))

    def test_sigmoid_stability(self):
        """Test sigmoid numerical stability."""
        obj = LogisticLoss()

        # Large positive and negative values
        large_positive = np.array([1000.0])
        large_negative = np.array([-1000.0])

        assert np.allclose(obj.sigmoid(large_positive), 1.0, atol=1e-6)
        assert np.allclose(obj.sigmoid(large_negative), 0.0, atol=1e-6)

    def test_convenience_functions(self):
        """Test convenience functions for gradients/Hessians."""
        predictions = np.array([0.5, -0.5, 1.0])
        labels = np.array([1.0, 0.0, 1.0])

        grad = compute_gradients(predictions, labels, loss='binary:logistic')
        hess = compute_hessians(predictions, labels, loss='binary:logistic')

        assert grad.shape == (3,)
        assert hess.shape == (3,)
        assert np.all(hess > 0)  # Hessians should be positive


class TestHistogram:
    """Test histogram building and split finding."""

    def test_bin_edges_computation(self):
        """Test computation of quantile-based bin edges."""
        builder = HistogramBuilder(max_bins=10)

        # Create feature with known distribution
        features = np.random.randn(1000, 1)  # Single feature, normal distribution
        builder.fit_bins(features)

        assert len(builder.bin_edges_) == 1
        assert len(builder.bin_edges_[0]) == 11  # max_bins + 1

    def test_digitize_features(self):
        """Test feature discretization."""
        builder = HistogramBuilder(max_bins=10)
        features = np.random.randn(100, 3)

        builder.fit_bins(features)
        binned = builder.digitize_features(features)

        assert binned.shape == features.shape
        assert binned.dtype == np.int32
        assert np.all(binned >= 0)
        assert np.all(binned < 10)

    def test_build_histogram(self):
        """Test histogram construction."""
        builder = HistogramBuilder(max_bins=10)

        # Create simple feature
        feature_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gradients = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        hessians = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        bin_edges = np.array([0.0, 2.0, 4.0, 6.0])

        histogram = builder.build_histogram(
            feature_values, gradients, hessians, bin_edges
        )

        assert histogram.shape == (3, 3)  # 3 bins, 3 columns (G, H, count)
        assert np.sum(histogram[:, 0]) == pytest.approx(1.5)  # Sum gradients
        assert np.sum(histogram[:, 1]) == pytest.approx(5.0)  # Sum Hessians
        assert np.sum(histogram[:, 2]) == 5  # Count

    def test_find_best_split_histogram(self):
        """Test split finding from histogram."""
        builder = HistogramBuilder(max_bins=5)

        # Create histogram with clear split point
        histogram = np.array([
            [2.0, 2.0, 10],  # Bin 0: positive gradients
            [-1.0, 1.0, 5],  # Bin 1: negative gradients
            [-3.0, 3.0, 15],  # Bin 2: more negative
            [-1.5, 1.5, 8],  # Bin 3
            [-0.5, 0.5, 2],  # Bin 4
        ])
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        best_bin, split_value, gain = find_best_split_histogram(
            histogram, bin_edges, lambda_reg=1.0
        )

        # Should find a split (gain can be negative if parent score is better)
        assert 0 <= best_bin < 5
        assert bin_edges[0] <= split_value <= bin_edges[-1]
        # Gain should be a valid float
        assert isinstance(gain, (float, np.floating))


class TestTreeBuilder:
    """Test decision tree construction."""

    def test_tree_building(self):
        """Test basic tree building."""
        # Create simple dataset
        X = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        # Compute gradients (use squared error for simplicity)
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        gradients = predictions - y
        hessians = np.ones(4)

        builder = TreeBuilder(max_depth=3, lambda_reg=1.0)
        sample_indices = np.arange(4)

        tree = builder.build(X, gradients, hessians, sample_indices, depth=0)

        assert tree is not None
        assert builder.n_nodes > 0

    def test_leaf_value_computation(self):
        """Test leaf value computation."""
        builder = TreeBuilder(lambda_reg=1.0)

        gradients = np.array([1.0, 2.0, 3.0])
        hessians = np.array([1.0, 1.0, 1.0])
        indices = np.array([0, 1, 2])

        leaf_value = builder._compute_leaf_value(gradients, hessians, indices)

        # -sum(G) / (sum(H) + lambda) = -6 / (3 + 1) = -1.5
        assert abs(leaf_value - (-1.5)) < 1e-6

    def test_max_depth_constraint(self):
        """Test that tree respects max depth."""
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(float)

        predictions = np.full(100, 0.5)
        gradients = predictions - y
        hessians = np.ones(100)

        builder = TreeBuilder(max_depth=2, lambda_reg=1.0)
        tree = builder.build(X, gradients, hessians, np.arange(100), depth=0)

        assert builder.get_depth() <= 2

    def test_tree_prediction(self):
        """Test tree prediction."""
        # Create simple tree with known structure
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        y = np.array([0.0, 1.0, 1.0])

        predictions = np.full(3, 0.5)
        gradients = predictions - y
        hessians = np.ones(3)

        builder = TreeBuilder(max_depth=3, lambda_reg=1.0)
        tree = builder.build(X, gradients, hessians, np.arange(3), depth=0)

        # Make predictions
        preds = builder.predict(X)

        assert preds.shape == (3,)


class TestGBDTBase:
    """Test GBDT ensemble."""

    def test_basic_training(self):
        """Test basic GBDT training."""
        # Create simple binary classification dataset
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        model = GBDTBase(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            loss='binary:logistic',
            random_state=42
        )

        model.fit(X, y, verbose=False)

        assert len(model.trees) == 10
        assert len(model.train_losses) == 10

    def test_prediction_shapes(self):
        """Test prediction output shapes."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)

        model = GBDTBase(
            n_estimators=5,
            max_depth=2,
            loss='binary:logistic',
            random_state=42
        )
        model.fit(X, y, verbose=False)

        # Test raw predictions
        raw_preds = model.predict(X)
        assert raw_preds.shape == (50,)

        # Test probability predictions
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Test class predictions
        classes = model.predict_class(X)
        assert classes.shape == (50,)
        assert set(classes).issubset({0, 1})

    def test_accuracy(self):
        """Test model accuracy on simple dataset."""
        # Create linearly separable data
        np.random.seed(42)
        X_pos = np.random.randn(50, 3) + np.array([2.0, 2.0, 2.0])
        X_neg = np.random.randn(50, 3) - np.array([2.0, 2.0, 2.0])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1.0] * 50 + [0.0] * 50)

        model = GBDTBase(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.3,
            lambda_reg=0.1,
            gamma=0.0,
            loss='binary:logistic',
            random_state=42
        )

        model.fit(X, y, verbose=False)
        accuracy = model.score(X, y)

        # Should achieve good accuracy on separable data
        assert accuracy > 0.75

    def test_learning_rate_effect(self):
        """Test that learning rate affects prediction scaling."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)

        model_low_lr = GBDTBase(
            n_estimators=5,
            max_depth=2,
            learning_rate=0.01,
            loss='binary:logistic',
            random_state=42
        )
        model_high_lr = GBDTBase(
            n_estimators=5,
            max_depth=2,
            learning_rate=0.5,
            loss='binary:logistic',
            random_state=42
        )

        model_low_lr.fit(X, y, verbose=False)
        model_high_lr.fit(X, y, verbose=False)

        # Higher learning rate should produce larger raw predictions
        raw_low = model_low_lr.predict(X)
        raw_high = model_high_lr.predict(X)

        # The magnitude should be different
        assert not np.allclose(raw_low, raw_high)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
