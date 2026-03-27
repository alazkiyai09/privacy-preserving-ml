"""
Unit tests for secure protocols.

Tests for PSI, split finding, and prediction protocols.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from protocols.psi import PSIProtocol, SampleAlignment, simulate_vertical_partition
from protocols.split_finding import (
    SecureSplitFinding,
    ApproximateSplitFinding,
    compute_split_gains_histograms
)
from protocols.prediction import (
    SecurePrediction,
    LocalPartyPrediction,
    simulate_feature_partition
)
from core.tree_builder import TreeNode


class TestPSI:
    """Test Private Set Intersection."""

    def test_hash_id(self):
        """Test ID hashing."""
        psi = PSIProtocol()
        hash1 = psi.hash_id(123)
        hash2 = psi.hash_id(123)
        hash3 = psi.hash_id(456)

        assert hash1 == hash2  # Same ID should hash to same value
        assert hash1 != hash3  # Different IDs should hash differently
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_compute_hashed_ids(self):
        """Test computing hashed IDs."""
        psi = PSIProtocol()
        ids = {1, 2, 3, 4, 5}
        hashed = psi.compute_hashed_ids(ids)

        assert len(hashed) == 5
        assert all(isinstance(h, str) for h in hashed)

    def test_intersection(self):
        """Test PSI intersection."""
        psi = PSIProtocol()

        client_ids = psi.compute_hashed_ids({1, 2, 3, 4, 5})
        server_ids = {3, 4, 5, 6, 7}

        intersection = psi.server_intersection(client_ids, server_ids)

        # Should have 3, 4, 5 in common
        assert len(intersection) == 3

    def test_multi_party_psi(self):
        """Test multi-party PSI."""
        psi = PSIProtocol()

        party_ids = [
            {1, 2, 3, 4, 5},
            {3, 4, 5, 6, 7},
            {5, 6, 7, 8, 9}
        ]

        intersection = psi.multi_party_psi(party_ids)

        # Only 5 is in all three
        assert len(intersection) == 1

    def test_align_samples(self):
        """Test sample alignment."""
        psi = PSIProtocol()

        # Create fake features
        features_a = np.random.randn(5, 2)
        features_b = np.random.randn(5, 2)
        features_c = np.random.randn(5, 2)

        # Sample IDs with some overlap
        ids_a = {1, 2, 3, 4, 5}
        ids_b = {3, 4, 5, 6, 7}
        ids_c = {5, 6, 7, 8, 9}

        party_features = [features_a, features_b, features_c]
        party_ids = [ids_a, ids_b, ids_c]

        aligned, common_ids = psi.align_samples(party_features, party_ids)

        assert len(aligned) == 3
        assert len(common_ids) == 1  # Only ID 5 in all


class TestSampleAlignment:
    """Test sample alignment utilities."""

    def test_align_parties(self):
        """Test aligning samples across parties."""
        aligner = SampleAlignment()

        # Create sample data
        features_list = [
            np.random.randn(5, 2),
            np.random.randn(5, 2),
        ]

        sample_ids_list = [
            np.array([1, 2, 3, 4, 5]),
            np.array([3, 4, 5, 6, 7]),
        ]

        aligned, common_ids = aligner.align_parties(features_list, sample_ids_list)

        assert len(aligned) == 2
        assert len(common_ids) == 3  # IDs 3, 4, 5 in common

    def test_overlap_stats(self):
        """Test computing overlap statistics."""
        aligner = SampleAlignment()

        sample_ids_list = [
            np.array([1, 2, 3, 4, 5]),
            np.array([3, 4, 5, 6, 7]),
            np.array([5, 6, 7, 8, 9]),
        ]

        stats = aligner.compute_overlap_stats(sample_ids_list)

        assert stats['n_parties'] == 3
        assert stats['full_intersection_size'] == 1  # Only ID 5
        assert 'avg_pairwise_overlap' in stats


class TestSimulateVerticalPartition:
    """Test vertical partition simulation."""

    def test_simulate_partition(self):
        """Test simulating vertical data partition."""
        X = np.random.randn(100, 10)

        features, ids = simulate_vertical_partition(
            X, n_parties=3, overlap_ratio=0.8, random_state=42
        )

        assert len(features) == 3
        assert len(ids) == 3

        # Check feature counts
        total_features = sum(f.shape[1] for f in features)
        assert total_features == X.shape[1]


class TestSecureSplitFinding:
    """Test secure split finding."""

    def test_approximate_histogram(self):
        """Test approximate histogram building."""
        finder = ApproximateSplitFinding(n_parties=2, n_approx_bins=5)

        feature_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gradients = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        hessians = np.ones(5)

        hist, edges = finder.approximate_histogram(
            feature_values, gradients, hessians
        )

        assert hist.shape[1] == 3  # 3 columns (G, H, count)
        assert len(edges) >= 2


class TestSecurePrediction:
    """Test secure prediction."""

    def test_simulate_feature_partition(self):
        """Test feature partition simulation."""
        party_features = simulate_feature_partition(
            n_features=10, n_parties=3, random_state=42
        )

        assert len(party_features) == 3
        assert sum(len(f) for f in party_features) == 10

    def test_local_party_can_predict(self):
        """Test checking if party can predict."""
        # Create simple tree
        leaf1 = TreeNode(node_id=1, is_leaf=True, depth=1, leaf_value=1.0, sample_count=10)
        leaf2 = TreeNode(node_id=2, is_leaf=True, depth=1, leaf_value=-1.0, sample_count=10)
        root = TreeNode(
            node_id=0, is_leaf=False, depth=0,
            feature_idx=0, split_value=0.5,
            left_child=leaf1, right_child=leaf2
        )

        # Party with feature 0
        party = LocalPartyPrediction(party_id=0, feature_indices=[0, 1, 2])
        assert party.can_predict(root) == True

        # Party without feature 0
        party2 = LocalPartyPrediction(party_id=1, feature_indices=[3, 4, 5])
        assert party2.can_predict(root) == False

    def test_local_party_predict(self):
        """Test local party prediction."""
        # Create simple tree
        leaf1 = TreeNode(node_id=1, is_leaf=True, depth=1, leaf_value=1.0, sample_count=10)
        leaf2 = TreeNode(node_id=2, is_leaf=True, depth=1, leaf_value=-1.0, sample_count=10)
        root = TreeNode(
            node_id=0, is_leaf=False, depth=0,
            feature_idx=0, split_value=0.5,
            left_child=leaf1, right_child=leaf2
        )

        party = LocalPartyPrediction(party_id=0, feature_indices=[0, 1, 2])

        # Feature value < split value
        result1 = party.predict_with_available_features(root, np.array([0.3, 1.0, 2.0]))
        assert result1 == 1.0

        # Feature value >= split value
        result2 = party.predict_with_available_features(root, np.array([0.7, 1.0, 2.0]))
        assert result2 == -1.0

        # Create a tree with feature 5 (which party doesn't have)
        leaf3 = TreeNode(node_id=3, is_leaf=True, depth=1, leaf_value=2.0, sample_count=10)
        leaf4 = TreeNode(node_id=4, is_leaf=True, depth=1, leaf_value=-2.0, sample_count=10)
        root2 = TreeNode(
            node_id=0, is_leaf=False, depth=0,
            feature_idx=5, split_value=0.5,  # Feature 5 not in party's features
            left_child=leaf3, right_child=leaf4
        )

        # Missing feature (feature 5 not in party's feature_indices)
        result3 = party.predict_with_available_features(root2, np.array([0.3, 1.0, 2.0]))
        assert result3 is None


class TestSecurePredictionClass:
    """Test SecurePrediction class."""

    def test_init(self):
        """Test initialization."""
        party_features = [[0, 1], [2, 3], [4, 5]]
        predictor = SecurePrediction(
            n_parties=3,
            party_feature_indices=party_features
        )

        assert predictor.n_parties == 3
        assert len(predictor.feature_to_party) == 6

    def test_predict_single_tree(self):
        """Test predicting with a single tree."""
        # Create tree that uses feature 0
        leaf1 = TreeNode(node_id=1, is_leaf=True, depth=1, leaf_value=1.0, sample_count=5)
        leaf2 = TreeNode(node_id=2, is_leaf=True, depth=1, leaf_value=-1.0, sample_count=5)
        root = TreeNode(
            node_id=0, is_leaf=False, depth=0,
            feature_idx=0, split_value=0.5,
            left_child=leaf1, right_child=leaf2
        )

        party_features = [[0, 1], [2, 3]]
        predictor = SecurePrediction(
            n_parties=2,
            party_feature_indices=party_features
        )

        # Create feature matrices
        X_dict = {
            0: np.array([[0.3, 1.0], [0.7, 2.0]]),  # Party 0 has feature 0
            1: np.array([[1.0, 2.0], [3.0, 4.0]])   # Party 1 has feature 2,3
        }

        # First sample: feature 0 value is 0.3 < 0.5 -> goes left -> leaf value 1.0
        result1 = predictor.predict_single_tree(root, X_dict, sample_idx=0)
        assert result1 == 1.0

        # Second sample: feature 0 value is 0.7 >= 0.5 -> goes right -> leaf value -1.0
        result2 = predictor.predict_single_tree(root, X_dict, sample_idx=1)
        assert result2 == -1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
