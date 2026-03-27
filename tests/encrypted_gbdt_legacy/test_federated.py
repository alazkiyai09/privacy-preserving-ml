"""
Integration tests for federated GBDT.

Tests end-to-end federated training workflow.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from federated.client import BankClient, ClientManager
from federated.server import FederatedServer, Coordinator, TrainingConfig
from federated.label_holder import LabelHolder


class TestBankClient:
    """Test bank client functionality."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = BankClient(
            client_id=0,
            feature_indices=[0, 1, 2],
            max_bins=10
        )

        assert client.client_id == 0
        assert client.get_feature_indices() == [0, 1, 2]

    def test_compute_histograms(self):
        """Test histogram computation."""
        client = BankClient(
            client_id=0,
            feature_indices=[0, 1],
            max_bins=5
        )

        # Create fake data
        features = np.random.randn(100, 2)
        gradients = np.random.randn(100)
        hessians = np.ones(100)
        sample_indices = np.arange(100)

        # Compute histograms
        histograms = client.compute_local_histograms(
            features, gradients, hessians, sample_indices
        )

        assert len(histograms) == 2  # 2 features
        assert all(h.shape[1] == 3 for h in histograms)  # Each has 3 columns


class TestClientManager:
    """Test client manager."""

    def test_partition_features(self):
        """Test feature partitioning."""
        manager = ClientManager(n_clients=3, n_features=10)

        partition = manager.partition_features(random_state=42)

        assert len(partition) == 3
        assert sum(len(f) for f in partition) == 10

    def test_create_clients(self):
        """Test creating clients."""
        manager = ClientManager(n_clients=3, n_features=9)

        clients = manager.create_clients(max_bins=10)

        assert len(clients) == 3
        assert all(isinstance(c, BankClient) for c in clients)

    def test_get_client(self):
        """Test getting specific client."""
        manager = ClientManager(n_clients=3, n_features=9)
        clients = manager.create_clients()

        client = manager.get_client(0)

        assert client.client_id == 0


class TestFederatedServer:
    """Test federated server."""

    def test_server_initialization(self):
        """Test server initialization."""
        config = TrainingConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )

        server = FederatedServer(n_clients=3, config=config)

        assert server.n_clients == 3
        assert server.config.n_estimators == 10

    def test_initialize_training(self):
        """Test training initialization."""
        config = TrainingConfig(loss='binary:logistic')
        server = FederatedServer(n_clients=3, config=config)

        labels = np.array([0, 1, 0, 1, 1])
        server.initialize_training(n_samples=5, labels=labels)

        assert server.current_predictions is not None
        assert len(server.current_predictions) == 5

    def test_compute_gradients(self):
        """Test gradient computation."""
        config = TrainingConfig(loss='binary:logistic')
        server = FederatedServer(n_clients=3, config=config)

        labels = np.array([0, 1, 0, 1, 1])
        server.initialize_training(n_samples=5, labels=labels)

        gradients, hessians = server.compute_gradients_hessians(labels)

        assert len(gradients) == 5
        assert len(hessians) == 5

    def test_distribute_gradients(self):
        """Test gradient distribution."""
        config = TrainingConfig()
        server = FederatedServer(n_clients=3, config=config)

        gradients = np.array([0.1, 0.2, 0.3])
        hessians = np.array([1.0, 1.0, 1.0])

        client_data = server.distribute_gradients(gradients, hessians)

        assert len(client_data) == 3
        assert all(0 in client_data for client_id in client_data)


class TestLabelHolder:
    """Test label holder."""

    def test_label_holder_initialization(self):
        """Test initialization."""
        labels = np.array([0, 1, 0, 1, 1])
        holder = LabelHolder(labels, loss='binary:logistic')

        assert holder.n_samples == 5
        assert np.array_equal(holder.get_labels(), labels)

    def test_initialize_base_score(self):
        """Test base score initialization."""
        labels = np.array([0, 0, 1, 1])  # Balanced
        holder = LabelHolder(labels, loss='binary:logistic')

        base_score = holder.initialize_base_score()

        # Should be close to 0 for balanced classes
        assert abs(base_score) < 0.1

    def test_compute_gradients_hessians(self):
        """Test gradient/hessian computation."""
        labels = np.array([0, 1, 0, 1])
        holder = LabelHolder(labels, loss='binary:logistic')

        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        holder.set_predictions(predictions)

        gradients, hessians = holder.compute_gradients_hessians()

        assert len(gradients) == 4
        assert len(hessians) == 4
        assert np.all(hessians > 0)  # Hessians should be positive for logistic loss


class TestCoordinator:
    """Test training coordinator."""

    def test_coordinator_initialization(self):
        """Test initialization."""
        config = TrainingConfig(n_estimators=10)
        coord = Coordinator(n_clients=3, config=config)

        assert coord.server.n_clients == 3

    def test_orchestrate_round(self):
        """Test training round orchestration."""
        config = TrainingConfig(n_estimators=5, max_depth=2)
        coord = Coordinator(n_clients=2, config=config)

        labels = np.array([0, 1, 0, 1])
        coord.server.initialize_training(n_samples=4, labels=labels)

        # Fake client features
        client_features = {
            0: np.random.randn(4, 2),
            1: np.random.randn(4, 2)
        }

        sample_indices = np.arange(4)

        result = coord.orchestrate_training_round(
            iteration=0,
            labels=labels,
            client_features=client_features,
            sample_indices=sample_indices
        )

        assert 'iteration' in result
        assert 'split' in result
        assert result['iteration'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
