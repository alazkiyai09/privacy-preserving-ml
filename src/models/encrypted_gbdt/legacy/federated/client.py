"""
Federated client (bank) implementation.

Each bank holds a subset of features and computes local histograms.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from core.histogram import HistogramBuilder


@dataclass
class ClientConfig:
    """Configuration for a federated client."""
    client_id: int
    feature_indices: List[int]
    n_features: int
    max_bins: int = 256


class BankClient:
    """
    Bank client in federated GBDT.

    Each bank holds a subset of features for the samples and participates
    in secure split finding by computing local histograms.
    """

    def __init__(self,
                 client_id: int,
                 feature_indices: List[int],
                 max_bins: int = 256):
        """
        Initialize bank client.

        Args:
            client_id: Unique identifier for this client
            feature_indices: Indices of features this client holds
            max_bins: Maximum number of histogram bins
        """
        self.client_id = client_id
        self.feature_indices = feature_indices
        self.max_bins = max_bins

        self.histogram_builder = HistogramBuilder(max_bins=max_bins)
        self.bin_edges_: Optional[List[np.ndarray]] = None

    def set_bin_edges(self, bin_edges: List[np.ndarray]) -> None:
        """
        Set pre-computed bin edges for this client's features.

        Args:
            bin_edges: List of bin edge arrays for each feature
        """
        self.bin_edges_ = bin_edges
        self.histogram_builder.bin_edges_ = bin_edges

    def compute_local_histograms(self,
                                 features: np.ndarray,
                                 gradients: np.ndarray,
                                 hessians: np.ndarray,
                                 sample_indices: np.ndarray) -> List[np.ndarray]:
        """
        Compute local histograms for all features.

        Args:
            features: Local feature matrix (n_samples, n_features)
            gradients: Gradient values (n_samples,)
            hessians: Hessian values (n_samples,)
            sample_indices: Indices of samples in current node

        Returns:
            List of histograms, one per local feature
        """
        histograms = []

        local_features = features[sample_indices, :]
        local_gradients = gradients[sample_indices]
        local_hessians = hessians[sample_indices]

        n_features = features.shape[1]

        if self.bin_edges_ is None:
            # Compute bin edges from features
            self.histogram_builder.fit_bins(features)

        for feat_idx in range(n_features):
            hist = self.histogram_builder.build_histogram(
                local_features[:, feat_idx],
                local_gradients,
                local_hessians,
                self.histogram_builder.bin_edges_[feat_idx]
            )
            histograms.append(hist)

        return histograms

    def receive_gradients(self,
                         gradients: np.ndarray,
                         hessians: np.ndarray) -> None:
        """
        Receive gradients and Hessians from label holder.

        Args:
            gradients: Gradient values (n_samples,)
            hessians: Hessian values (n_samples,)
        """
        self.gradients_ = gradients
        self.hessians_ = hessians

    def get_client_id(self) -> int:
        """Get client ID."""
        return self.client_id

    def get_feature_indices(self) -> List[int]:
        """Get feature indices this client holds."""
        return self.feature_indices


class ClientManager:
    """
    Manages multiple bank clients for federated training.
    """

    def __init__(self, n_clients: int, n_features: int):
        """
        Initialize client manager.

        Args:
            n_clients: Number of clients (banks)
            n_features: Total number of features
        """
        self.n_clients = n_clients
        self.n_features = n_features
        self.clients: List[BankClient] = []

    def partition_features(self, random_state: int = 42) -> List[List[int]]:
        """
        Partition features among clients.

        Args:
            random_state: Random seed

        Returns:
            List of feature indices for each client
        """
        np.random.seed(random_state)

        all_indices = np.arange(self.n_features)
        np.random.shuffle(all_indices)

        features_per_client = self.n_features // self.n_clients
        client_features = []

        for i in range(self.n_clients):
            start = i * features_per_client
            end = (i + 1) * features_per_client if i < self.n_clients - 1 else self.n_features
            client_features.append(all_indices[start:end].tolist())

        return client_features

    def create_clients(self,
                     feature_partition: Optional[List[List[int]]] = None,
                     max_bins: int = 256) -> List[BankClient]:
        """
        Create bank clients.

        Args:
            feature_partition: Pre-defined feature partition (if None, generates one)
            max_bins: Maximum histogram bins

        Returns:
            List of BankClient instances
        """
        if feature_partition is None:
            feature_partition = self.partition_features()

        self.clients = []

        for client_id, features in enumerate(feature_partition):
            client = BankClient(
                client_id=client_id,
                feature_indices=features,
                max_bins=max_bins
            )
            self.clients.append(client)

        return self.clients

    def get_client(self, client_id: int) -> BankClient:
        """Get client by ID."""
        return self.clients[client_id]

    def get_all_clients(self) -> List[BankClient]:
        """Get all clients."""
        return self.clients
