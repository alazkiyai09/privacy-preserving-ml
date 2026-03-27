"""
Data Loader

Utilities for loading and preparing phishing email data.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, List
import os


class PhishingDataset(Dataset):
    """
    Dataset for phishing email classification.

    Features are pre-extracted (TF-IDF or similar).
    Labels: 0 = legitimate, 1 = phishing
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.

        Args:
            features: Feature array of shape (num_samples, num_features)
            labels: Label array of shape (num_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dummy_phishing_data(
    num_samples: int = 1000,
    num_features: int = 100,
    phishing_ratio: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dummy phishing email data for testing.

    Args:
        num_samples: Number of samples
        num_features: Number of features
        phishing_ratio: Ratio of phishing emails

    Returns:
        - Features array
        - Labels array
    """
    # Generate random features
    features = np.random.randn(num_samples, num_features)

    # Generate labels
    num_phishing = int(num_samples * phishing_ratio)
    labels = np.zeros(num_samples, dtype=int)
    labels[:num_phishing] = 1
    np.random.shuffle(labels)

    return features, labels


def load_client_data(
    client_id: int,
    total_clients: int,
    features: np.ndarray,
    labels: np.ndarray,
    iid: bool = True
) -> PhishingDataset:
    """
    Load data for a specific client.

    Args:
        client_id: Client identifier
        total_clients: Total number of clients
        features: Full feature array
        labels: Full label array
        iid: Whether to distribute IID or non-IID

    Returns:
        Dataset for this client
    """
    num_samples = len(features)
    samples_per_client = num_samples // total_clients

    if iid:
        # IID: Random split
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < total_clients - 1 else num_samples

        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
    else:
        # Non-IID: Sort by label then split
        sorted_indices = np.argsort(labels)
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < total_clients - 1 else num_samples

        client_indices = sorted_indices[start_idx:end_idx]
        client_features = features[client_indices]
        client_labels = labels[client_indices]

    return PhishingDataset(client_features, client_labels)


def create_client_loaders(
    total_clients: int,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    iid: bool = True,
    train_split: float = 0.8
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Create data loaders for all clients.

    Args:
        total_clients: Number of clients
        features: Feature array
        labels: Label array
        batch_size: Batch size
        iid: IID or non-IID distribution
        train_split: Training split ratio

    Returns:
        - List of client train loaders
        - Test loader (combined from all clients)
    """
    client_loaders = []

    # Create per-client datasets
    for client_id in range(total_clients):
        dataset = load_client_data(client_id, total_clients, features, labels, iid)

        # Split into train/val
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        client_loaders.append(train_loader)

    # Create combined test loader
    full_dataset = PhishingDataset(features, labels)
    test_size = int(len(full_dataset) * 0.2)
    train_test_split = len(full_dataset) - test_size

    _, test_dataset = random_split(
        full_dataset,
        [train_test_split, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return client_loaders, test_loader


def load_real_phishing_data(
    data_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real phishing email data.

    Args:
        data_path: Path to data file

    Returns:
        - Features array
        - Labels array

    Note:
        This is a placeholder. Implement actual data loading
        based on your data format.
    """
    # TODO: Implement actual data loading
    # For now, create dummy data
    return create_dummy_phishing_data(num_samples=10000, num_features=1000)
