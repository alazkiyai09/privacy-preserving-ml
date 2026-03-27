"""
Data loading utilities for phishing detection.

Handles loading, preprocessing, and splitting data for federated training.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_phishing_data(n_samples: int = 10000,
                       n_features: int = 20,
                       n_informative: int = 15,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic phishing detection dataset.

    In production, this would load real phishing data.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        random_state: Random seed

    Returns:
        Tuple of (X, y) features and labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.01,  # 1% label noise
        random_state=random_state
    )

    return X, y


def partition_data_vertical(X: np.ndarray,
                           y: np.ndarray,
                           n_parties: int,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Partition data vertically for federated learning.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        n_parties: Number of parties
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        Tuple of (X_train_dict, X_test_dict, y_train, y_test)
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Partition features among parties
    n_features = X.shape[1]
    features_per_party = n_features // n_parties

    X_train_dict = {}
    X_test_dict = {}

    for i in range(n_parties):
        start = i * features_per_party
        end = (i + 1) * features_per_party if i < n_parties - 1 else n_features

        X_train_dict[i] = X_train[:, start:end]
        X_test_dict[i] = X_test[:, start:end]

    return X_train_dict, X_test_dict, y_train, y_test


def create_realistic_phishing_data(n_samples: int = 5000,
                                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create more realistic phishing dataset with feature groups.

    Simulates different feature types:
    - Transaction features (amount, frequency, timing)
    - Email content features (keywords, sender info)
    - URL features (length, domain, suspicious patterns)

    Args:
        n_samples: Number of samples
        random_state: Random seed

    Returns:
        Tuple of (X, y)
    """
    np.random.seed(random_state)

    # Feature groups
    n_transaction = 5
    n_email = 5
    n_url = 5
    n_other = 5
    n_features = n_transaction + n_email + n_url + n_other

    # Generate base features
    X = np.random.randn(n_samples, n_features)

    # Generate labels with some signal
    # Create composite risk score
    risk_score = (
        X[:, 0] * 0.3 +  # Transaction amount
        X[:, 5] * 0.2 +  # Email keywords
        X[:, 10] * 0.3 +  # URL patterns
        X[:, 15] * 0.2    # Other indicators
    )

    # Add some class-specific signal
    phishing_samples = n_samples // 2
    risk_score[:phishing_samples] += np.random.randn(phishing_samples) * 0.5
    risk_score[phishing_samples:] -= np.random.randn(n_samples - phishing_samples) * 0.5

    # Convert to binary labels
    y = (risk_score > 0).astype(int)

    return X, y


def add_missing_features(X_dict: Dict[int, np.ndarray],
                        missing_ratio: float = 0.1,
                        random_state: int = 42) -> Dict[int, np.ndarray]:
    """
    Simulate missing features (some features unavailable to some parties).

    Args:
        X_dict: Feature dictionary
        missing_ratio: Fraction of features to mark as missing
        random_state: Random seed

    Returns:
        Modified feature dictionary with NaN values
    """
    np.random.seed(random_state)

    X_dict_missing = {}

    for party_id, X in X_dict.items():
        X_missing = X.copy()
        n_features = X.shape[1]
        n_missing = int(n_features * missing_ratio)

        # Randomly select features to set as missing
        missing_indices = np.random.choice(n_features, n_missing, replace=False)

        # Set to NaN
        for idx in missing_indices:
            mask = np.random.rand(len(X)) < 0.1  # 10% of samples have missing this feature
            X_missing[mask, idx] = np.nan

        X_dict_missing[party_id] = X_missing

    return X_dict_missing
