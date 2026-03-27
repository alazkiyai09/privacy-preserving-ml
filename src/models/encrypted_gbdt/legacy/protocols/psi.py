"""
Private Set Intersection (PSI) for sample alignment in vertical federated learning.

In vertical FL, different parties have different features for the same set of samples.
PSI allows parties to find their common sample IDs without revealing which IDs
they have that others don't.
"""

import numpy as np
from typing import List, Set, Dict, Tuple
import hashlib


class PSIProtocol:
    """
    Private Set Intersection protocol.

    Implements simple hashing-based PSI for educational purposes.
    In production, use cryptographic protocols like ECDH, RSA-PSI, or KKRT.
    """

    def __init__(self, hash_function: str = 'sha256'):
        """
        Initialize PSI protocol.

        Args:
            hash_function: Hash function to use ('sha256', 'md5', etc.)
        """
        self.hash_function = hash_function

    def hash_id(self, sample_id: int) -> str:
        """
        Hash a sample ID.

        Args:
            sample_id: Sample ID to hash

        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.new(self.hash_function)
        hasher.update(str(sample_id).encode())
        return hasher.hexdigest()

    def compute_hashed_ids(self, sample_ids: Set[int]) -> Set[str]:
        """
        Compute hashes for a set of sample IDs.

        Args:
            sample_ids: Set of sample IDs

        Returns:
            Set of hashed IDs
        """
        return {self.hash_id(sid) for sid in sample_ids}

    def server_intersection(self,
                           client_ids: Set[str],
                           server_ids: Set[int]) -> Set[str]:
        """
        Server-side PSI computation (simplified protocol).

        The server hashes its IDs and the client sends its hashed IDs.
        The intersection of hashed IDs is returned.

        Args:
            client_ids: Hashed sample IDs from client
            server_ids: Raw sample IDs from server

        Returns:
            Intersection (hashed IDs that both parties have)
        """
        server_hashed = self.compute_hashed_ids(server_ids)
        return client_ids.intersection(server_hashed)

    def client_intersection(self,
                           client_ids: Set[int],
                           server_hashed_ids: Set[str]) -> Set[str]:
        """
        Client-side PSI computation.

        Args:
            client_ids: Raw sample IDs from client
            server_hashed_ids: Hashed sample IDs from server

        Returns:
            Intersection (hashed IDs)
        """
        client_hashed = self.compute_hashed_ids(client_ids)
        return client_hashed.intersection(server_hashed_ids)

    def multi_party_psi(self,
                       party_ids: List[Set[int]]) -> Set[str]:
        """
        Multi-party PSI to find common sample IDs across all parties.

        Args:
            party_ids: List of sample ID sets from each party

        Returns:
            Hashed intersection of all parties' sample IDs
        """
        if not party_ids:
            return set()

        # First party hashes their IDs
        intersection = self.compute_hashed_ids(party_ids[0])

        # Each subsequent party hashes and intersects
        for i in range(1, len(party_ids)):
            party_hashed = self.compute_hashed_ids(party_ids[i])
            intersection = intersection.intersection(party_hashed)

        return intersection

    def align_samples(self,
                     party_features: List[np.ndarray],
                     party_ids: List[Set[int]]) -> Tuple[List[np.ndarray], Set[str]]:
        """
        Align samples across parties using PSI.

        Args:
            party_features: List of feature matrices from each party
            party_ids: List of sample ID sets from each party

        Returns:
            Tuple of (aligned_features, common_sample_ids)
        """
        if len(party_features) != len(party_ids):
            raise ValueError("Number of feature sets must match number of ID sets")

        # Find common sample IDs
        common_ids = self.multi_party_psi(party_ids)

        if not common_ids:
            return party_features, common_ids

        # Create mapping from hashed ID to index for each party
        aligned_features = []

        for party_id, features in enumerate(party_features):
            # Create ID to index mapping
            id_to_idx = {self.hash_id(sid): idx for idx, sid in enumerate(party_ids[party_id])}

            # Select rows corresponding to common IDs
            common_indices = [id_to_idx[hid] for hid in common_ids if hid in id_to_idx]

            if common_indices:
                aligned_features.append(features[common_indices, :])
            else:
                aligned_features.append(features)

        return aligned_features, common_ids


class SampleAlignment:
    """
    Sample alignment utilities for vertical federated learning.

    In vertical FL, parties have different features for the same samples.
    We need to align samples to ensure we're training on the same data points.
    """

    def __init__(self):
        """Initialize sample alignment."""
        self.psi = PSIProtocol()

    def align_parties(self,
                     features_list: List[np.ndarray],
                     sample_ids_list: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Align samples across all parties.

        Args:
            features_list: List of feature matrices (n_samples_i, n_features_i)
            sample_ids_list: List of sample ID arrays (n_samples_i,)

        Returns:
            Tuple of (aligned_features, common_sample_ids)
        """
        if len(features_list) != len(sample_ids_list):
            raise ValueError("features_list and sample_ids_list must have same length")

        # Convert to sets
        id_sets = [set(ids) for ids in sample_ids_list]

        # Find intersection
        common_ids = self.psi.multi_party_psi(id_sets)

        # Create ID to index mapping for each party
        aligned_features = []

        for features, ids in zip(features_list, sample_ids_list):
            id_to_idx = {self.psi.hash_id(sid): idx for idx, sid in enumerate(ids)}
            common_indices = [id_to_idx[hid] for hid in common_ids if hid in id_to_idx]
            aligned_features.append(features[common_indices, :])

        # Convert hashed IDs back to original IDs
        # (in practice, you'd keep the mapping)
        common_sample_ids = np.array(sorted(common_ids))

        return aligned_features, common_sample_ids

    def compute_overlap_stats(self,
                             sample_ids_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics about sample overlap.

        Args:
            sample_ids_list: List of sample ID arrays

        Returns:
            Dictionary with overlap statistics
        """
        n_parties = len(sample_ids_list)
        id_sets = [set(ids) for ids in sample_ids_list]

        stats = {
            'n_parties': n_parties,
            'party_sizes': [len(ids) for ids in id_sets],
        }

        # Compute all pairwise overlaps
        pairwise_overlaps = []
        for i in range(n_parties):
            for j in range(i + 1, n_parties):
                overlap = len(id_sets[i].intersection(id_sets[j]))
                pairwise_overlaps.append(overlap)

        stats['pairwise_overlaps'] = pairwise_overlaps
        stats['avg_pairwise_overlap'] = np.mean(pairwise_overlaps) if pairwise_overlaps else 0

        # Compute full intersection
        full_intersection = self.psi.multi_party_psi(id_sets)
        stats['full_intersection_size'] = len(full_intersection)

        return stats


def simulate_vertical_partition(X: np.ndarray,
                                n_parties: int,
                                overlap_ratio: float = 0.8,
                                random_state: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Simulate vertical data partitioning for testing.

    Args:
        X: Full feature matrix
        n_parties: Number of parties to split among
        overlap_ratio: Fraction of samples to have in all parties
        random_state: Random seed

    Returns:
        Tuple of (partitioned_features, partitioned_ids)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Split features among parties
    features_per_party = n_features // n_parties
    partitioned_features = []
    partitioned_ids = []

    for i in range(n_parties):
        start_feat = i * features_per_party
        end_feat = (i + 1) * features_per_party if i < n_parties - 1 else n_features

        # Decide which samples this party has
        n_common = int(n_samples * overlap_ratio)
        common_indices = np.random.choice(n_samples, n_common, replace=False)
        party_indices = np.concatenate([common_indices, np.setdiff1d(np.arange(n_samples), common_indices)])
        np.random.shuffle(party_indices)

        partitioned_features.append(X[party_indices, start_feat:end_feat])
        partitioned_ids.append(party_indices)

    return partitioned_features, partitioned_ids
