"""
Anomaly Detection for Federated Learning

Detects anomalous gradients from clients that may indicate attacks.

Methods:
1. Z-Score Detection: Flag gradients >k standard deviations from mean
2. Clustering Detection: Use DBSCAN to find outliers
3. Distance-Based Detection: Flag gradients far from centroid
4. Combined Detection: Ensemble of multiple methods

Application:
- Catches malicious gradients that pass ZK verification
- Complements Byzantine aggregation
- Works with reputation system to track repeated anomalies
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


def flatten_gradients(gradients: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Flatten list of gradient arrays into 1D vectors."""
    return [np.concatenate([layer.flatten() for layer in grad]) for grad in gradients]


class ZScoreDetector:
    """
    Z-Score Anomaly Detection.

    Flags gradients that are more than k standard deviations from the mean.

    Algorithm:
    1. Compute mean and std of gradients
    2. For each gradient, compute z-score
    3. Flag if z-score > threshold

    Simple and fast but assumes normal distribution.
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-Score detector.

        Args:
            threshold: Z-score threshold for anomaly (default: 3.0)
        """
        self.threshold = threshold

    def detect(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[bool]:
        """
        Detect anomalies using z-score.

        Args:
            gradients: List of gradients (each is list of arrays)

        Returns:
            List of booleans (True = anomalous)
        """
        if len(gradients) < 3:
            # Need at least 3 gradients to compute std
            return [False] * len(gradients)

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        # Compute mean and std
        mean = np.mean(all_gradients_array, axis=0)
        std = np.std(all_gradients_array, axis=0) + 1e-8  # Avoid division by zero

        # Compute z-scores for each gradient
        z_scores = []
        for flat_grad in flat_gradients:
            # Absolute z-score
            z = np.abs((flat_grad - mean) / std)
            # Average z-score across all dimensions
            avg_z = np.mean(z)
            z_scores.append(avg_z)

        # Flag anomalies
        anomalies = [z > self.threshold for z in z_scores]

        return anomalies

    def get_z_scores(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[float]:
        """Get z-scores for gradients."""
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        mean = np.mean(all_gradients_array, axis=0)
        std = np.std(all_gradients_array, axis=0) + 1e-8

        z_scores = []
        for flat_grad in flat_gradients:
            z = np.abs((flat_grad - mean) / std)
            z_scores.append(float(np.mean(z)))

        return z_scores


class ClusteringDetector:
    """
    Clustering-Based Anomaly Detection using DBSCAN.

    Flags gradients that don't belong to any cluster (outliers).

    Algorithm:
    1. Use DBSCAN to cluster gradients
    2. Gradients with label -1 are outliers (anomalies)

    More robust than z-score for non-Gaussian distributions.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        """
        Initialize clustering detector.

        Args:
            eps: Maximum distance between samples in same cluster
            min_samples: Minimum samples in a cluster
        """
        self.eps = eps
        self.min_samples = min_samples

    def detect(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[bool]:
        """
        Detect anomalies using DBSCAN clustering.

        Args:
            gradients: List of gradients

        Returns:
            List of booleans (True = anomalous)
        """
        if len(gradients) < self.min_samples:
            # Not enough samples to form clusters
            return [False] * len(gradients)

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        # Normalize gradients for better clustering
        # (compute norms and normalize)
        norms = np.linalg.norm(all_gradients_array, axis=1, keepdims=True) + 1e-8
        normalized = all_gradients_array / norms

        # Apply DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(normalized)

        # Outliers have label -1
        anomalies = [label == -1 for label in labels]

        return anomalies

    def get_cluster_info(
        self,
        gradients: List[List[np.ndarray]]
    ) -> Dict[str, Any]:
        """Get clustering information."""
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        norms = np.linalg.norm(all_gradients_array, axis=1, keepdims=True) + 1e-8
        normalized = all_gradients_array / norms

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(normalized)

        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_outliers = list(labels).count(-1)

        return {
            "num_clusters": num_clusters,
            "num_outliers": num_outliers,
            "labels": [int(label) for label in labels]
        }


class DistanceBasedDetector:
    """
    Distance-Based Anomaly Detection.

    Flags gradients that are far from the centroid.

    Algorithm:
    1. Compute centroid of all gradients
    2. Flag gradients with distance > threshold * mean_distance
    """

    def __init__(self, threshold_multiplier: float = 2.0):
        """
        Initialize distance-based detector.

        Args:
            threshold_multiplier: Multiplier for mean distance threshold
        """
        self.threshold_multiplier = threshold_multiplier

    def detect(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[bool]:
        """
        Detect anomalies using distance from centroid.

        Args:
            gradients: List of gradients

        Returns:
            List of booleans (True = anomalous)
        """
        if len(gradients) < 3:
            return [False] * len(gradients)

        # Flatten gradients
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        # Compute centroid
        centroid = np.mean(all_gradients_array, axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(all_gradients_array - centroid, axis=1)

        # Compute threshold
        mean_distance = np.mean(distances)
        threshold = mean_distance * self.threshold_multiplier

        # Flag anomalies
        anomalies = [dist > threshold for dist in distances]

        return anomalies

    def get_distances(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[float]:
        """Get distances from centroid."""
        flat_gradients = flatten_gradients(gradients)
        all_gradients_array = np.array(flat_gradients)

        centroid = np.mean(all_gradients_array, axis=0)
        distances = np.linalg.norm(all_gradients_array - centroid, axis=1)

        return [float(dist) for dist in distances]


class CombinedAnomalyDetector:
    """
    Combined Anomaly Detection using multiple methods.

    Combines z-score, clustering, and distance-based detection.
    Uses voting to determine final anomaly flags.

    Voting Strategies:
    - majority: Flag if majority of methods flag as anomaly
    - unanimous: Flag only if all methods agree
    - any: Flag if any method flags as anomaly
    """

    def __init__(
        self,
        methods: List[str] = ["zscore", "clustering", "distance"],
        voting: str = "majority",
        zscore_threshold: float = 3.0,
        clustering_eps: float = 0.5,
        distance_threshold: float = 2.0
    ):
        """
        Initialize combined detector.

        Args:
            methods: List of methods to use
            voting: Voting strategy ("majority", "unanimous", "any")
            zscore_threshold: Threshold for z-score method
            clustering_eps: Eps for clustering method
            distance_threshold: Threshold multiplier for distance method
        """
        self.methods = methods
        self.voting = voting

        # Initialize detectors
        self.detectors = {}

        if "zscore" in methods:
            self.detectors["zscore"] = ZScoreDetector(threshold=zscore_threshold)

        if "clustering" in methods:
            self.detectors["clustering"] = ClusteringDetector(
                eps=clustering_eps,
                min_samples=3
            )

        if "distance" in methods:
            self.detectors["distance"] = DistanceBasedDetector(
                threshold_multiplier=distance_threshold
            )

    def detect(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[bool]:
        """
        Detect anomalies using combined methods.

        Args:
            gradients: List of gradients

        Returns:
            List of booleans (True = anomalous)
        """
        if not self.detectors:
            return [False] * len(gradients)

        # Get detections from all methods
        all_detections = []
        for method_name, detector in self.detectors.items():
            detections = detector.detect(gradients)
            all_detections.append(detections)

        # Combine detections using voting
        num_methods = len(all_detections)
        num_gradients = len(gradients)

        combined_anomalies = []

        for grad_idx in range(num_gradients):
            # Count how many methods flag this gradient as anomalous
            votes = sum([detections[grad_idx] for detections in all_detections])

            if self.voting == "majority":
                # Flag if majority agree
                is_anomaly = votes > num_methods / 2
            elif self.voting == "unanimous":
                # Flag only if all agree
                is_anomaly = votes == num_methods
            elif self.voting == "any":
                # Flag if any method flags
                is_anomaly = votes > 0
            else:
                raise ValueError(f"Unknown voting strategy: {self.voting}")

            combined_anomalies.append(is_anomaly)

        return combined_anomalies

    def get_method_votes(
        self,
        gradients: List[List[np.ndarray]]
    ) -> Dict[str, List[bool]]:
        """Get anomaly flags from each method."""
        votes = {}
        for method_name, detector in self.detectors.items():
            votes[method_name] = detector.detect(gradients)
        return votes


# Example usage
if __name__ == "__main__":
    print("Anomaly Detection Demonstration")
    print("=" * 60)

    # Create synthetic gradients
    # 8 honest gradients
    honest_gradients = []
    for i in range(8):
        grad = [np.random.randn(5, 10) * 0.1 for _ in range(3)]
        honest_gradients.append(grad)

    # 2 malicious gradients (outliers)
    malicious_gradients = []
    for i in range(2):
        grad = [np.random.randn(5, 10) * 5.0 for _ in range(3)]
        malicious_gradients.append(grad)

    all_gradients = honest_gradients + malicious_gradients

    print(f"Total gradients: {len(all_gradients)}")
    print(f"Malicious: 2 (indices 8, 9)")

    # Test Z-Score
    print("\n--- Z-Score Detection ---")
    zscore_detector = ZScoreDetector(threshold=2.0)
    zscore_anomalies = zscore_detector.detect(all_gradients)
    zscore_z_scores = zscore_detector.get_z_scores(all_gradients)

    print(f"Z-scores: {[f'{z:.2f}' for z in zscore_z_scores]}")
    print(f"Anomalies detected: {sum(zscore_anomalies)}")
    print(f"Which indices: {[i for i, a in enumerate(zscore_anomalies) if a]}")

    # Test Clustering
    print("\n--- Clustering Detection (DBSCAN) ---")
    cluster_detector = ClusteringDetector(eps=0.5, min_samples=3)
    cluster_anomalies = cluster_detector.detect(all_gradients)
    cluster_info = cluster_detector.get_cluster_info(all_gradients)

    print(f"Clusters found: {cluster_info['num_clusters']}")
    print(f"Outliers detected: {cluster_info['num_outliers']}")
    print(f"Anomaly indices: {[i for i, a in enumerate(cluster_anomalies) if a]}")

    # Test Distance-Based
    print("\n--- Distance-Based Detection ---")
    distance_detector = DistanceBasedDetector(threshold_multiplier=2.0)
    distance_anomalies = distance_detector.detect(all_gradients)
    distance_dists = distance_detector.get_distances(all_gradients)

    print(f"Distances from centroid: {[f'{d:.2f}' for d in distance_dists]}")
    print(f"Anomalies detected: {sum(distance_anomalies)}")
    print(f"Which indices: {[i for i, a in enumerate(distance_anomalies) if a]}")

    # Test Combined
    print("\n--- Combined Detection (Majority Voting) ---")
    combined_detector = CombinedAnomalyDetector(
        methods=["zscore", "clustering", "distance"],
        voting="majority"
    )
    combined_anomalies = combined_detector.detect(all_gradients)
    method_votes = combined_detector.get_method_votes(all_gradients)

    print(f"Method votes:")
    for method, votes in method_votes.items():
        print(f"  {method}: {[i for i, v in enumerate(votes) if v]}")
    print(f"Combined (majority): {[i for i, a in enumerate(combined_anomalies) if a]}")
    print(f"Malicious caught: {sum([combined_anomalies[i] for i in [8, 9]])} / 2")
