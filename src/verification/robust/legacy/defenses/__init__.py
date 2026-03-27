"""
Defense Implementations for Federated Phishing Detection

This module implements various defense mechanisms against attacks:
- Byzantine-Robust Aggregation
- Anomaly Detection
- Reputation System
- Robust Training
"""

from .byzantine_aggregation import (
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator
)
from .anomaly_detection import (
    ZScoreDetector,
    ClusteringDetector,
    CombinedAnomalyDetector
)
from .reputation_system import ClientReputationSystem
from .robust_training import PGDAdversarialTraining, TRADESTraining

__all__ = [
    "KrumAggregator",
    "MultiKrumAggregator",
    "TrimmedMeanAggregator",
    "ZScoreDetector",
    "ClusteringDetector",
    "CombinedAnomalyDetector",
    "ClientReputationSystem",
    "PGDAdversarialTraining",
    "TRADESTraining",
]
