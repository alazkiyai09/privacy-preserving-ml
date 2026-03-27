"""
Secure protocols for privacy-preserving GBDT.

This module contains protocols for:
- Private Set Intersection (PSI) for sample alignment
- Secure split finding across parties
- Privacy-preserving prediction
"""

from .psi import PSIProtocol, SampleAlignment, simulate_vertical_partition
from .split_finding import (
    SecureSplitFinding,
    ApproximateSplitFinding,
    CandidateSplit,
    compute_split_gains_histograms
)
from .prediction import (
    SecurePrediction,
    LocalPartyPrediction,
    simulate_feature_partition
)

__all__ = [
    # PSI
    'PSIProtocol',
    'SampleAlignment',
    'simulate_vertical_partition',

    # Split Finding
    'SecureSplitFinding',
    'ApproximateSplitFinding',
    'CandidateSplit',
    'compute_split_gains_histograms',

    # Prediction
    'SecurePrediction',
    'LocalPartyPrediction',
    'simulate_feature_partition',
]
