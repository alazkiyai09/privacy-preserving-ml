"""
Cryptographic primitives for privacy-preserving GBDT.

This module contains:
- Additive secret sharing for secure multi-party computation
- Differential privacy mechanisms (Laplace, Gaussian)
- Secure aggregation protocols
"""

from .secret_sharing import (
    SecretSharing,
    share,
    reconstruct,
    add_shares
)

from .dp_mechanisms import (
    DifferentialPrivacy,
    HistogramDP,
    TreeDP,
    laplace_noise,
    gaussian_noise
)

from .secure_aggregation import (
    SecureAggregator,
    DPSecureAggregator,
    GradientAggregator,
    compute_privacy_cost
)

__all__ = [
    # Secret Sharing
    'SecretSharing',
    'share',
    'reconstruct',
    'add_shares',

    # Differential Privacy
    'DifferentialPrivacy',
    'HistogramDP',
    'TreeDP',
    'laplace_noise',
    'gaussian_noise',

    # Secure Aggregation
    'SecureAggregator',
    'DPSecureAggregator',
    'GradientAggregator',
    'compute_privacy_cost',
]
