"""
Inference Module for HT2ML
===========================

Provides inference engines for hybrid HE/TEE computation
and baseline implementations (HE-only, TEE-only).
"""

from src.encryption.hybrid.legacy.inference.hybrid_engine import (
    HybridInferenceEngine,
    HybridInferenceError,
    HybridInferenceStats,
    create_hybrid_engine,
    run_single_inference,
)

from src.encryption.hybrid.legacy.inference.he_only_engine import (
    HEOnlyInferenceEngine,
    HEOnlyEngineError,
    HEApproximation,
    create_he_only_engine,
)

from src.encryption.hybrid.legacy.inference.tee_only_engine import (
    TEEOnlyInferenceEngine,
    TEEOnlyEngineError,
    create_tee_only_engine,
    run_tee_only_inference,
)

__all__ = [
    # Hybrid
    'HybridInferenceEngine',
    'HybridInferenceError',
    'HybridInferenceStats',
    'create_hybrid_engine',
    'run_single_inference',

    # HE-only baseline
    'HEOnlyInferenceEngine',
    'HEOnlyEngineError',
    'HEApproximation',
    'create_he_only_engine',

    # TEE-only baseline
    'TEEOnlyInferenceEngine',
    'TEEOnlyEngineError',
    'create_tee_only_engine',
    'run_tee_only_inference',
]
