"""
TEE Module for HT2ML
====================

Provides Trusted Execution Environment functionality
for secure computation of non-linear operations.
"""

from src.encryption.hybrid.legacy.tee.enclave import (
    TEEEnclave,
    TEEContext,
    TEEAttestationReport,
    TEEState,
    create_tee_enclave,
    verify_attestation_report,
)

from src.encryption.hybrid.legacy.tee.operations import (
    TEEOperationEngine,
    TEEHandoffManager,
    TEEOperationResult,
    TEEOperationError,
    TEEComputationError,
    create_tee_engine,
    create_handoff_manager,
)

from src.encryption.hybrid.legacy.tee.attestation import (
    AttestationService,
    AttestationPolicy,
    AttestationRecord,
    AttestationStatus,
    create_attestation_service,
    create_default_policy,
)

from src.encryption.hybrid.legacy.tee.sealed_storage import (
    SealedStorage,
    SealedData,
    SealedModelBundle,
    SealingError,
    UnsealingError,
    create_sealed_storage,
    seal_and_save_model,
    load_and_unseal_model,
)

__all__ = [
    # Enclave
    'TEEEnclave',
    'TEEContext',
    'TEEAttestationReport',
    'TEEState',
    'create_tee_enclave',
    'verify_attestation_report',

    # Operations
    'TEEOperationEngine',
    'TEEHandoffManager',
    'TEEOperationResult',
    'TEEOperationError',
    'TEEComputationError',
    'create_tee_engine',
    'create_handoff_manager',

    # Attestation
    'AttestationService',
    'AttestationPolicy',
    'AttestationRecord',
    'AttestationStatus',
    'create_attestation_service',
    'create_default_policy',

    # Sealed Storage
    'SealedStorage',
    'SealedData',
    'SealedModelBundle',
    'SealingError',
    'UnsealingError',
    'create_sealed_storage',
    'seal_and_save_model',
    'load_and_unseal_model',
]
