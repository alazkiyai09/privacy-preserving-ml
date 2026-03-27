"""
TEE ML: Trusted Execution Environment for Machine Learning
===========================================================

A simulation framework for privacy-preserving ML using TEEs.
"""

__version__ = "0.1.0"
__author__ = "Privacy-Preserving ML Researcher"

from tee_ml.core.enclave import Enclave, EnclaveSession, SecureMemory
from tee_ml.core.attestation import AttestationReport, AttestationService
from tee_ml.core.sealed_storage import SealedStorage

__all__ = [
    "Enclave",
    "EnclaveSession",
    "SecureMemory",
    "AttestationReport",
    "AttestationService",
    "SealedStorage",
]
