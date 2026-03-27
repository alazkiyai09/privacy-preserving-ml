"""
Core HE operations: key management, encryption/decryption, noise tracking.
"""

# Type aliases for TenSEAL objects
from typing import Any
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any
PublicKey = Any
CiphertextVector = Any

__all__ = [
    'SecretKey',
    'RelinKeys',
    'GaloisKeys',
    'PublicKey',
    'CiphertextVector',
]
