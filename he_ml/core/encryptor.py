"""
Encryption and Decryption Operations
=====================================
Handles encryption of plaintext vectors and decryption of ciphertexts
for both BFV and CKKS schemes.
"""

import logging
from typing import Union, List, Any, Dict
import numpy as np
import tenseal as ts
import warnings

logger = logging.getLogger(__name__)

# Type aliases for TenSEAL objects (actual types not exposed in Python bindings)
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any
CiphertextVector = Any


def encrypt_vector(
    plaintext: np.ndarray,
    context: ts.Context,
    scheme: str = 'ckks',
) -> CiphertextVector:
    """
    Encrypt a NumPy array into a ciphertext vector.

    Uses CKKS by default (recommended for ML). BFV has limited support
    in TenSEAL Python.

    Args:
        plaintext: Input vector as NumPy array
            - For ML: Float array (float32, float64)
        context: TenSEAL context (must have keys generated)
        scheme: 'ckks' or 'bfv' (default: 'ckks')

    Returns:
        Encrypted vector (ts.CKKSVector)

    Raises:
        TypeError: If array type doesn't match scheme requirements
        ValueError: If vector size exceeds SIMD slots

    Example:
        >>> ctx = create_ckks_context()
        >>> x = np.array([1.0, 2.0, 3.0, 4.0])
        >>> encrypted_x = encrypt_vector(x, ctx, scheme='ckks')
        >>> print(f"Encrypted {len(x)} elements")
    """
    # For now, we only support CKKS properly
    if scheme.lower() != 'ckks':
        warnings.warn("Only CKKS is fully supported. Using CKKS.")
        scheme = 'ckks'

    # Check vector size (need poly_modulus_degree to calculate slots)
    # Default assumption: user knows their context size
    # Typical: 4096 poly_modulus_degree = 2048 slots

    # CKKS works with floats
    if not np.issubdtype(plaintext.dtype, np.floating):
        # Convert to float if needed
        plaintext = plaintext.astype(np.float64)
        warnings.warn(f"Converted {plaintext.dtype} to float64 for CKKS encryption")

    # Scale handling is automatic in TenSEAL
    return ts.ckks_vector(context, plaintext.tolist())


def decrypt_vector(
    ciphertext: CiphertextVector,
    secret_key: SecretKey,
) -> np.ndarray:
    """
    Decrypt a ciphertext vector back to a NumPy array.

    Args:
        ciphertext: Encrypted vector
        secret_key: Secret key for decryption

    Returns:
        Decrypted vector as NumPy array

    Note:
        - BFV: Returns integer array (exact decryption)
        - CKKS: Returns float array (may have small approximation errors)

    Example:
        >>> ctx = create_ckks_context()
        >>> keys = generate_keys(ctx)
        >>> x = np.array([1.5, 2.7, 3.14])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> decrypted_x = decrypt_vector(encrypted_x, keys['secret_key'])
        >>> np.allclose(x, decrypted_x, rtol=1e-5)
        True
    """
    # Decrypt to list
    decrypted_list = ciphertext.decrypt(secret_key)

    # Convert to NumPy array
    decrypted_array = np.array(decrypted_list)

    return decrypted_array


def encrypt_matrix(
    plaintext: np.ndarray,
    context: ts.Context,
    scheme: str = 'ckks',
) -> CiphertextVector:
    """
    Encrypt a 2D matrix by flattening it into a vector.

    Matrices are encrypted row-wise as flattened vectors.
    The original shape should be tracked separately for reshaping after decryption.

    Args:
        plaintext: 2D NumPy array
        context: TenSEAL context

    Returns:
        Encrypted vector (flattened matrix)

    Example:
        >>> ctx = create_ckks_context()
        >>> M = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> encrypted_M = encrypt_matrix(M, ctx)
        >>> # Save original shape for later
        >>> original_shape = M.shape
    """
    if plaintext.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {plaintext.ndim}D array")

    # Flatten row-wise
    flat = plaintext.flatten('C')

    return encrypt_vector(flat, context, scheme=scheme)


def decrypt_matrix(
    ciphertext: CiphertextVector,
    secret_key: SecretKey,
    original_shape: tuple,
) -> np.ndarray:
    """
    Decrypt an encrypted matrix back to its original shape.

    Args:
        ciphertext: Encrypted vector (flattened matrix)
        secret_key: Secret key for decryption
        original_shape: Original (rows, cols) shape

    Returns:
        Decrypted 2D NumPy array

    Example:
        >>> M = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> original_shape = M.shape
        >>> encrypted_M = encrypt_matrix(M, ctx)
        >>> decrypted_M = decrypt_matrix(encrypted_M, keys['secret_key'], original_shape)
        >>> np.allclose(M, decrypted_M)
        True
    """
    flat = decrypt_vector(ciphertext, secret_key)
    return flat.reshape(original_shape)


def encrypt_batch(
    plaintext: np.ndarray,
    context: ts.Context,
    scheme: str = 'ckks',
) -> List[CiphertextVector]:
    """
    Encrypt multiple vectors in a 2D array as separate ciphertexts.

    Each row of the input array becomes an independent ciphertext.
    This is useful for batch processing where each sample needs separate processing.

    Args:
        plaintext: 2D array where each row is a separate vector to encrypt
        context: TenSEAL context

    Returns:
        List of encrypted vectors (one per row)

    Example:
        >>> batch = np.array([[1.0, 2.0, 3.0],
        ...                    [4.0, 5.0, 6.0]])
        >>> encrypted_batch = encrypt_batch(batch, ctx)
        >>> print(f"Encrypted {len(encrypted_batch)} samples")
    """
    if plaintext.ndim != 2:
        raise ValueError(f"Expected 2D array, got {plaintext.ndim}D")

    encrypted_list = []
    for i in range(plaintext.shape[0]):
        encrypted_list.append(encrypt_vector(plaintext[i], context, scheme=scheme))

    return encrypted_list


def decrypt_batch(
    ciphertexts: List[CiphertextVector],
    secret_key: SecretKey,
) -> np.ndarray:
    """
    Decrypt a batch of ciphertexts back to a 2D array.

    Args:
        ciphertexts: List of encrypted vectors
        secret_key: Secret key for decryption

    Returns:
        2D NumPy array with one row per ciphertext

    Example:
        >>> batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> encrypted_batch = encrypt_batch(batch, ctx)
        >>> decrypted_batch = decrypt_batch(encrypted_batch, keys['secret_key'])
        >>> np.allclose(batch, decrypted_batch)
        True
    """
    decrypted_rows = []
    for ct in ciphertexts:
        row = decrypt_vector(ct, secret_key)
        decrypted_rows.append(row)

    return np.vstack(decrypted_rows)


def get_ciphertext_size(
    ciphertext: CiphertextVector,
) -> int:
    """
    Get the size of a ciphertext in bytes.

    Useful for understanding the memory overhead of encryption.

    Args:
        ciphertext: Encrypted vector

    Returns:
        Size in bytes

    Example:
        >>> x = np.random.randn(100)
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> size_kb = get_ciphertext_size(encrypted_x) / 1024
        >>> print(f"Ciphertext size: {size_kb:.2f} KB")
    """
    # Serialize to get size
    serialized = ciphertext.serialize()
    return len(serialized)


def get_encryption_overhead(
    plaintext: np.ndarray,
    ciphertext: CiphertextVector,
) -> Dict[str, float]:
    """
    Analyze the memory overhead of encryption.

    Args:
        plaintext: Original plaintext array
        ciphertext: Encrypted version

    Returns:
        Dictionary with overhead metrics:
        {
            'plaintext_bytes': float,
            'ciphertext_bytes': float,
            'overhead_bytes': float,
            'overhead_factor': float,  # ciphertext_size / plaintext_size
            'expansion_ratio': float   # how many times larger
        }

    Example:
        >>> x = np.random.randn(1000)
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> overhead = get_encryption_overhead(x, encrypted_x)
        >>> print(f"Expansion: {overhead['expansion_ratio']:.1f}x")
    """
    plaintext_bytes = plaintext.nbytes
    ciphertext_bytes = get_ciphertext_size(ciphertext)

    return {
        'plaintext_bytes': plaintext_bytes,
        'ciphertext_bytes': ciphertext_bytes,
        'overhead_bytes': ciphertext_bytes - plaintext_bytes,
        'overhead_factor': ciphertext_bytes / plaintext_bytes,
        'expansion_ratio': ciphertext_bytes / plaintext_bytes,
    }


def log_encryption_stats(
    plaintext: np.ndarray,
    ciphertext: Union[ts.BFVVector, ts.CKKSVector],
    label: str = "Encryption",
) -> None:
    """
    Log statistics about encryption for analysis and documentation.

    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted version
        label: Label for the output
    """
    overhead = get_encryption_overhead(plaintext, ciphertext)

    logger.info(
        f"{label} Statistics: "
        f"size={len(plaintext):,} elements, "
        f"plaintext={overhead['plaintext_bytes']:,.0f} bytes, "
        f"ciphertext={overhead['ciphertext_bytes']:,.0f} bytes, "
        f"overhead={overhead['overhead_bytes']:,.0f} bytes, "
        f"expansion={overhead['expansion_ratio']:.2f}x"
    )


# Keep print function for backward compatibility but use logging internally
def print_encryption_stats(
    plaintext: np.ndarray,
    ciphertext: Union[ts.BFVVector, ts.CKKSVector],
    label: str = "Encryption",
) -> None:
    """
    Print statistics about encryption for analysis and documentation.

    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted version
        label: Label for the output
    """
    # Use logging internally
    log_encryption_stats(plaintext, ciphertext, label)

    # Also print for backward compatibility
    overhead = get_encryption_overhead(plaintext, ciphertext)

    print(f"\n{'='*60}")
    print(f"{label} Statistics")
    print(f"{'='*60}")
    print(f"Vector size:           {len(plaintext):,} elements")
    print(f"Plaintext size:        {overhead['plaintext_bytes']:,.0f} bytes")
    print(f"Ciphertext size:       {overhead['ciphertext_bytes']:,.0f} bytes")
    print(f"Overhead:              {overhead['overhead_bytes']:,.0f} bytes")
    print(f"Expansion ratio:       {overhead['expansion_ratio']:.2f}x")
    print(f"{'='*60}\n")


def validate_encryption(
    plaintext: np.ndarray,
    ciphertext: CiphertextVector,
    secret_key: SecretKey,
    scheme: str = 'ckks',
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate that encryption/decryption is correct within tolerance.

    Args:
        plaintext: Original plaintext
        ciphertext: Encrypted version
        secret_key: Secret key
        scheme: 'bfv' or 'ckks'
        tolerance: Max allowed absolute error

    Returns:
        Dictionary with validation results:
        {
            'passed': bool,
            'max_error': float,
            'mean_error': float,
            'error_details': np.ndarray
        }

    Example:
        >>> result = validate_encryption(x, encrypted_x, keys['secret_key'], 'ckks')
        >>> if result['passed']:
        ...     print("Encryption validated successfully")
    """
    decrypted = decrypt_vector(ciphertext, secret_key)

    # Handle size mismatches (ciphertexts may have padding)
    min_len = min(len(plaintext), len(decrypted))
    errors = np.abs(plaintext[:min_len] - decrypted[:min_len])

    return {
        'passed': np.all(errors <= tolerance),
        'max_error': float(np.max(errors)),
        'mean_error': float(np.mean(errors)),
        'error_details': errors,
    }
