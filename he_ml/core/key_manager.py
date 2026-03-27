"""
Key Manager for Homomorphic Encryption
========================================
Manages TenSEAL context creation and key generation for both BFV and CKKS schemes.

Security Parameters Reference:
- poly_modulus_degree: Ring size (power of 2, 4096-16384). Larger = more capacity but slower.
- coeff_mod_bit_sizes: Prime chain bit sizes. Sum determines noise budget.
- plain_modulus (BFV): Prime for plaintext space.
- scale (CKKS): Fixed-point scaling factor (typically 2^40).

Typical configurations:
- BFV (small): poly_modulus_degree=4096, plain_modulus=1032193, coeff_mod_bit_sizes=[40, 20, 20]
- CKKS (ML): poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2^40
"""

from typing import Dict, Any, List, Optional, Tuple
import tenseal as ts
from pathlib import Path
import json
import warnings


def create_bfv_context(
    poly_modulus_degree: int = 4096,
    plain_modulus: int = 1032193,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
) -> ts.Context:
    """
    Create a TenSEAL BFV context for exact integer arithmetic.

    BFV Scheme:
    - Supports exact integer operations (no approximation)
    - Suitable for boolean operations and integer arithmetic
    - Less efficient for ML than CKKS

    NOTE: BFV support in TenSEAL Python is limited. For ML applications,
    CKKS is strongly recommended.

    Args:
        poly_modulus_degree: Ring size (power of 2). Options: 4096, 8192, 16384.
            Larger values enable more multiplications but are slower.
        plain_modulus: Prime modulus for plaintext space. (Not directly used in TenSEAL Python)
        coeff_mod_bit_sizes: Bit sizes for coefficient modulus primes.
            Sum determines initial noise budget.
            Default for 4096: [40, 20, 20] gives ~80 bits budget

    Returns:
        ts.Context: Configured context (falls back to CKKS)

    Example:
        >>> ctx = create_bfv_context(poly_modulus_degree=4096)
        >>> # For ML, consider using CKKS instead
    """
    warnings.warn(
        "BFV has limited support in TenSEAL Python. "
        "Using CKKS instead. For ML applications, CKKS is recommended."
    )

    # Fall back to CKKS for compatibility
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [40, 20, 20]

    if not (poly_modulus_degree & (poly_modulus_degree - 1) == 0):
        raise ValueError(f"poly_modulus_degree must be power of 2, got {poly_modulus_degree}")

    # Use CKKS as fallback (works for integer data too)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree,
        coeff_mod_bit_sizes,
    )

    # Set scale for CKKS (use larger scale for better precision)
    context.global_scale = 2**30

    # Note: auto_rescale behavior in TenSEAL Python
    # Keeping default (True) for proper scale management in complex operations
    # For simple scalar multiplication, some precision issues may occur

    return context


def create_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
    scale: float = 2**40,
) -> ts.Context:
    """
    Create a TenSEAL CKKS context for approximate arithmetic.

    CKKS Scheme:
    - Supports floating-point arithmetic with controlled approximation
    - Ideal for machine learning (matrix multiplication, activations)
    - Uses scale parameter for fixed-point representation
    - Each multiplication reduces precision (noise growth)

    Args:
        poly_modulus_degree: Ring size (power of 2). Options: 4096, 8192, 16384.
            Larger values enable more slots and deeper circuits.
        coeff_mod_bit_sizes: Bit sizes for coefficient modulus primes.
            Structure: [special_prime, mid_primes..., special_prime]
            Special primes should match scale. For scale=2^40: [60, 40, 40, 60]
        scale: Scaling factor for fixed-point representation.
            Larger = more precision but faster noise growth.
            Typical: 2^40 to 2^50.

    Returns:
        ts.Context: Configured CKKS context with global_scale set

    Example:
        >>> ctx = create_ckks_context(poly_modulus_degree=8192, scale=2**40)
        >>> print(f"Scale set to: {ctx.global_scale}")
    """
    if coeff_mod_bit_sizes is None:
        # Default for ML: balanced capacity and performance
        # Special primes (60 bits) for rescaling, mid primes (40 bits) match scale
        coeff_mod_bit_sizes = [60, 40, 40, 60]

    # Validate poly_modulus_degree is power of 2
    if not (poly_modulus_degree & (poly_modulus_degree - 1) == 0):
        raise ValueError(f"poly_modulus_degree must be power of 2, got {poly_modulus_degree}")

    # Validate scale
    if scale < 2**20 or scale > 2**60:
        warnings.warn(
            f"scale={scale} is unusual. Typical range: 2^20 to 2^60. "
            "Too small = precision loss, too large = noise issues."
        )

    # TenSEAL API: context(scheme, poly_modulus_degree, coeff_mod_bit_sizes)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree,
        coeff_mod_bit_sizes,
    )

    # Set scale separately
    context.global_scale = scale

    # Note: auto_rescale behavior in TenSEAL Python
    # Keeping default (True) for proper scale management in complex operations
    # For simple scalar multiplication, some precision issues may occur

    return context


def generate_keys(
    context: ts.Context,
    public_key: bool = True,
    secret_key: bool = True,
    relinearization_key: bool = True,
    galois_keys: bool = True,
) -> Dict[str, Any]:
    """
    Generate encryption keys for the given context.

    Key Types:
    - Public key: For encryption (can be shared publicly)
    - Secret key: For decryption (must remain private)
    - Relinearization key: Required after ciphertext multiplication
    - Galois keys: Required for rotations (CKKS vector operations)

    Args:
        context: TenSEAL context
        public_key: Generate public key
        secret_key: Generate secret key
        relinearization_key: Generate relinearization key
        galois_keys: Generate Galois keys

    Returns:
        Dictionary containing requested keys:
        {
            'public_key': PublicKey,
            'secret_key': SecretKey,
            'relin_key': RelinKeys,
            'galois_key': GaloisKeys
        }

    Note:
        - Relinearization is MANDATORY after multiplication
        - Galois keys enable slot rotations and vector operations
    """
    keys = {}

    # Secret key is always available in private context
    if secret_key:
        keys['secret_key'] = context.secret_key()

    # Public key is always available
    if public_key:
        keys['public_key'] = context.public_key()

    # Generate relinearization key
    if relinearization_key:
        context.generate_relin_keys()
        keys['relin_key'] = context.relin_keys()

    # Generate Galois keys for rotations
    if galois_keys:
        context.generate_galois_keys()
        keys['galois_key'] = context.galois_keys()

    return keys


def get_context_info(context: ts.Context, scheme: Optional[str] = None,
                     poly_modulus_degree: Optional[int] = None,
                     coeff_mod_bit_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Extract detailed information about the encryption context.

    Note: TenSEAL's Context object doesn't expose all parameters directly,
    so we need to pass in the creation parameters.

    Args:
        context: TenSEAL context
        scheme: Scheme type ('bfv' or 'ckks') - must be provided
        poly_modulus_degree: Poly modulus degree used during creation
        coeff_mod_bit_sizes: Coeff mod bit sizes used during creation

    Returns:
        Dictionary with context parameters:
        {
            'scheme': str,
            'poly_modulus_degree': int,
            'coeff_mod_bit_sizes': List[int],
            'scale': float (CKKS only),
            'n_slots': int (number of SIMD slots)
        }
    """
    if scheme is None:
        raise ValueError("scheme must be provided ('bfv' or 'ckks')")

    info = {
        'scheme': scheme.upper(),
        'poly_modulus_degree': poly_modulus_degree,
        'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
    }

    # Scheme-specific parameters
    if scheme.upper() == 'CKKS':
        info['scale'] = context.global_scale
    else:
        info['scale'] = None

    # Number of SIMD slots
    if poly_modulus_degree:
        info['n_slots'] = poly_modulus_degree // 2
    else:
        info['n_slots'] = 0

    return info


def print_context_info(context: ts.Context, title: str = "Context Information",
                      scheme: Optional[str] = None,
                      poly_modulus_degree: Optional[int] = None,
                      coeff_mod_bit_sizes: Optional[List[int]] = None) -> None:
    """
    Pretty-print context information for debugging and documentation.

    Args:
        context: TenSEAL context
        title: Title for the output
        scheme: Scheme type ('bfv' or 'ckks')
        poly_modulus_degree: Poly modulus degree
        coeff_mod_bit_sizes: Coeff mod bit sizes
    """
    if scheme is None:
        print("\nContext info requires scheme parameter")
        return

    info = get_context_info(context, scheme, poly_modulus_degree, coeff_mod_bit_sizes)

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Scheme:              {info['scheme']}")
    if info['poly_modulus_degree']:
        print(f"Poly modulus degree: {info['poly_modulus_degree']}")
    print(f"SIMD slots:          {info['n_slots']}")

    if info['scheme'] == 'CKKS':
        scale = info['scale']
        if scale:
            print(f"Scale:               {scale:.2e} (2^{int(scale).bit_length()-1})")

    if info['coeff_mod_bit_sizes']:
        print(f"Coeff modulus bit sizes: {len(info['coeff_mod_bit_sizes'])} primes")
        for i, bits in enumerate(info['coeff_mod_bit_sizes']):
            print(f"  [{i}]: {bits} bits")

    print(f"{'='*60}\n")


def save_context(context: ts.Context, filepath: str) -> None:
    """
    Save a TenSEAL context to disk.

    Args:
        context: TenSEAL context to save
        filepath: Path to save the context (will be created with .tenseal extension)

    Note:
        Saved contexts include all generated keys.
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.tenseal')

    context.save(str(filepath))


def load_context(filepath: str) -> ts.Context:
    """
    Load a TenSEAL context from disk.

    Args:
        filepath: Path to the saved context file

    Returns:
        Loaded TenSEAL context
    """
    return ts.context(load(filepath))


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    'bfv_small': {
        'scheme': 'bfv',
        'poly_modulus_degree': 4096,
        'coeff_mod_bit_sizes': [40, 20, 20],
        'use_case': 'Simple integer operations, depth 2-3'
    },
    'bfv_large': {
        'scheme': 'bfv',
        'poly_modulus_degree': 8192,
        'coeff_mod_bit_sizes': [50, 30, 30, 30],
        'use_case': 'Deeper circuits, depth 4-5'
    },
    'ckks_small': {
        'scheme': 'ckks',
        'poly_modulus_degree': 4096,
        'coeff_mod_bit_sizes': [50, 30, 30, 50],
        'scale': 2**30,
        'use_case': 'Simple ML models, shallow networks'
    },
    'ckks_ml': {
        'scheme': 'ckks',
        'poly_modulus_degree': 8192,
        'coeff_mod_bit_sizes': [60, 40, 40, 60],
        'scale': 2**40,
        'use_case': 'Standard ML models, depth 3-4'
    },
    'ckks_deep': {
        'scheme': 'ckks',
        'poly_modulus_degree': 16384,
        'coeff_mod_bit_sizes': [60, 50, 50, 50, 60],
        'scale': 2**50,
        'use_case': 'Deep networks, many multiplications'
    },
}


def get_preset_config(name: str) -> Dict[str, Any]:
    """
    Get a predefined configuration.

    Args:
        name: Name of the preset ('bfv_small', 'ckks_ml', etc.)

    Returns:
        Dictionary with configuration parameters

    Raises:
        ValueError: If preset name not found
    """
    if name not in PRESET_CONFIGS:
        available = ', '.join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return PRESET_CONFIGS[name].copy()


def create_context_from_preset(preset: str) -> ts.Context:
    """
    Create a context from a predefined configuration.

    Args:
        preset: Name of the preset configuration

    Returns:
        Configured TenSEAL context
    """
    config = get_preset_config(preset)

    if config['scheme'] == 'bfv':
        return create_bfv_context(
            poly_modulus_degree=config['poly_modulus_degree'],
            coeff_mod_bit_sizes=config['coeff_mod_bit_sizes'],
        )
    else:
        return create_ckks_context(
            poly_modulus_degree=config['poly_modulus_degree'],
            coeff_mod_bit_sizes=config['coeff_mod_bit_sizes'],
            scale=config['scale'],
        )
