"""
Homomorphic Encryption Configuration for HT2ML
=============================================

Defines CKKS encryption parameters for the hybrid HE/TEE system.
Based on TenSEAL library and HT2ML paper.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(Enum):
    """Security level for HE parameters."""
    STANDARD = "standard"  # 128-bit security
    HIGH = "high"  # 192-bit security
    ULTRA = "ultra"  # 256-bit security


@dataclass
class CKKSParams:
    """
    CKKS scheme parameters.

    These parameters balance:
    - Security level (through polynomial modulus)
    - Noise budget (through coefficient modulus sizes)
    - Precision (through scale)
    - Performance (through poly_modulus_degree)

    Reference: HT2ML paper, TenSEAL documentation
    """

    # Polynomial modulus degree (determines batch size and security)
    # Larger = more slots but slower computation
    # Must be power of 2: 4096, 8192, 16384
    poly_modulus_degree: int = 4096

    # Coefficient modulus sizes (bits)
    # Determines noise budget - each multiplication consumes noise
    # Format: [scale_mod, ...chain of coeff_mods...]
    # Larger values = more noise capacity but slower
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])

    # Scale (delta) for CKKS encoding
    # Determines precision of fixed-point numbers
    # Must be a power of 2
    # Larger = more precision but more noise
    scale: float = 2**40

    # Base scale for initial encoding
    # Typically 2^40 for good precision
    scale_bits: int = 40

    # Number of slots available (batch size limit)
    # Calculated as: poly_modulus_degree / 2
    @property
    def num_slots(self) -> int:
        """Calculate number of available slots."""
        return self.poly_modulus_degree // 2

    def get_security_level(self) -> int:
        """
        Get security level in bits.

        Based on log2(coeff_modulus[-1])

        Returns:
            Security level in bits (128, 192, or 256)
        """
        total_bits = sum(self.coeff_mod_bit_sizes)
        # Remove scale bits to get security level
        security_bits = total_bits - self.scale_bits
        return security_bits


@dataclass
class HEConfig:
    """
    Complete HE configuration for HT2ML system.

    Configuration for CKKS scheme optimized for:
    - 50 input features
    - 64 hidden neurons
    - 2 output classes
    - Multiple HE/TEE handoffs
    """

    # CKKS scheme parameters
    scheme: str = "CKKS"

    # CKKS parameters
    ckks_params: CKKSParams = field(default_factory=CKKSParams)

    # Noise budget management
    initial_noise_budget: int = 200  # Total noise bits available
    noise_consumption_per_mul: int = 40  # Noise per multiplication
    noise_consumption_per_add: int = 40  # Noise per addition

    # Security level
    security_level: SecurityLevel = SecurityLevel.STANDARD

    # Model dimensions
    input_size: int = 50  # Input feature dimension

    # Encryption settings
    # Whether to use batching for multiple samples
    enable_batching: bool = False
    batch_size: int = 1  # If batching enabled

    # Key management
    generate_keys: bool = True
    save_public_key: bool = True
    public_key_path: str = "he_public_key.bin"
    secret_key_path: str = "he_secret_key.bin"

    # Validation
    def validate(self) -> bool:
        """Validate HE configuration."""
        # Check that poly_modulus_degree is power of 2
        if not (self.ckks_params.poly_modulus_degree &
                (self.ckks_params.poly_modulus_degree - 1) == 0):
            raise ValueError(
                f"poly_modulus_degree must be power of 2, "
                f"got {self.ckks_params.poly_modulus_degree}"
            )

        # Check that scale is power of 2
        if not (self.ckks_params.scale &
                (self.ckks_params.scale - 1) == 0):
            raise ValueError(
                f"scale must be power of 2, "
                f"got {self.ckks_params.scale}"
            )

        # Check that coeff_mod_bit_sizes has at least 3 elements
        if len(self.ckks_params.coeff_mod_bit_sizes) < 3:
            raise ValueError(
                f"coeff_mod_bit_sizes must have at least 3 elements, "
                f"got {len(self.ckks_params.coeff_mod_bit_sizes)}"
            )

        return True

    def get_max_multiplications(self) -> int:
        """
        Calculate maximum number of multiplications before noise budget exhausted.

        Returns:
            Maximum multiplications possible
        """
        return self.initial_noise_budget // self.noise_consumption_per_mul

    def estimate_noise_for_layers(
        self,
        num_linear_layers: int,
        matrix_size: int
    ) -> int:
        """
        Estimate total noise consumption for given network.

        Args:
            num_linear_layers: Number of linear layers in HE
            matrix_size: Size of weight matrices (product of dimensions)

        Returns:
            Estimated noise consumption in bits
        """
        # Each linear layer does:
        # - Matrix multiplication (consumes noise)
        # - Bias addition (consumes noise)

        # Simplified model: each output element consumes noise
        total_multiplications = 0

        for _ in range(num_linear_layers):
            # Matrix multiplication: rows * cols
            total_multiplications += matrix_size
            # Bias addition (vector add)
            total_multiplications += matrix_size // matrix_size  # Just bias

        noise_per_mul = self.noise_consumption_per_mul
        estimated_noise = total_multiplications * noise_per_mul

        return estimated_noise


def create_default_config() -> HEConfig:
    """
    Create default HE configuration optimized for HT2ML.

    Returns:
        HEConfig with safe defaults
    """
    return HEConfig()


def create_high_security_config() -> HEConfig:
    """
    Create high-security HE configuration (256-bit).

    Returns:
        HEConfig with enhanced security parameters
    """
    config = HEConfig()
    config.security_level = SecurityLevel.ULTRA
    config.ckks_params.coeff_mod_bit_sizes = [60, 40, 40, 60, 60]
    return config


def create_performance_config() -> HEConfig:
    """
    Create performance-oriented HE configuration.

    Uses smaller parameters for faster computation.

    Returns:
        HEConfig optimized for speed
    """
    config = HEConfig()
    config.ckks_params.poly_modulus_degree = 2048  # Smaller for speed
    config.ckks_params.scale_bits = 30
    config.ckks_params.scale = 2**30
    return config


def estimate_ciphertext_size(
    config: HEConfig,
    feature_count: int
) -> int:
    """
    Estimate ciphertext size in bytes.

    Args:
        config: HE configuration
        feature_count: Number of features in input

    Returns:
        Estimated size in bytes
    """
    # Rough estimation based on TenSEAL
    # Each ciphertext is roughly:
    # - poly_modulus_degree * coeff_mod_bits / 8 bytes per polynomial
    # - Number of polynomials based on slots needed

    slots_needed = (feature_count + config.ckks_params.num_slots - 1) // config.ckks_params.num_slots
    if slots_needed == 0:
        slots_needed = 1

    bits_per_poly = config.ckks_params.poly_modulus_degree * config.ckks_params.coeff_mod_bit_sizes[0]
    bytes_per_poly = bits_per_poly // 8

    estimated_size = slots_needed * bytes_per_poly * 3  # 3 polynomials per ciphertext (ciphertext)

    return estimated_size


# Predefined configurations for different scenarios

STANDARD_CONFIG = HEConfig(
    ckks_params=CKKSParams(
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        scale=2**40
    )
)

FAST_CONFIG = HEConfig(
    ckks_params=CKKSParams(
        poly_modulus_degree=2048,
        coeff_mod_bit_sizes=[40, 30, 30, 40],
        scale=2**30
    )
)

HIGH_SECURITY_CONFIG = HEConfig(
    ckks_params=CKKSParams(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60, 60],
        scale=2**40
    )
)
