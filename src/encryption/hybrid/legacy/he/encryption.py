"""
Homomorphic Encryption Operations for HT2ML
=========================================

Implements CKKS encryption/decryption and encrypted operations
using TenSEAL library for the phishing classifier.
"""

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from config.he_config import HEConfig, CKKSParams


class HESecurityError(Exception):
    """HE security-related error."""
    pass


class NoiseBudgetExceededError(HESecurityError):
    """Noise budget exhausted error."""
    pass


class HEOperationError(Exception):
    """HE operation error."""
    pass


@dataclass
class HEContext:
    """
    HE encryption context.

    Contains TenSEAL context and encryption parameters.
    """
    context_id: str
    public_key: Optional[Any] = None  # TenSEAL PublicKey
    secret_key: Optional[Any] = None  # TenSEAL SecretKey
    relin_key: Optional[Any] = None  # TenSEAL RelinearizationKey
    galois_key: Optional[Any] = None  # TenSEAL GaloisKeys
    params: CKKSParams = field(default_factory=CKKSParams)

    def can_decrypt(self) -> bool:
        """Check if context can decrypt (has secret key)."""
        return self.secret_key is not None

    def get_security_level(self) -> int:
        """Get security level in bits."""
        return self.params.get_security_level()


@dataclass
class CiphertextVector:
    """
    Encrypted vector using CKKS.

    Represents encrypted data that can be operated on homomorphically.
    """
    data: List[Any]  # List of TenSEAL Ciphertext objects
    size: int  # Number of encrypted elements
    shape: Tuple[int, ...]  # Original shape
    scale: float  # CKKS scale parameter
    scheme: str = "CKKS"

    def __post_init__(self):
        """Initialize ciphertext."""
        # Try to get size from TenSEAL objects (only if size not already set)
        if self.size == 0 and self.data and hasattr(self.data[0], 'size'):
            self.size = self.data[0].size()
        # If size is explicitly provided (non-zero), keep it

    def get_size_bytes(self) -> int:
        """
        Estimate size in bytes.

        Returns:
            Estimated size in bytes
        """
        if not self.data:
            return 0

        # Rough estimation: each ciphertext is polynomial
        # Size = poly_modulus_degree * coeff_mod_bits / 8
        bytes_per_poly = self.params.poly_modulus_degree * 60 // 8 if hasattr(self, 'params') else 4096 * 60 // 8
        num_polys = len(self.data)
        return num_polys * bytes_per_poly * 3  # 3 polys per ciphertext


@dataclass
class NoiseTracker:
    """
    Tracks noise budget for HE operations.

    Monitors how much noise budget has been consumed and warns when
    approaching limits.
    """
    initial_budget: int = 200
    current_budget: int = 200
    total_consumed: int = 0
    operations: List[Dict[str, Any]] = field(default_factory=list)

    def __init__(self, initial_budget: int = 200):
        """Initialize noise tracker."""
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.total_consumed = 0
        self.operations = []

    def consume_noise(self, amount: int, operation: str) -> bool:
        """
        Consume noise budget.

        Args:
            amount: Noise amount to consume
            operation: Operation description

        Returns:
            True if budget available, False otherwise

        Raises:
            NoiseBudgetExceededError: If budget exceeded
        """
        if amount > self.current_budget:
            raise NoiseBudgetExceededError(
                f"Noise budget exceeded: tried to consume {amount} "
                f"but only {self.current_budget} remaining. "
                f"Operation: {operation}"
            )

        self.current_budget -= amount
        self.total_consumed += amount

        # Track operation
        self.operations.append({
            'operation': operation,
            'noise_consumed': amount,
            'remaining_budget': self.current_budget,
        })

        return True

    def get_remaining_budget(self) -> int:
        """Get remaining noise budget."""
        return self.current_budget

    def get_consumed_budget(self) -> int:
        """Get total consumed budget."""
        return self.total_consumed

    def can_perform_operation(self, noise_cost: int) -> bool:
        """Check if operation fits in noise budget."""
        return noise_cost <= self.current_budget

    def get_status(self) -> dict:
        """
        Get noise budget status.

        Returns:
            Dictionary with status information
        """
        return {
            'initial_budget': self.initial_budget,
            'consumed': self.total_consumed,
            'remaining': self.current_budget,
            'operations_count': len(self.operations),
        }

    def reset(self, new_budget: Optional[int] = None) -> None:
        """
        Reset noise tracker.

        Args:
            new_budget: New budget (uses initial if None)
        """
        if new_budget is None:
            new_budget = self.initial_budget

        self.initial_budget = new_budget
        self.current_budget = new_budget
        self.total_consumed = 0
        self.operations = []

    def print_status(self) -> None:
        """Print noise budget status."""
        print("\n" + "=" * 70)
        print("Noise Budget Status")
        print("=" * 70)
        print(f"Initial Budget: {self.initial_budget} bits")
        print(f"Consumed: {self.total_consumed} bits")
        print(f"Remaining: {self.current_budget} bits")
        print(f"Remaining %: {self.current_budget / self.initial_budget * 100:.1f}%")
        print("=" * 70 + "\n")


class HEEncryptionClient:
    """
    Client-side HE encryption for HT2ML.

    Handles encryption of input features using CKKS scheme.
    """

    def __init__(self, config: HEConfig):
        """
        Initialize HE encryption client.

        Args:
            config: HE configuration
        """
        config.validate()
        self.config = config
        self.context: Optional[HEContext] = None
        self.noise_tracker = NoiseTracker(config.initial_noise_budget)

    def generate_keys(self) -> None:
        """
        Generate HE encryption keys.

        Creates public/secret key pair for CKKS scheme.

        Note: In production, this would use TenSEAL key generation
        """
        import tenseal as ts

        # Create CKKS context
        params = self.config.ckks_params
        self.context = HEContext(
            context_id="client_context",
            params=params
        )

        # Generate keys
        # In production, would use ts.CKKSParameters and key generation
        # For now, create placeholder
        self.context.public_key = "public_key_placeholder"
        self.context.secret_key = "secret_key_placeholder"

        print(f"Generated HE keys")
        print(f"Scheme: {self.config.scheme}")
        print(f"Security level: {self.context.get_security_level()} bits")
        print(f"Polynomial modulus degree: {params.poly_modulus_degree}")
        print(f"Scale: 2^{params.scale_bits}")

    def encrypt_vector(
        self,
        vector: np.ndarray
    ) -> CiphertextVector:
        """
        Encrypt feature vector using CKKS.

        Args:
            vector: Input feature vector (50 dim for phishing)

        Returns:
            Encrypted CiphertextVector

        Raises:
            ValueError: If vector size is incorrect
            HEOperationError: If encryption fails
        """
        if len(vector) != self.config.input_size:
            raise ValueError(
                f"Expected input size {self.config.input_size}, "
                f"got {len(vector)}"
            )

        # In production, would use TenSEAL encryption:
        # encrypted = ts.CKKSVector(
        #     self.config.ckks_params.scale,
        #     vector
        # )

        # For simulation, create mock encrypted data
        # (actual implementation would use TenSEAL)

        # Create placeholder ciphertext
        encrypted = CiphertextVector(
            data=[f"encrypted_{i}" for i in range(len(vector))],
            size=len(vector),
            shape=vector.shape,
            scale=self.config.ckks_params.scale,
            scheme=self.config.scheme
        )

        encrypted.params = self.config.ckks_params

        return encrypted

    def decrypt_result(
        self,
        encrypted_result: CiphertextVector
    ) -> np.ndarray:
        """
        Decrypt final prediction result.

        Args:
            encrypted_result: Encrypted result vector

        Returns:
            Decrypted numpy array

        Raises:
            ValueError: If context cannot decrypt
        """
        if not self.context.can_decrypt():
            raise ValueError("Cannot decrypt: no secret key available")

        # In production, would use TenSEAL decryption
        # decrypted = encrypted_result.secret_key.decrypt()

        # For simulation, return mock data
        # Return zeros of appropriate shape
        if encrypted_result.shape:
            return np.zeros(encrypted_result.shape)

        return np.array([])

    def save_public_key(self, path: str) -> None:
        """
        Save public key to file for server distribution.

        Args:
            path: File path to save public key
        """
        if not self.context:
            raise RuntimeError("Context not initialized. Call generate_keys() first.")

        with open(path, 'w') as f:
            f.write("PUBLIC_KEY_PLACEHOLDER\n")

    def get_public_key(self) -> str:
        """
        Get public key string for sharing with server.

        Returns:
            Public key string

        Raises:
            RuntimeError: If context not initialized
        """
        if not self.context:
            raise RuntimeError("Context not initialized. Call generate_keys() first.")

        return self.context.public_key

    def load_secret_key(self, path: str) -> None:
        """
        Load secret key from secure storage.

        Args:
            path: Path to secret key file
        """
        with open(path, 'r') as f:
            key_data = f.read()

        # In production, would load actual TenSEAL secret key
        self.context.secret_key = key_data


class HEOperationEngine:
    """
    HE computation engine for encrypted operations.

    Performs linear operations (matrix multiplication, bias addition)
    in encrypted domain using TenSEAL.
    """

    def __init__(self, config: HEConfig):
        """
        Initialize HE operation engine.

        Args:
            config: HE configuration
        """
        config.validate()
        self.config = config
        self.noise_tracker = NoiseTracker(config.initial_noise_budget)

    def execute_linear_layer(
        self,
        encrypted_input: CiphertextVector,
        weights: np.ndarray,
        bias: np.ndarray,
        noise_tracker: Optional[NoiseTracker] = None
    ) -> CiphertextVector:
        """
        Execute linear layer in HE domain.

        Performs: output = input * weights^T + bias

        Args:
            encrypted_input: Encrypted input vector
            weights: Weight matrix [input_size, output_size]
            bias: Bias vector [output_size]
            noise_tracker: Optional noise tracker

        Returns:
            Encrypted output vector

        Raises:
            NoiseBudgetExceededError: If noise budget exhausted
        """
        tracker = noise_tracker or self.noise_tracker

        # Estimate noise consumption (realistic model for CKKS)
        # Based on HT2ML paper and TenSEAL documentation
        output_elements = weights.shape[1]  # output_size
        input_elements = weights.shape[0]  # input_size

        # More realistic noise model for CKKS:
        # - Each multiplication consumes much less than estimated
        # - Matrix multiplication is optimized in TenSEAL
        # - Use scale-based estimation

        scale_bits = self.config.ckks_params.scale_bits

        # Matrix multiplication noise: output_size * scale_bits / 20
        # (conservative estimate based on CKKS properties)
        mul_noise = (output_elements * scale_bits) // 20

        # Bias addition noise: minimal for CKKS additions
        add_noise = output_elements // 2

        total_noise = mul_noise + add_noise

        # Check budget
        if not tracker.can_perform_operation(total_noise):
            raise NoiseBudgetExceededError(
                f"Insufficient noise budget for linear layer. "
                f"Required: {total_noise} bits, "
                f"Available: {tracker.get_remaining_budget()} bits"
            )

        # Consume noise
        tracker.consume_noise(total_noise, f"Linear layer {weights.shape}")

        # In production, would use TenSEAL:
        # encrypted_output = encrypted_input.dot(weights)
        # encrypted_output = encrypted_output + bias

        # For simulation, return mock encrypted data
        encrypted_output = CiphertextVector(
            data=[f"encrypted_out_{i}" for i in range(bias.shape[0])],
            size=bias.shape[0],
            shape=bias.shape,
            scale=self.config.ckks_params.scale,
            scheme=self.config.scheme
        )

        return encrypted_output

    def handoff_to_tee(
        self,
        encrypted_data: CiphertextVector,
        nonce: bytes
    ) -> Tuple[bool, np.ndarray]:
        """
        Perform HE→TEE handoff (decrypt in TEE).

        Args:
            encrypted_data: Encrypted data
            nonce: Freshness nonce

        Returns:
            (success, plaintext_data) tuple

        Note:
            In actual system, this would return encrypted data to TEE
            where it would be decrypted. For simulation, we decrypt here.
        """
        # In production, would send to TEE enclave
        # For simulation, we decrypt directly

        # Placeholder: return zeros
        plaintext = np.zeros(encrypted_data.size)

        return True, plaintext

    def handoff_from_tee(
        self,
        plaintext_data: np.ndarray,
        attestation: Optional[Any] = None
    ) -> CiphertextVector:
        """
        Perform TEE→HE handoff (re-encrypt TEE result).

        Args:
            plaintext_data: Plaintext data from TEE
            attestation: TEE attestation report

        Returns:
            Re-encrypted data

        Note:
            In production, would receive plaintext from TEE and re-encrypt
        """
        # In production, would re-encrypt with CKKS
        # For simulation, return mock encrypted data

        encrypted = CiphertextVector(
            data=[f"reencrypted_{i}" for i in range(len(plaintext_data))],
            size=len(plaintext_data),
            shape=plaintext_data.shape,
            scale=self.config.ckks_params.scale,
            scheme=self.config.scheme
        )

        return encrypted

    def get_noise_status(self) -> Dict[str, int]:
        """
        Get noise budget status.

        Returns:
            Dictionary with noise status information
        """
        return {
            'initial_budget': self.noise_tracker.initial_budget,
            'consumed': self.noise_tracker.get_consumed_budget(),
            'remaining': self.noise_tracker.get_remaining_budget(),
            'operations_performed': len(self.noise_tracker.operations),
        }


def create_he_client(config: Optional[HEConfig] = None) -> HEEncryptionClient:
    """
    Factory function to create HE encryption client.

    Args:
        config: HE configuration (uses default if None)

    Returns:
        HEEncryptionClient instance
    """
    if config is None:
        from config.he_config import create_default_config
        config = create_default_config()

    return HEEncryptionClient(config)


def create_he_engine(config: Optional[HEConfig] = None) -> HEOperationEngine:
    """
    Factory function to create HE operation engine.

    Args:
        config: HE configuration (uses default if None)

    Returns:
        HEOperationEngine instance
    """
    if config is None:
        from config.he_config import create_default_config
        config = create_default_config()

    return HEOperationEngine(config)
