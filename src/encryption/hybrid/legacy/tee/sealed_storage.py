"""
Sealed Storage for TEE Model Deployment
========================================

Implements secure storage for model weights in TEE.
Encrypts model weights to specific TEE measurement.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pickle
import hashlib
from pathlib import Path


class SealingError(Exception):
    """Sealing operation error."""
    pass


class UnsealingError(SealingError):
    """Unsealing operation error."""
    pass


@dataclass
class SealedData:
    """
    Sealed data for TEE storage.

    Contains encrypted data bound to TEE measurement.
    """
    encrypted_data: bytes  # Encrypted payload
    measurement: bytes  # TEE measurement this is bound to
    nonce: bytes  # Encryption nonce
    tag: bytes  # Authentication tag
    version: int = 1  # Sealing version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'encrypted_data': self.encrypted_data.hex(),
            'measurement': self.measurement.hex(),
            'nonce': self.nonce.hex(),
            'tag': self.tag.hex(),
            'version': self.version,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SealedData':
        """Create from dictionary."""
        return SealedData(
            encrypted_data=bytes.fromhex(data['encrypted_data']),
            measurement=bytes.fromhex(data['measurement']),
            nonce=bytes.fromhex(data['nonce']),
            tag=bytes.fromhex(data['tag']),
            version=data.get('version', 1),
        )


@dataclass
class SealedModelBundle:
    """
    Complete sealed model bundle.

    Contains all sealed weights and metadata.
    """
    model_id: str
    sealed_weights: Dict[str, SealedData]  # Layer name -> sealed weight
    model_config: Dict[str, Any]  # Model architecture config
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())

    def save(self, path: str) -> None:
        """
        Save sealed model to file.

        Args:
            path: File path to save
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f"Sealed model saved to: {path}")

    @staticmethod
    def load(path: str) -> 'SealedModelBundle':
        """
        Load sealed model from file.

        Args:
            path: File path to load

        Returns:
            SealedModelBundle instance

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file is corrupted
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Sealed model not found: {path}")

        with open(path, 'rb') as f:
            bundle = pickle.load(f)

        print(f"Sealed model loaded from: {path}")
        return bundle


class SealedStorage:
    """
    Manages sealed storage for TEE model deployment.

    Handles encryption of model weights to specific TEE measurement.
    """

    def __init__(self):
        """Initialize sealed storage."""
        self.sealed_models: Dict[str, SealedModelBundle] = {}

    def seal_weight(
        self,
        weight: np.ndarray,
        measurement: bytes,
        layer_name: str
    ) -> SealedData:
        """
        Seal a single weight array.

        Args:
            weight: Weight array to seal
            measurement: TEE measurement to bind to
            layer_name: Name of layer (for metadata)

        Returns:
            SealedData

        Note:
            In production, would use AEAD encryption (AES-GCM)
            with key derived from TEE measurement.
            For simulation, uses simplified encryption.
        """
        # Convert weight to bytes
        weight_bytes = weight.tobytes()

        # Generate nonce
        import secrets
        nonce = secrets.token_bytes(12)

        # In production, would derive key from measurement:
        # key = HKDF(measurement, salt=nonce, info=b"HT2ML_SEALING")
        # Then encrypt with AES-GCM

        # For simulation, use simple XOR with measurement
        # (NOT secure - for demonstration only!)
        measurement_expanded = self._expand_measurement(measurement, len(weight_bytes))
        encrypted = bytes(a ^ b for a, b in zip(weight_bytes, measurement_expanded))

        # Create authentication tag (hash of encrypted + measurement)
        tag = hashlib.sha256(encrypted + measurement + nonce).digest()

        sealed = SealedData(
            encrypted_data=encrypted,
            measurement=measurement,
            nonce=nonce,
            tag=tag
        )

        return sealed

    def unseal_weight(
        self,
        sealed_data: SealedData,
        expected_measurement: bytes
    ) -> np.ndarray:
        """
        Unseal a weight array.

        Args:
            sealed_data: Sealed weight data
            expected_measurement: Expected TEE measurement

        Returns:
            Unsealed weight array

        Raises:
            UnsealingError: If measurement doesn't match or tag invalid
        """
        # Verify measurement matches
        if sealed_data.measurement != expected_measurement:
            raise UnsealingError(
                f"Measurement mismatch.\n"
                f"Expected: {expected_measurement.hex()[:40]}...\n"
                f"Got: {sealed_data.measurement.hex()[:40]}..."
            )

        # Verify authentication tag
        expected_tag = hashlib.sha256(
            sealed_data.encrypted_data +
            sealed_data.measurement +
            sealed_data.nonce
        ).digest()

        if sealed_data.tag != expected_tag:
            raise UnsealingError("Authentication tag mismatch - data may be tampered")

        # Decrypt (reverse of seal operation)
        measurement_expanded = self._expand_measurement(
            sealed_data.measurement,
            len(sealed_data.encrypted_data)
        )
        weight_bytes = bytes(
            a ^ b for a, b in zip(sealed_data.encrypted_data, measurement_expanded)
        )

        # Convert back to numpy array
        weight = np.frombuffer(weight_bytes, dtype=np.float32)

        return weight

    def seal_model(
        self,
        weights: Dict[str, np.ndarray],
        measurement: bytes,
        model_config: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> SealedModelBundle:
        """
        Seal complete model.

        Args:
            weights: Dictionary of layer weights
            measurement: TEE measurement
            model_config: Model architecture configuration
            model_id: Optional model identifier

        Returns:
            SealedModelBundle
        """
        if model_id is None:
            import secrets
            model_id = f"model_{secrets.token_urlsafe(16)}"

        # Seal each weight
        sealed_weights = {}
        for layer_name, weight_array in weights.items():
            sealed = self.seal_weight(weight_array, measurement, layer_name)
            sealed_weights[layer_name] = sealed

        # Create bundle
        bundle = SealedModelBundle(
            model_id=model_id,
            sealed_weights=sealed_weights,
            model_config=model_config,
            metadata={
                'num_layers': len(weights),
                'total_weights': sum(w.size for w in weights.values()),
            }
        )

        self.sealed_models[model_id] = bundle

        return bundle

    def unseal_model(
        self,
        bundle: SealedModelBundle,
        expected_measurement: bytes
    ) -> Dict[str, np.ndarray]:
        """
        Unseal complete model.

        Args:
            bundle: Sealed model bundle
            expected_measurement: Expected TEE measurement

        Returns:
            Dictionary of unsealed weights

        Raises:
            UnsealingError: If unsealing fails
        """
        weights = {}

        for layer_name, sealed_data in bundle.sealed_weights.items():
            try:
                weight = self.unseal_weight(sealed_data, expected_measurement)
                weights[layer_name] = weight
            except UnsealingError as e:
                raise UnsealingError(f"Failed to unseal {layer_name}: {e}")

        return weights

    def _expand_measurement(self, measurement: bytes, length: int) -> bytes:
        """
        Expand measurement to desired length using SHA256.

        Args:
            measurement: Input measurement
            length: Desired output length

        Returns:
            Expanded bytes
        """
        result = b''
        index = 0

        while len(result) < length:
            # Hash measurement + index
            data = measurement + index.to_bytes(4, 'big')
            hash_result = hashlib.sha256(data).digest()
            result += hash_result
            index += 1

        return result[:length]

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about sealed model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model info or None if not found
        """
        bundle = self.sealed_models.get(model_id)

        if bundle is None:
            return None

        return {
            'model_id': bundle.model_id,
            'num_layers': len(bundle.sealed_weights),
            'created_at': bundle.created_at,
            'measurement': bundle.sealed_weights[
                list(bundle.sealed_weights.keys())[0]
            ].measurement.hex()[:40] + '...',
        }


def create_sealed_storage() -> SealedStorage:
    """
    Factory function to create sealed storage.

    Returns:
        SealedStorage instance
    """
    return SealedStorage()


def seal_and_save_model(
    weights: Dict[str, np.ndarray],
    measurement: bytes,
    model_config: Dict[str, Any],
    output_path: str,
    model_id: Optional[str] = None
) -> None:
    """
    Convenience function to seal and save model.

    Args:
        weights: Model weights
        measurement: TEE measurement
        model_config: Model configuration
        output_path: Path to save sealed model
        model_id: Optional model ID
    """
    storage = create_sealed_storage()
    bundle = storage.seal_model(weights, measurement, model_config, model_id)
    bundle.save(output_path)


def load_and_unseal_model(
    sealed_model_path: str,
    expected_measurement: bytes
) -> Dict[str, np.ndarray]:
    """
    Convenience function to load and unseal model.

    Args:
        sealed_model_path: Path to sealed model file
        expected_measurement: Expected TEE measurement

    Returns:
        Unsealed model weights

    Raises:
        FileNotFoundError: If file doesn't exist
        UnsealingError: If unsealing fails
    """
    bundle = SealedModelBundle.load(sealed_model_path)
    storage = create_sealed_storage()
    return storage.unseal_model(bundle, expected_measurement)
