"""
Sealed Storage for TEE
=======================

Simulates sealed storage (encrypted persistent data) for TEE.

In Intel SGX:
- Sealing encrypts data with enclave-specific key
- Sealed data can only be unsealed by same enclave
- Keys derived from measurement + CPU secret
- Provides confidentiality and integrity

This simulation:
- Encrypts data with enclave-specific encryption key
- Derives keys from enclave measurement
- Provides sealed storage interface
"""

from typing import Dict, Optional
from dataclasses import dataclass
import os
import hashlib
import json
import pathlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

from tee_ml.core.enclave import Enclave


class SealedStorageError(Exception):
    """Exception raised for sealed storage operations."""
    pass


@dataclass
class SealedData:
    """
    Encrypted sealed data blob.

    Contains:
    - ciphertext: Encrypted data
    - nonce: AES-GCM nonce
    - tag: Authentication tag
    - enclave_id: Enclave that sealed the data
    - measurement: Enclave measurement at sealing time
    """

    ciphertext: bytes
    nonce: bytes
    tag: bytes
    enclave_id: str
    measurement: bytes

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "ciphertext": self.ciphertext.hex(),
            "nonce": self.nonce.hex(),
            "tag": self.tag.hex(),
            "enclave_id": self.enclave_id,
            "measurement": self.measurement.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'SealedData':
        """Create from dictionary."""
        return cls(
            ciphertext=bytes.fromhex(data["ciphertext"]),
            nonce=bytes.fromhex(data["nonce"]),
            tag=bytes.fromhex(data["tag"]),
            enclave_id=data["enclave_id"],
            measurement=bytes.fromhex(data["measurement"]),
        )

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'SealedData':
        """Create from JSON."""
        return cls.from_dict(json.loads(json_str))


class SealedStorage:
    """
    Sealed storage service for TEE persistent data.

    Uses AES-256-GCM for encryption and authentication.

    Key derivation:
    - Key is derived from enclave measurement
    - Only same enclave can unseal data
    - Provides confidentiality and integrity

    Example:
        >>> enclave = Enclave(enclave_id="my-enclave")
        >>> storage = SealedStorage()
        >>> data = b"secret data"
        >>> sealed = storage.seal(data, enclave.enclave_id, enclave.get_measurement())
        >>> unsealed = storage.unseal(sealed, enclave.enclave_id, enclave.get_measurement())
        >>> assert unsealed == data
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize sealed storage.

        Args:
            storage_path: Directory to store sealed data (optional)
        """
        self.storage_path = storage_path
        self._sealed_data_cache: Dict[str, SealedData] = {}

        if storage_path:
            pathlib.Path(storage_path).mkdir(parents=True, exist_ok=True)

    def _derive_key(
        self,
        enclave_id: str,
        measurement: bytes,
        salt: bytes = None,
    ) -> bytes:
        """
        Derive encryption key from enclave identity.

        In real SGX, key is derived from:
        - Measurement (MRENCLAVE)
        - CPU secret (fused in hardware)
        - Key policy (seal vs. unseal)

        This simulation uses:
        - SHA256(measurement || enclave_id || salt)

        Args:
            enclave_id: Enclave identifier
            measurement: Enclave measurement hash
            salt: Salt for key derivation

        Returns:
            256-bit encryption key
        """
        if salt is None:
            salt = b"sealed_storage_salt"

        # Derive key using HKDF-like construction
        key_material = measurement + enclave_id.encode() + salt
        key = hashlib.sha256(key_material).digest()

        return key

    def seal(
        self,
        data: bytes,
        enclave_id: str,
        measurement: bytes,
    ) -> SealedData:
        """
        Seal (encrypt) data for enclave.

        Args:
            data: Plaintext data to seal
            enclave_id: Enclave identifier
            measurement: Enclave measurement hash

        Returns:
            SealedData with encrypted blob

        Raises:
            SealedStorageError: If sealing fails
        """
        try:
            # Derive encryption key
            key = self._derive_key(enclave_id, measurement)

            # Generate random nonce
            nonce = os.urandom(12)  # 96-bit nonce for GCM

            # Encrypt with AES-256-GCM
            aesgcm = AESGCM(key)
            ciphertext_with_tag = aesgcm.encrypt(nonce, data, None)

            # Split ciphertext and tag (GCM appends 16-byte tag)
            ciphertext = ciphertext_with_tag[:-16]
            tag = ciphertext_with_tag[-16:]

            # Create sealed data blob
            sealed = SealedData(
                ciphertext=ciphertext,
                nonce=nonce,
                tag=tag,
                enclave_id=enclave_id,
                measurement=measurement,
            )

            return sealed

        except Exception as e:
            raise SealedStorageError(f"Sealing failed: {e}")

    def unseal(
        self,
        sealed_data: SealedData,
        enclave_id: str,
        measurement: bytes,
    ) -> bytes:
        """
        Unseal (decrypt) data for enclave.

        Args:
            sealed_data: Sealed data blob
            enclave_id: Enclave identifier
            measurement: Enclave measurement hash

        Returns:
            Decrypted plaintext data

        Raises:
            SealedStorageError: If unsealing fails
        """
        try:
            # Verify enclave identity
            if sealed_data.enclave_id != enclave_id:
                raise SealedStorageError(
                    f"Enclave ID mismatch: expected {sealed_data.enclave_id}, "
                    f"got {enclave_id}"
                )

            if sealed_data.measurement != measurement:
                raise SealedStorageError(
                    "Enclave measurement mismatch: "
                    "data was sealed by different enclave"
                )

            # Derive decryption key
            key = self._derive_key(enclave_id, measurement)

            # Reconstruct ciphertext with tag
            ciphertext_with_tag = sealed_data.ciphertext + sealed_data.tag

            # Decrypt with AES-256-GCM
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(
                sealed_data.nonce,
                ciphertext_with_tag,
                None
            )

            return plaintext

        except Exception as e:
            raise SealedStorageError(f"Unsealing failed: {e}")

    def save_sealed(
        self,
        key: str,
        data: bytes,
        enclave_id: str,
        measurement: bytes,
    ) -> None:
        """
        Seal and save data to persistent storage.

        Args:
            key: Key to store sealed data under
            data: Data to seal
            enclave_id: Enclave identifier
            measurement: Enclave measurement

        Raises:
            SealedStorageError: If storage path not set or saving fails
        """
        if not self.storage_path:
            raise SealedStorageError("No storage path configured")

        # Seal data
        sealed = self.seal(data, enclave_id, measurement)

        # Save to file
        file_path = pathlib.Path(self.storage_path) / f"{key}.sealed"
        with open(file_path, 'wb') as f:
            f.write(sealed.to_json().encode())

        # Cache in memory
        self._sealed_data_cache[key] = sealed

    def load_sealed(
        self,
        key: str,
        enclave_id: str,
        measurement: bytes,
    ) -> bytes:
        """
        Load and unseal data from persistent storage.

        Args:
            key: Key to load sealed data from
            enclave_id: Enclave identifier
            measurement: Enclave measurement

        Returns:
            Unsealed plaintext data

        Raises:
            SealedStorageError: If loading or unsealing fails
        """
        if not self.storage_path:
            raise SealedStorageError("No storage path configured")

        # Load from file
        file_path = pathlib.Path(self.storage_path) / f"{key}.sealed"
        try:
            with open(file_path, 'rb') as f:
                json_str = f.read().decode()
        except FileNotFoundError:
            raise SealedStorageError(f"No sealed data found for key: {key}")

        # Parse sealed data
        sealed = SealedData.from_json(json_str)

        # Unseal
        return self.unseal(sealed, enclave_id, measurement)

    def delete_sealed(self, key: str) -> None:
        """
        Delete sealed data from storage.

        Args:
            key: Key to delete
        """
        if not self.storage_path:
            return

        file_path = pathlib.Path(self.storage_path) / f"{key}.sealed"
        if file_path.exists():
            file_path.unlink()

        # Remove from cache
        if key in self._sealed_data_cache:
            del self._sealed_data_cache[key]

    def list_sealed_keys(self) -> list:
        """
        List all keys in sealed storage.

        Returns:
            List of keys
        """
        if not self.storage_path:
            return []

        storage_dir = pathlib.Path(self.storage_path)
        if not storage_dir.exists():
            return []

        keys = []
        for file_path in storage_dir.glob("*.sealed"):
            key = file_path.stem
            keys.append(key)

        return keys


def seal_model_weights(
    weights: Dict[str, bytes],
    enclave: Enclave,
    storage: SealedStorage = None,
) -> Dict[str, str]:
    """
    Seal model weights for persistent storage.

    Args:
        weights: Dictionary of layer name -> weight bytes
        enclave: Enclave to seal for
        storage: SealedStorage instance (created if None)

    Returns:
        Dictionary of layer name -> storage key
    """
    if storage is None:
        storage = SealedStorage()

    storage_keys = {}
    for layer_name, weight_data in weights.items():
        # Generate storage key
        key = f"{enclave.enclave_id}_{layer_name}_weights"

        # Seal and save
        storage.save_sealed(
            key=key,
            data=weight_data,
            enclave_id=enclave.enclave_id,
            measurement=enclave.get_measurement(),
        )

        storage_keys[layer_name] = key

    return storage_keys


def unseal_model_weights(
    storage_keys: Dict[str, str],
    enclave: Enclave,
    storage: SealedStorage = None,
) -> Dict[str, bytes]:
    """
    Unseal model weights from persistent storage.

    Args:
        storage_keys: Dictionary of layer name -> storage key
        enclave: Enclave to unseal for
        storage: SealedStorage instance (created if None)

    Returns:
        Dictionary of layer name -> weight bytes
    """
    if storage is None:
        storage = SealedStorage()

    weights = {}
    for layer_name, key in storage_keys.items():
        # Load and unseal
        weight_data = storage.load_sealed(
            key=key,
            enclave_id=enclave.enclave_id,
            measurement=enclave.get_measurement(),
        )

        weights[layer_name] = weight_data

    return weights


def create_sealed_storage(
    storage_path: str = "/tmp/tee_sealed_data",
) -> SealedStorage:
    """
    Factory function to create sealed storage.

    Args:
        storage_path: Directory for sealed data

    Returns:
        SealedStorage instance
    """
    return SealedStorage(storage_path=storage_path)
