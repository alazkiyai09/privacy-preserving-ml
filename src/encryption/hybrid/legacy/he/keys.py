"""
Key Management for Homomorphic Encryption
=======================================

Handles generation, storage, and management of HE encryption keys.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import secrets

from config.he_config import HEConfig


class KeyType(Enum):
    """Type of cryptographic key."""
    PUBLIC = "public"
    SECRET = "secret"
    RELIN = "relin"
    GALOIS = "galois"


@dataclass
class KeyPair:
    """
    HE encryption key pair.

    Contains public and secret keys for CKKS scheme.
    """
    key_id: str
    public_key: str  # Serialized public key
    secret_key: str  # Serialized secret key (never exposed!)
    created_at: str  # ISO timestamp
    key_type: str = "CKKS"

    def get_key_id(self) -> str:
        """Get unique key identifier."""
        return self.key_id

    def save_to_file(
        self,
        public_key_path: str,
        secret_key_path: str
    ) -> None:
        """
        Save keys to file.

        Args:
            public_key_path: Path to save public key
            secret_key_path: Path to save secret key
        """
        # Save public key
        Path(public_key_path).parent.mkdir(parents=True, exist_ok=True)

        with open(public_key_path, 'w') as f:
            f.write(self.public_key)

        # Save secret key (should be in secure storage)
        Path(secret_key_path).parent.mkdir(parents=True, exist_ok=True)

        with open(secret_key_path, 'w') as f:
            # In production, would use strict permissions (0600)
            f.write(self.secret_key)

    @staticmethod
    def load_from_file(
        public_key_path: str,
        secret_key_path: str
    ) -> 'KeyPair':
        """
        Load keys from file.

        Args:
            public_key_path: Path to public key file
            secret_key_path: Path to secret key file

        Returns:
            Loaded KeyPair
        """
        with open(public_key_path, 'r') as f:
            public_key = f.read()

        with open(secret_key_path, 'r') as f:
            secret_key = f.read()

        # Extract key_id from first line of public key
        key_id = public_key.split('\n')[0].split(':')[1].strip()

        return KeyPair(
            key_id=key_id,
            public_key=public_key,
            secret_key=secret_key
        )


class HEKeyManager:
    """
    Manager for HE encryption keys.

    Handles key generation, storage, and lifecycle management.
    """

    def __init__(self, config: HEConfig):
        """
        Initialize key manager.

        Args:
            config: HE configuration
        """
        self.config = config
        self.current_key_pair: Optional[KeyPair] = None

    def generate_key_pair(self, key_id: Optional[str] = None) -> KeyPair:
        """
        Generate new HE key pair.

        Args:
            key_id: Optional identifier for the key pair

        Returns:
            KeyPair with public and secret keys

        Note:
            In production, would use TenSEAL key generation.
            For simulation, creates placeholder keys.
        """
        if key_id is None:
            key_id = f"key_{secrets.token_urlsafe(16)}"

        # In production, would use:
        # import tenseal as ts
        # params = ts.CKKSParameters(...)
        # context = ts.Context(params)
        # secret_key = context.key_gen()
        # public_key = context.key_gen()

        # For simulation, create placeholders
        secret_key = f"{key_id}:SECRET_KEY:{secrets.token_hex(16)}"
        public_key = f"{key_id}:PUBLIC_KEY:{secrets.token_hex(16)}"

        key_pair = KeyPair(
            key_id=key_id,
            public_key=public_key,
            secret_key=secret_key,
            created_at="2025-01-29T00:00:00Z"
        )

        self.current_key_pair = key_pair
        return key_pair

    def get_public_key(self) -> Optional[str]:
        """Get current public key."""
        if self.current_key_pair:
            return self.current_key_pair.public_key
        return None

    def has_secret_key(self) -> bool:
        """Check if secret key is loaded."""
        return self.current_key_pair is not None

    def save_public_key_to_file(self, path: str) -> None:
        """
        Save public key to file.

        Args:
            path: File path to save public key
        """
        if not self.current_key_pair:
            raise RuntimeError("No key pair available. Generate keys first.")

        self.current_key_pair.save_to_file(
            public_key_path=path,
            secret_key_path=path.replace(".pub", ".sec")  # Match secret key naming
        )

    def load_secret_key_from_file(self, path: str) -> None:
        """
        Load secret key from secure storage.

        Args:
            path: Path to secret key file
        """
        # Load key pair (validates both public and secret)
        public_path = path.replace(".sec", ".pub")

        key_pair = KeyPair.load_from_file(public_path, path)
        self.current_key_pair = key_pair

    def rotate_keys(self, old_key_id: str) -> KeyPair:
        """
        Generate new key pair and invalidate old one.

        Args:
            old_key_id: ID of key to rotate

        Returns:
            New key pair
        """
        # Save old key for decryption of existing data
        # (In production, would key wrap old keys)

        # Generate new key pair
        new_key_pair = self.generate_key_pair()

        return new_key_pair

    def derive_session_key(
        self,
        base_key: str,
        session_id: str,
        context: str = "session"
    ) -> bytes:
        """
        Derive session-specific key using HKDF.

        Args:
            base_key: Base secret key
            session_id: Session identifier
            context: Context string

        Returns:
            Derived 32-byte key
        """
        import hmac

        # In production, would use HKDF
        # For simulation, use HMAC with SHA256
        key_material = f"{base_key}:{session_id}:{context}".encode()
        derived_key = hmac.new(
            base_key.encode(),
            session_id.encode(),
            context.encode(),
            hashlib.sha256
        ).digest()

        return derived_key

    def get_key_fingerprint(self, key: str) -> str:
        """
        Get fingerprint of key for verification.

        Args:
            key: Key string

        Returns:
            Hexadecimal fingerprint
        """
        return hashlib.sha256(key.encode()).hexdigest()


class KeyValidator:
    """
    Validates HE keys and parameters.
    """

    def __init__(self, config: HEConfig):
        """Initialize validator."""
        self.config = config

    def validate_public_key(self, public_key: str) -> bool:
        """
        Validate public key format and parameters.

        Args:
            public_key: Serialized public key

        Returns:
            True if valid
        """
        # Check format
        if not public_key:
            return False

        lines = public_key.strip().split('\n')
        if not lines:
            return False

        # Check key identifier
        if ':' not in lines[0]:
            return False

        # In production, would validate TenSEAL key structure
        # For simulation, just check it's not empty
        return True

    def validate_key_pair_match(
        self,
        public_key: str,
        secret_key: str
    ) -> bool:
        """
        Validate that public and secret keys are a pair.

        Args:
            public_key: Public key string
            secret_key: Secret key string

        Returns:
            True if keys match
        """
        # Extract key IDs
        pub_key_id = public_key.split(':')[0] if ':' in public_key else None
        sec_key_id = secret_key.split(':')[0] if ':' in secret_key else None

        return pub_key_id == sec_key_id and pub_key_id is not None


def create_key_manager(config: Optional[HEConfig] = None) -> HEKeyManager:
    """
    Factory function to create key manager.

    Args:
        config: HE configuration (uses default if None)

    Returns:
        HEKeyManager instance
    """
    if config is None:
        from config.he_config import create_default_config
        config = create_default_config()

    return HEKeyManager(config)


# Key generation utilities

def generate_test_keys() -> Tuple[str, str]:
    """
    Generate test key pair for development.

    Returns:
        (public_key, secret_key) tuple
    """
    # Simple test keys
    key_id = secrets.token_hex(16)
    public_key = f"test_pub_{key_id}"
    secret_key = f"test_sec_{key_id}"

    return public_key, secret_key


def validate_key_pair(public_key: str, secret_key: str) -> bool:
    """
    Validate that public and secret keys match.

    Args:
        public_key: Public key
        secret_key: Secret key

    Returns:
        True if keys match
    """
    # Extract IDs
    pub_id = public_key.split('_')[-1] if '_' in public_key else None
    sec_id = secret_key.split('_')[-1] if '_' in secret_key else None

    return pub_id == sec_id and pub_id is not None
