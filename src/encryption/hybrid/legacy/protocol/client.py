"""
Client-Side Protocol Logic for HT2ML
=====================================

Implements client-side operations for HT2ML inference.
Handles encryption, result decryption, and session management.
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

from src.encryption.hybrid.legacy.he.encryption import HEEncryptionClient, CiphertextVector
from src.encryption.hybrid.legacy.protocol.message import MessageBuilder, create_session_id
from src.encryption.hybrid.legacy.protocol.handoff import HandoffProtocol, HandoffSession


class ClientProtocolError(Exception):
    """Client protocol error."""
    pass


@dataclass
class ClientSession:
    """
    Client session state.

    Tracks client-side state for inference request.
    """
    session_id: str
    encrypted_input: Optional[CiphertextVector] = None
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    result: Optional[np.ndarray] = None

    def has_keys(self) -> bool:
        """Check if keys are generated."""
        return self.public_key is not None and self.secret_key is not None

    def has_encrypted_input(self) -> bool:
        """Check if input is encrypted."""
        return self.encrypted_input is not None

    def has_result(self) -> bool:
        """Check if result is available."""
        return self.result is not None


class HT2MLClient:
    """
    HT2ML client implementation.

    Handles client-side operations:
    - HE key generation
    - Input encryption
    - Result decryption
    - Session management
    """

    def __init__(self, he_config=None):
        """
        Initialize HT2ML client.

        Args:
            he_config: HE configuration (uses default if None)
        """
        if he_config is None:
            from config.he_config import create_default_config
            he_config = create_default_config()

        self.he_client = HEEncryptionClient(he_config)
        self.handoff_protocol = HandoffProtocol()
        self.current_session: Optional[ClientSession] = None

    def generate_keys(self) -> None:
        """
        Generate HE encryption keys.

        Creates public/secret key pair for CKKS scheme.

        Raises:
            ClientProtocolError: If key generation fails
        """
        try:
            self.he_client.generate_keys()
            print("HE keys generated successfully")
        except Exception as e:
            raise ClientProtocolError(f"Key generation failed: {e}")

    def get_public_key(self) -> str:
        """
        Get public key for sharing with server.

        Returns:
            Public key string

        Raises:
            ClientProtocolError: If keys not generated
        """
        if not self.he_client.context:
            raise ClientProtocolError("Keys not generated. Call generate_keys() first.")

        return self.he_client.get_public_key()

    def encrypt_input(self, features: np.ndarray) -> CiphertextVector:
        """
        Encrypt input features.

        Args:
            features: Input feature vector (50 dim)

        Returns:
            Encrypted CiphertextVector

        Raises:
            ClientProtocolError: If encryption fails
        """
        try:
            encrypted = self.he_client.encrypt_vector(features)
            print(f"Input encrypted: {features.shape} -> {encrypted.size} ciphertext elements")
            return encrypted
        except Exception as e:
            raise ClientProtocolError(f"Encryption failed: {e}")

    def create_inference_session(
        self,
        features: np.ndarray
    ) -> ClientSession:
        """
        Create session for inference.

        Args:
            features: Input features

        Returns:
            ClientSession with encrypted input

        Raises:
            ClientProtocolError: If session creation fails
        """
        if not self.he_client.context:
            raise ClientProtocolError("HE context not initialized. Call generate_keys() first.")

        # Create session
        session_id = create_session_id()
        session = ClientSession(session_id=session_id)

        # Encrypt input
        encrypted = self.encrypt_input(features)
        session.encrypted_input = encrypted

        # Store keys
        session.public_key = self.he_client.get_public_key()
        session.secret_key = self.he_client.context.secret_key

        self.current_session = session

        print(f"Inference session created: {session_id}")
        return session

    def decrypt_result(
        self,
        encrypted_result: CiphertextVector
    ) -> np.ndarray:
        """
        Decrypt inference result.

        Args:
            encrypted_result: Encrypted result from server

        Returns:
            Decrypted numpy array

        Raises:
            ClientProtocolError: If decryption fails
        """
        try:
            decrypted = self.he_client.decrypt_result(encrypted_result)

            if self.current_session:
                self.current_session.result = decrypted

            print(f"Result decrypted: {decrypted.shape}")
            return decrypted

        except Exception as e:
            raise ClientProtocolError(f"Decryption failed: {e}")

    def save_public_key(self, path: str) -> None:
        """
        Save public key to file.

        Args:
            path: File path to save public key

        Raises:
            ClientProtocolError: If save fails
        """
        try:
            self.he_client.save_public_key(path)
            print(f"Public key saved to: {path}")
        except Exception as e:
            raise ClientProtocolError(f"Failed to save public key: {e}")

    def load_secret_key(self, path: str) -> None:
        """
        Load secret key from file.

        Args:
            path: Path to secret key file

        Raises:
            ClientProtocolError: If load fails
        """
        try:
            self.he_client.load_secret_key(path)
            print(f"Secret key loaded from: {path}")
        except Exception as e:
            raise ClientProtocolError(f"Failed to load secret key: {e}")

    def get_noise_status(self) -> Dict[str, Any]:
        """
        Get noise budget status from HE context.

        Returns:
            Dictionary with noise status
        """
        return self.he_client.noise_tracker.get_status()

    def print_noise_status(self) -> None:
        """Print noise budget status."""
        self.he_client.noise_tracker.print_status()


def create_ht2ml_client(he_config=None) -> HT2MLClient:
    """
    Factory function to create HT2ML client.

    Args:
        he_config: HE configuration (uses default if None)

    Returns:
        HT2MLClient instance
    """
    return HT2MLClient(he_config)


def prepare_inference_request(
    features: np.ndarray,
    he_config=None
) -> tuple:
    """
    Convenience function to prepare inference request.

    Args:
        features: Input features (50 dim)
        he_config: HE configuration

    Returns:
        (session_id, encrypted_input, public_key) tuple

    Raises:
        ClientProtocolError: If preparation fails
    """
    client = create_ht2ml_client(he_config)
    client.generate_keys()
    session = client.create_inference_session(features)

    return session.session_id, session.encrypted_input, session.public_key
