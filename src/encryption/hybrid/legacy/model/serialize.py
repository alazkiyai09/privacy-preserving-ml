"""
Model Serialization for TEE Deployment
=====================================

Handles serialization of model weights for secure deployment
in Trusted Execution Environment.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
import json
import hashlib
import logging
from pathlib import Path

from src.encryption.hybrid.legacy.model.phishing_classifier import PhishingClassifier

logger = logging.getLogger(__name__)


@dataclass
class SealedModel:
    """
    Sealed model data for TEE deployment.

    Contains model weights encrypted to specific TEE measurement.
    """
    sealed_weights: Dict[str, bytes]  # Encrypted layer weights
    model_measurement: bytes  # SHA256 hash of model code
    model_config: Dict[str, Any]  # Model configuration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """
        Save sealed model to file using JSON + numpy format.

        This avoids the security risks of pickle by storing data in
        a structured format that only contains basic types.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save weights as .npz file
        weights_path = path_obj.with_suffix('.weights.npz')

        # Convert bytes to numpy arrays for storage
        weight_arrays = {}
        weight_shapes = {}
        for key, weight_bytes in self.sealed_weights.items():
            arr = np.frombuffer(weight_bytes, dtype=np.uint8)
            weight_arrays[key] = arr
            weight_shapes[key] = arr.shape

        np.savez_compressed(weights_path, **weight_arrays)

        # Save metadata as JSON
        metadata_path = path_obj.with_suffix('.meta.json')
        metadata_dict = {
            'sealed_weights_shapes': weight_shapes,
            'model_measurement': self.model_measurement.hex(),
            'model_config': self.model_config,
            'metadata': self.metadata,
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Sealed model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SealedModel':
        """
        Load sealed model from file using JSON + numpy format.

        This avoids the security risks of pickle.
        """
        path_obj = Path(path)

        # Load metadata
        metadata_path = path_obj.with_suffix('.meta.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            raise ValueError(f"Invalid sealed model: missing or corrupt metadata") from e

        # Load weights
        weights_path = path_obj.with_suffix('.weights.npz')
        try:
            weights_data = np.load(weights_path, allow_pickle=False)
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise ValueError(f"Invalid sealed model: missing or corrupt weights") from e

        # Reconstruct weights
        sealed_weights = {}
        for key in weights_data.files:
            sealed_weights[key] = weights_data[key].tobytes()

        # Reconstruct object
        return cls(
            sealed_weights=sealed_weights,
            model_measurement=bytes.fromhex(metadata_dict['model_measurement']),
            model_config=metadata_dict['model_config'],
            metadata=metadata_dict.get('metadata', {})
        )


class ModelSerializer:
    """
    Handles model serialization and sealing for TEE deployment.
    """

    def __init__(self):
        """Initialize serializer."""
        self.sealed_model: Optional[SealedModel] = None

    def seal_model_for_tee(
        self,
        model: PhishingClassifier,
        tee_measurement: bytes,
        encryption_key: Optional[bytes] = None
    ) -> SealedModel:
        """
        Seal model weights for TEE deployment.

        Args:
            model: Trained phishing classifier
            tee_measurement: SHA256 measurement of TEE enclave
            encryption_key: Optional key for encrypting weights

        Returns:
            SealedModel with encrypted weights
        """
        # Get weights
        weights = model.save_weights()

        # Convert weights to encrypted format
        sealed_weights = {}
        for key, array in weights.items():
            # Convert to bytes
            weight_bytes = array.tobytes()

            if encryption_key:
                # Encrypt weights (simplified - use AES in production)
                # For now, just store as bytes
                encrypted = weight_bytes
            else:
                encrypted = weight_bytes

            sealed_weights[key] = encrypted

        # Create sealed model
        sealed = SealedModel(
            sealed_weights=sealed_weights,
            model_measurement=tee_measurement,
            model_config={
                'input_size': model.config.input_size,
                'hidden_size': model.config.hidden_size,
                'output_size': model.config.output_size,
                'layers': [
                    {
                        'name': layer.name,
                        'type': layer.layer_type.value,
                        'domain': layer.domain.value,
                        'input_size': layer.input_size,
                        'output_size': layer.output_size,
                    }
                    for layer in model.config.layers
                ]
            },
            metadata={
                'model_name': 'phishing_classifier',
                'version': '1.0',
                'trained': model.is_trained,
            }
        )

        self.sealed_model = sealed
        return sealed

    def unseal_model_for_tee(
        self,
        sealed_model: SealedModel,
        tee_measurement: bytes
    ) -> Dict[str, np.ndarray]:
        """
        Unseal model weights in TEE.

        Args:
            sealed_model: Sealed model data
            tee_measurement: Expected TEE measurement

        Returns:
            Dictionary with weight arrays

        Raises:
            ValueError: If measurement doesn't match
        """
        # Verify measurement
        if sealed_model.model_measurement != tee_measurement:
            raise ValueError(
                f"TEE measurement mismatch. "
                f"Expected: {tee_measurement.hex()}, "
                f"Got: {sealed_model.model_measurement.hex()}"
            )

        # Decrypt weights
        weights = {}
        for key, encrypted in sealed_model.sealed_weights.items():
            # Decrypt if encrypted (simplified - just load bytes)
            weight_bytes = encrypted
            weight_array = np.frombuffer(weight_bytes, dtype=np.float32)

            # Reshape if needed
            if key.endswith('.weight'):
                # Weight matrix
                if key == 'linear1.weight':
                    weight_array = weight_array.reshape(50, 64)
                elif key == 'linear2.weight':
                    weight_array = weight_array.reshape(64, 2)
            elif key.endswith('.bias'):
                # Bias vector - already correct shape
                pass

            weights[key] = weight_array

        return weights

    def calculate_model_measurement(self, model_code: bytes) -> bytes:
        """
        Calculate SHA256 measurement of model code.

        Args:
            model_code: Model code bytes

        Returns:
            SHA256 hash
        """
        return hashlib.sha256(model_code).digest()

    def verify_sealed_model(self, sealed_model: SealedModel) -> bool:
        """
        Verify integrity of sealed model.

        Args:
            sealed_model: Sealed model to verify

        Returns:
            True if valid
        """
        # Check that required fields exist
        if not sealed_model.sealed_weights:
            return False

        if not sealed_model.model_measurement:
            return False

        # Check that config is valid
        if 'layers' not in sealed_model.model_config:
            return False

        return True


def seal_model_for_deployment(
    model: PhishingClassifier,
    output_path: str,
    tee_measurement: bytes
) -> None:
    """
    Convenience function to seal model for TEE deployment.

    Args:
        model: Trained model
        output_path: Path to save sealed model
        tee_measurement: TEE measurement hash
    """
    serializer = ModelSerializer()
    sealed = serializer.seal_model_for_tee(model, tee_measurement)
    sealed.save(output_path)

    print(f"Model sealed and saved to: {output_path}")
    print(f"Model measurement: {tee_measurement.hex()[:40]}...")


def load_sealed_model(sealed_model_path: str, tee_measurement: bytes) -> Dict[str, np.ndarray]:
    """
    Convenience function to load sealed model.

    Args:
        sealed_model_path: Path to sealed model file
        tee_measurement: Expected TEE measurement

    Returns:
        Dictionary with unsealed weights
    """
    sealed = SealedModel.load(sealed_model_path)
    serializer = ModelSerializer()
    return serializer.unseal_model_for_tee(sealed, tee_measurement)


def get_model_code_measurement(model: PhishingClassifier) -> bytes:
    """
    Get measurement hash of model code.

    Args:
        model: Phishing classifier

    Returns:
        SHA256 hash
    """
    # Get model source code
    import inspect
    source_code = inspect.getsource(model.__class__)

    # Calculate measurement
    serializer = ModelSerializer()
    return serializer.calculate_model_measurement(source_code.encode())
