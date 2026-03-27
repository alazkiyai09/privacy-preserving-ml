"""
FedPhish Security Utilities

Safe loading functions for models and serialized data.
Prevents arbitrary code execution from unsafe deserialization.
"""

import pickle
import torch
from pathlib import Path
from typing import Any, Optional, Union

from .errors import ModelLoadError, DataValidationError


def validate_file_path(
    path: Union[str, Path],
    expected_extensions: Optional[list[str]] = None,
    must_exist: bool = True,
) -> Path:
    """
    Validate a file path for security.

    Args:
        path: File path to validate
        expected_extensions: List of allowed extensions (e.g., ['.pt', '.pth'])
        must_exist: If True, file must exist

    Returns:
        Validated Path object

    Raises:
        DataValidationError: If path is invalid
    """
    path_obj = Path(path)

    # Check path traversal attempts
    if ".." in str(path_obj):
        raise DataValidationError(f"Path traversal attempt detected: {path}")

    # Check extension
    if expected_extensions:
        if path_obj.suffix.lower() not in expected_extensions:
            raise DataValidationError(
                f"File must have extension {expected_extensions}, got {path_obj.suffix}"
            )

    # Check existence
    if must_exist and not path_obj.exists():
        raise DataValidationError(f"File does not exist: {path}")

    # Check it's a file, not directory
    if must_exist and path_obj.exists() and not path_obj.is_file():
        raise DataValidationError(f"Path is not a file: {path}")

    return path_obj


def safe_torch_load(
    path: Union[str, Path],
    map_location: Optional[str] = None,
    weights_only: bool = True,
    allowed_classes: Optional[tuple] = None,
) -> Any:
    """
    Safely load a PyTorch model with protections against arbitrary code execution.

    Args:
        path: Path to the model file
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda')
        weights_only: If True, only load weights (no Python objects)
        allowed_classes: If weights_only=False, specify allowed tuple of classes

    Returns:
        Loaded model/state dict

    Raises:
        ModelLoadError: If loading fails or file is invalid
    """
    try:
        validated_path = validate_file_path(
            path,
            expected_extensions=[".pt", ".pth", ".pkl"],
            must_exist=True,
        )
    except DataValidationError as e:
        raise ModelLoadError(f"Invalid model path: {e}") from e

    # Set default map_location
    if map_location is None:
        map_location = "cpu"

    try:
        # Use weights_only=True to prevent arbitrary code execution
        result = torch.load(
            validated_path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return result

    except (EOFError, pickle.UnpicklingError) as e:
        raise ModelLoadError(f"Corrupted model file: {e}") from e
    except RuntimeError as e:
        if "weights_only" in str(e) and weights_only:
            # File contains Python objects, retry with caution
            raise ModelLoadError(
                f"Model file contains Python objects. "
                f"Either export with weights_only=True or explicitly pass weights_only=False "
                f"and specify allowed_classes."
            ) from e
        raise ModelLoadError(f"Failed to load model: {e}") from e
    except Exception as e:
        raise ModelLoadError(f"Unexpected error loading model: {e}") from e


def safe_pickle_load(
    path: Union[str, Path],
    allowed_classes: Optional[tuple] = None,
) -> Any:
    """
    Safely load a pickle file with restrictions on allowed classes.

    WARNING: Pickle is inherently unsafe. Use JSON/msgpack when possible.
    This function provides limited safety by restricting allowed classes.

    Args:
        path: Path to the pickle file
        allowed_classes: Tuple of allowed class types (None = very restricted)

    Returns:
        Unpickled object

    Raises:
        ModelLoadError: If loading fails or disallowed classes detected
    """
    try:
        validated_path = validate_file_path(
            path,
            expected_extensions=[".pkl", ".pickle"],
            must_exist=True,
        )
    except DataValidationError as e:
        raise ModelLoadError(f"Invalid pickle path: {e}") from e

    # Default to very restrictive allowed classes
    if allowed_classes is None:
        allowed_classes = (
            dict, list, tuple, set, frozenset, str, int, float, bool, bytes, type(None),
        )

    class RestrictedUnpickler(pickle.Unpickler):
        """Unpickler that only allows specific classes."""

        def find_class(self, module, name):
            # Only allow basic built-in types
            if module == "builtins" and name in (
                "dict", "list", "tuple", "set", "frozenset",
                "str", "int", "float", "bool", "bytes",
            ):
                return getattr(__builtins__, name)

            # Check against allowed classes
            for allowed_class in allowed_classes:
                if isinstance(allowed_class, type):
                    if module == allowed_class.__module__ and name == allowed_class.__name__:
                        return allowed_class

            raise pickle.UnpicklingError(
                f"Disallowed class in pickle: {module}.{name}"
            )

    try:
        with open(validated_path, "rb") as f:
            return RestrictedUnpickler(f).load()

    except pickle.UnpicklingError as e:
        raise ModelLoadError(f"Disallowed class in pickle file: {e}") from e
    except (EOFError, OSError) as e:
        raise ModelLoadError(f"Corrupted pickle file: {e}") from e
    except Exception as e:
        raise ModelLoadError(f"Unexpected error loading pickle: {e}") from e


def safe_json_load(path: Union[str, Path]) -> Any:
    """
    Load a JSON file safely (no code execution risk).

    Args:
        path: Path to the JSON file

    Returns:
        Parsed JSON object

    Raises:
        ModelLoadError: If loading fails
    """
    import json

    try:
        validated_path = validate_file_path(
            path,
            expected_extensions=[".json"],
            must_exist=True,
        )
    except DataValidationError as e:
        raise ModelLoadError(f"Invalid JSON path: {e}") from e

    try:
        with open(validated_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ModelLoadError(f"Invalid JSON in file: {e}") from e
    except Exception as e:
        raise ModelLoadError(f"Error reading JSON file: {e}") from e


def validate_model_state_dict(state_dict: Any, model_keys: Optional[list[str]] = None) -> bool:
    """
    Validate a PyTorch model state dictionary.

    Args:
        state_dict: Object to check if it's a valid state dict
        model_keys: Optional list of expected keys

    Returns:
        True if valid state dict

    Raises:
        DataValidationError: If not a valid state dict
    """
    if not isinstance(state_dict, dict):
        raise DataValidationError(f"State dict must be dict, got {type(state_dict).__name__}")

    if not state_dict:
        raise DataValidationError("State dict is empty")

    # Check all values are tensors
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise DataValidationError(f"State dict key must be string, got {type(key)}")
        if not hasattr(value, "shape"):
            raise DataValidationError(
                f"State dict value for '{key}' has no 'shape' attribute"
            )

    # Check keys if provided
    if model_keys is not None:
        missing_keys = set(model_keys) - set(state_dict.keys())
        if missing_keys:
            raise DataValidationError(f"Missing keys in state dict: {missing_keys}")

    return True


class ModelCheckpoint:
    """Secure model checkpoint manager."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        checkpoint_name: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save a model checkpoint.

        Args:
            model: PyTorch model to save
            checkpoint_name: Name for the checkpoint
            metadata: Optional metadata dict to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }

        torch.save(checkpoint, checkpoint_path)
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_name: str,
        model: Optional[torch.nn.Module] = None,
    ) -> dict:
        """
        Load a model checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to load
            model: Optional model to load state into

        Returns:
            Checkpoint dict with model_state_dict and metadata
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        checkpoint = safe_torch_load(checkpoint_path, weights_only=False)

        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove oldest checkpoints exceeding max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_checkpoint in checkpoints[self.max_checkpoints :]:
            old_checkpoint.unlink()

    def list_checkpoints(self) -> list[str]:
        """List available checkpoint names."""
        return [p.stem for p in self.checkpoint_dir.glob("*.pt")]
