"""
FedPhish Input Validation

Common validation functions and decorators for all projects.
"""

import re
from typing import Any, Callable, Optional, Union, TypeVar
from functools import wraps
from pathlib import Path

from .errors import DataValidationError, ConfigError
from .types import ModelUpdate, TensorOrArray


T = TypeVar("T")


# Email validation patterns
EMAIL_ADDRESS_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)
PHISHING_INDICATORS = [
    "urgent", "verify", "account", "suspended", "click here",
    "password", "security", "update", "immediate", "confirm",
]


def validate_email_input(email: Union[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Validate email input for phishing detection.

    Args:
        email: Either a raw email string or a dictionary with email fields

    Returns:
        Validated email dictionary

    Raises:
        DataValidationError: If email format is invalid
    """
    if isinstance(email, str):
        if len(email.strip()) == 0:
            raise DataValidationError("Email content is empty")
        if len(email) > 10_000_000:  # 10MB limit
            raise DataValidationError("Email content too large")
        return {"content": email, "raw": email}

    elif isinstance(email, dict):
        if not email:
            raise DataValidationError("Email dictionary is empty")

        # Check for required fields
        if "content" not in email and "body" not in email and "raw" not in email:
            raise DataValidationError(
                "Email dict must contain 'content', 'body', or 'raw' field"
            )

        # Validate string fields
        for key in ["subject", "from", "to", "content", "body", "raw"]:
            if key in email:
                if not isinstance(email[key], str):
                    raise DataValidationError(f"Email field '{key}' must be a string")
                if len(email[key]) > 1_000_000:  # 1MB per field limit
                    raise DataValidationError(f"Email field '{key}' too large")

        return email

    else:
        raise DataValidationError(
            f"Email must be string or dict, got {type(email).__name__}"
        )


def validate_model_update(update: ModelUpdate) -> ModelUpdate:
    """
    Validate a model update for federated aggregation.

    Args:
        update: Dictionary mapping layer names to tensors/arrays

    Returns:
        The validated update

    Raises:
        DataValidationError: If update format is invalid
    """
    if not isinstance(update, dict):
        raise DataValidationError(
            f"Model update must be dict, got {type(update).__name__}"
        )

    if not update:
        raise DataValidationError("Model update is empty")

    for layer_name, tensor in update.items():
        if not isinstance(layer_name, str):
            raise DataValidationError(f"Layer name must be string, got {type(layer_name)}")

        # Check tensor has expected properties
        if not hasattr(tensor, "shape"):
            raise DataValidationError(f"Layer '{layer_name}' has no 'shape' attribute")

    return update


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """
    Validate configuration against a schema.

    Args:
        config: Configuration dictionary
        schema: Schema dict with keys mapping to (type, required, default)

    Returns:
        Validated and normalized config

    Raises:
        ConfigError: If config is invalid
    """
    result = {}

    for key, (expected_type, required, default) in schema.items():
        if key not in config:
            if required:
                raise ConfigError(f"Missing required config key: '{key}'")
            result[key] = default
        else:
            value = config[key]
            if not isinstance(value, expected_type):
                try:
                    # Try to coerce type
                    if expected_type == int:
                        value = int(value)
                    elif expected_type == float:
                        value = float(value)
                    elif expected_type == bool:
                        value = bool(value)
                    elif expected_type == str:
                        value = str(value)
                    else:
                        raise ConfigError(
                            f"Config key '{key}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                except (ValueError, TypeError):
                    raise ConfigError(
                        f"Config key '{key}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
            result[key] = value

    # Check for unknown keys (optional, can be disabled)
    unknown_keys = set(config.keys()) - set(schema.keys())
    if unknown_keys:
        # Log warning but don't fail
        pass

    return result


def validate_positive_number(
    value: Union[int, float],
    name: str = "value",
    allow_zero: bool = False,
) -> Union[int, float]:
    """
    Validate that a number is positive (and optionally, non-zero).

    Args:
        value: Number to validate
        name: Name for error messages
        allow_zero: If False, zero values are rejected

    Raises:
        DataValidationError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise DataValidationError(f"{name} must be a number, got {type(value).__name__}")

    if value < 0:
        raise DataValidationError(f"{name} must be non-negative, got {value}")

    if not allow_zero and value == 0:
        raise DataValidationError(f"{name} must be positive, got 0")

    return value


def validate_client_id(client_id: Any) -> str:
    """
    Validate and normalize a client ID.

    Args:
        client_id: Client identifier to validate

    Returns:
        Normalized string client ID

    Raises:
        DataValidationError: If client_id is invalid
    """
    if client_id is None:
        raise DataValidationError("Client ID cannot be None")

    return str(client_id)


def validate_batch_size(size: int, max_size: int = 1000) -> int:
    """
    Validate batch size for processing.

    Args:
        size: Batch size to validate
        max_size: Maximum allowed batch size

    Returns:
        Validated batch size

    Raises:
        DataValidationError: If size is invalid
    """
    if not isinstance(size, int):
        raise DataValidationError(f"Batch size must be int, got {type(size).__name__}")

    if size <= 0:
        raise DataValidationError(f"Batch size must be positive, got {size}")

    if size > max_size:
        raise DataValidationError(f"Batch size exceeds maximum ({max_size}), got {size}")

    return size


# Decorator for input validation
def validate_inputs(**validators: Callable[[Any], Any]):
    """
    Decorator to validate function inputs.

    Usage:
        @validate_inputs(email=validate_email_input, threshold=validate_positive_number)
        def analyze_email(email, threshold=0.5):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    bound_args.arguments[param_name] = validator(
                        bound_args.arguments[param_name]
                    )

            return func(**bound_args.arguments)
        return wrapper
    return decorator


def validate_url(url: str) -> str:
    """
    Validate a URL string.

    Args:
        url: URL to validate

    Returns:
        Normalized URL

    Raises:
        DataValidationError: If URL is invalid
    """
    if not isinstance(url, str):
        raise DataValidationError(f"URL must be string, got {type(url).__name__}")

    url = url.strip()

    if not url:
        raise DataValidationError("URL cannot be empty")

    # Basic URL pattern check
    if not re.match(r"^https?://", url):
        raise DataValidationError(f"URL must start with http:// or https://, got: {url[:50]}")

    if len(url) > 2048:
        raise DataValidationError(f"URL too long (max 2048 chars)")

    return url
