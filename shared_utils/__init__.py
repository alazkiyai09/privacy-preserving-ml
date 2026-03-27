"""
FedPhish Shared Utilities Package

Common utilities used across all FedPhish projects including:
- Error handling hierarchy
- Structured logging
- Type definitions
- Input validation
- Security utilities
"""

__version__ = "1.0.0"

from .errors import (
    FedPhishError,
    ModelLoadError,
    DataValidationError,
    CryptographicError,
    PrivacyError,
    CommunicationError,
)

from .logging import get_logger, configure_logging, StructuredLogger
from .types import (
    ModelUpdate,
    ClientId,
    RoundNum,
    TensorOrArray,
    JSONSerializable,
)
from .validation import validate_email_input, validate_model_update, validate_config
from .security import safe_torch_load, safe_pickle_load, validate_file_path

__all__ = [
    # Errors
    "FedPhishError",
    "ModelLoadError",
    "DataValidationError",
    "CryptographicError",
    "PrivacyError",
    "CommunicationError",
    # Logging
    "get_logger",
    "configure_logging",
    "StructuredLogger",
    # Types
    "ModelUpdate",
    "ClientId",
    "RoundNum",
    "TensorOrArray",
    "JSONSerializable",
    # Validation
    "validate_email_input",
    "validate_model_update",
    "validate_config",
    # Security
    "safe_torch_load",
    "safe_pickle_load",
    "validate_file_path",
]
