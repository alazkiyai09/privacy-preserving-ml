"""
FedPhish Error Hierarchy

Base exception classes for all FedPhish projects.
Provides consistent error handling across the portfolio.
"""

from typing import Optional, Any


class FedPhishError(Exception):
    """Base exception for all FedPhish-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }
        if self.original_error:
            result["original_error"] = str(self.original_error)
            result["original_error_type"] = type(self.original_error).__name__
        return result


class ModelLoadError(FedPhishError):
    """Raised when model loading fails."""

    pass


class DataValidationError(FedPhishError):
    """Raised when input data validation fails."""

    pass


class CryptographicError(FedPhishError):
    """Raised when cryptographic operations fail."""

    pass


class PrivacyError(FedPhishError):
    """Raised when privacy guarantees cannot be met."""

    pass


class CommunicationError(FedPhishError):
    """Raised when network/communication operations fail."""

    pass


class AggregationError(FedPhishError):
    """Raised when federated aggregation fails."""

    pass


class ConfigError(FedPhishError):
    """Raised when configuration is invalid."""

    pass


class ByzantineError(FedPhishError):
    """Raised when Byzantine behavior is detected."""

    pass
