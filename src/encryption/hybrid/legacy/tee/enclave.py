"""
TEE Enclave Wrapper for HT2ML
==============================

Wraps TEE functionality for secure execution in trusted environment.
Uses TEE simulation from Day 7 work.
"""

from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import hashlib
import secrets


class TEEState(Enum):
    """TEE enclave states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    ATTESTED = "attested"
    ACTIVE = "active"
    TERMINATED = "terminated"


@dataclass
class TEEAttestationReport:
    """
    TEE attestation report for integrity verification.

    Contains cryptographic proof of enclave integrity.
    """
    report_data: bytes  # SHA256 hash of enclave measurement
    nonce: bytes  # Freshness guarantee
    signature: bytes  # TEE signature
    enclave_measurement: bytes  # Measurement of enclave code
    timestamp: float

    def verify(self, expected_measurement: bytes) -> bool:
        """
        Verify attestation report.

        Args:
            expected_measurement: Expected enclave measurement

        Returns:
            True if attestation is valid
        """
        # In production, would verify TEE signature
        # For simulation, check measurement matches

        return self.enclave_measurement == expected_measurement

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_data': self.report_data.hex(),
            'nonce': self.nonce.hex(),
            'signature': self.signature.hex(),
            'enclave_measurement': self.enclave_measurement.hex(),
            'timestamp': self.timestamp,
        }


@dataclass
class TEEContext:
    """
    TEE execution context.

    Contains enclave state and attestation information.
    """
    enclave_id: str
    state: TEEState
    measurement: bytes  # SHA256 of enclave code
    attestation_report: Optional[TEEAttestationReport] = None
    loaded_model: Optional[Dict[str, np.ndarray]] = None

    def is_attested(self) -> bool:
        """Check if enclave is attested."""
        return (
            self.state == TEEState.ATTESTED or
            self.state == TEEState.ACTIVE
        ) and self.attestation_report is not None

    def can_execute(self) -> bool:
        """Check if enclave can execute operations."""
        return self.state == TEEState.ACTIVE and self.loaded_model is not None


class TEEEnclave:
    """
    TEE enclave wrapper for HT2ML.

    Manages secure enclave lifecycle for TEE operations.
    """

    def __init__(self, enclave_id: Optional[str] = None):
        """
        Initialize TEE enclave.

        Args:
            enclave_id: Optional enclave identifier
        """
        self.enclave_id = enclave_id or f"enclave_{secrets.token_urlsafe(16)}"
        self.context = TEEContext(
            enclave_id=self.enclave_id,
            state=TEEState.UNINITIALIZED,
            measurement=self._calculate_enclave_measurement()
        )

    def _calculate_enclave_measurement(self) -> bytes:
        """
        Calculate measurement of enclave code.

        Returns:
            SHA256 hash of enclave measurement

        Note:
            In production, this would hash actual enclave code.
            For simulation, uses a fixed measurement.
        """
        # Simulated measurement of TEE code
        measurement_data = f"TEE_ENCLAVE_{self.enclave_id}".encode()
        return hashlib.sha256(measurement_data).digest()

    def initialize(self) -> None:
        """
        Initialize TEE enclave.

        Sets up secure execution environment.

        Raises:
            RuntimeError: If initialization fails
        """
        if self.context.state != TEEState.UNINITIALIZED:
            raise RuntimeError(
                f"Enclave already initialized (state: {self.context.state})"
            )

        # In production, would initialize actual TEE (Intel SGX, etc.)
        # For simulation, just update state

        self.context.state = TEEState.INITIALIZED

        print(f"TEE Enclave initialized: {self.enclave_id}")
        print(f"Measurement: {self.context.measurement.hex()[:40]}...")

    def generate_attestation(
        self,
        nonce: Optional[bytes] = None
    ) -> TEEAttestationReport:
        """
        Generate attestation report.

        Args:
            nonce: Freshness nonce (generated if None)

        Returns:
            Attestation report

        Raises:
            RuntimeError: If enclave not initialized
        """
        if self.context.state not in [TEEState.INITIALIZED, TEEState.ATTESTED, TEEState.ACTIVE]:
            raise RuntimeError(
                f"Cannot generate attestation in state: {self.context.state}"
            )

        if nonce is None:
            nonce = secrets.token_bytes(32)

        import time

        # Create attestation report
        report = TEEAttestationReport(
            report_data=hashlib.sha256(self.context.measurement + nonce).digest(),
            nonce=nonce,
            signature=f"SIGNURE_{secrets.token_hex(32)}".encode(),  # Placeholder
            enclave_measurement=self.context.measurement,
            timestamp=time.time()
        )

        self.context.attestation_report = report
        self.context.state = TEEState.ATTESTED

        return report

    def load_model(self, model_weights: Dict[str, np.ndarray], require_attestation: bool = True) -> None:
        """
        Load model weights into enclave.

        Args:
            model_weights: Dictionary with layer weights
            require_attestation: If True, require attestation before loading (default: True)

        Raises:
            RuntimeError: If enclave not attested and require_attestation is True
        """
        if require_attestation and not self.context.is_attested():
            raise RuntimeError(
                "Cannot load model: enclave not attested. "
                "Call generate_attestation() first."
            )

        # Load model into secure memory
        self.context.loaded_model = model_weights

        # Only set to ACTIVE if attested (otherwise stays INITIALIZED)
        if self.context.is_attested():
            self.context.state = TEEState.ACTIVE

        print(f"Model loaded into TEE enclave")
        print(f"Layers: {list(model_weights.keys())}")

    def execute_operation(
        self,
        operation: str,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Execute operation in TEE.

        Args:
            operation: Operation name (relu, softmax, argmax)
            data: Input data

        Returns:
            Output data

        Raises:
            RuntimeError: If enclave cannot execute
            ValueError: If operation unknown
        """
        if not self.context.can_execute():
            raise RuntimeError(
                "Cannot execute: enclave not active or no model loaded"
            )

        # Execute operation in secure environment
        if operation == "relu":
            return np.maximum(0, data)
        elif operation == "softmax":
            # Stable softmax
            exp_x = np.exp(data - np.max(data))
            return exp_x / exp_x.sum()
        elif operation == "argmax":
            return np.argmax(data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_measurement(self) -> bytes:
        """Get enclave measurement."""
        return self.context.measurement

    def get_state(self) -> TEEState:
        """Get enclave state."""
        return self.context.state

    def terminate(self) -> None:
        """Terminate enclave."""
        self.context.state = TEEState.TERMINATED
        self.context.loaded_model = None
        print(f"TEE Enclave terminated: {self.enclave_id}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get enclave status.

        Returns:
            Dictionary with enclave status
        """
        return {
            'enclave_id': self.enclave_id,
            'state': self.context.state.value,
            'measurement': self.context.measurement.hex(),
            'attested': self.context.is_attested(),
            'active': self.context.can_execute(),
            'has_model': self.context.loaded_model is not None,
        }


def create_tee_enclave(enclave_id: Optional[str] = None) -> TEEEnclave:
    """
    Factory function to create TEE enclave.

    Args:
        enclave_id: Optional enclave identifier

    Returns:
        TEEEnclave instance
    """
    enclave = TEEEnclave(enclave_id)
    enclave.initialize()
    return enclave


def verify_attestation_report(
    report: TEEAttestationReport,
    expected_measurement: bytes
) -> bool:
    """
    Verify attestation report.

    Args:
        report: Attestation report to verify
        expected_measurement: Expected enclave measurement

    Returns:
        True if report is valid
    """
    return report.verify(expected_measurement)
