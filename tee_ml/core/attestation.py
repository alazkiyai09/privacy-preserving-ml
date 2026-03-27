"""
Remote Attestation Simulation
=============================

Simulates remote attestation for TEE integrity verification.

In real Intel SGX:
1. Enclave generates quote (signed report with measurement)
2. Verifier checks quote against expected measurement
3. Verifier checks signature using Intel's attestation service (IAS)
4. Attestation proves enclave is running correct code

This simulation:
- Generates signed reports with measurement
- Verifies reports against expected measurement
- Simulates network attestation protocol
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import hmac
import json
import uuid

from tee_ml.core.enclave import Enclave


class AttestationError(Exception):
    """Exception raised for attestation failures."""
    pass


@dataclass
class AttestationReport:
    """
    Attestation report for enclave integrity verification.

    Contains:
    - enclave_id: Identifier for the enclave
    - measurement: Hash of enclave code (MRENCLAVE)
    - nonce: Random value to prevent replay attacks
    - timestamp: When report was generated
    - signature: HMAC signature for authenticity
    """

    enclave_id: str
    measurement: bytes
    nonce: bytes
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    signature: bytes = b""
    user_data: bytes = b""

    def to_dict(self) -> Dict[str, any]:
        """Convert report to dictionary."""
        return {
            "enclave_id": self.enclave_id,
            "measurement": self.measurement.hex(),
            "nonce": self.nonce.hex(),
            "timestamp": self.timestamp,
            "signature": self.signature.hex(),
            "user_data": self.user_data.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'AttestationReport':
        """Create report from dictionary."""
        return cls(
            enclave_id=data["enclave_id"],
            measurement=bytes.fromhex(data["measurement"]),
            nonce=bytes.fromhex(data["nonce"]),
            timestamp=data["timestamp"],
            signature=bytes.fromhex(data["signature"]),
            user_data=bytes.fromhex(data.get("user_data", "")),
        )

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'AttestationReport':
        """Create report from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AttestationResult:
    """Result of attestation verification."""
    valid: bool
    reason: str
    enclave_id: str
    measurement_match: bool
    signature_valid: bool
    timestamp_recent: bool


class AttestationService:
    """
    Service for generating and verifying attestation reports.

    Simulates Intel Attestation Service (IAS) for SGX.

    In production:
    - Use real IAS for SGX
    - Use PSA Attestation for ARM TrustZone
    - Use Nitro Attestation for AWS Nitro Enclaves

    This simulation:
    - Generates HMAC-signed reports
    - Verifies measurements against expected values
    - Checks timestamps for freshness
    """

    def __init__(self, signing_key: bytes = None):
        """
        Initialize attestation service.

        Args:
            signing_key: Key for signing reports (auto-generated if None)
        """
        self.signing_key = signing_key or hashlib.sha256(b"attestation_key").digest()

        # Store expected measurements for enclaves
        self._expected_measurements: Dict[str, bytes] = {}

        # Attestation history for debugging
        self._attestation_history: List[AttestationReport] = []

    def register_enclave(self, enclave_id: str, measurement: bytes) -> None:
        """
        Register expected measurement for an enclave.

        In real attestation, this would come from a trusted source
        (e.g., developer, secure build system).

        Args:
            enclave_id: Enclave identifier
            measurement: Expected measurement hash
        """
        self._expected_measurements[enclave_id] = measurement

    def generate_nonce(self) -> bytes:
        """
        Generate random nonce for attestation.

        Nonce prevents replay attacks.

        Returns:
            Random 16-byte nonce
        """
        return uuid.uuid4().bytes

    def generate_report(
        self,
        enclave: Enclave,
        nonce: bytes = None,
        user_data: bytes = b"",
    ) -> AttestationReport:
        """
        Generate attestation report for enclave.

        Args:
            enclave: Enclave to attest
            nonce: Random nonce (auto-generated if None)
            user_data: Optional user data to include in report

        Returns:
            Signed attestation report
        """
        if nonce is None:
            nonce = self.generate_nonce()

        # Get measurement from enclave
        measurement = enclave.get_measurement()

        # Create report
        report = AttestationReport(
            enclave_id=enclave.enclave_id,
            measurement=measurement,
            nonce=nonce,
            user_data=user_data,
        )

        # Sign report
        report.signature = self._sign_report(report)

        # Store in history
        self._attestation_history.append(report)

        return report

    def _sign_report(self, report: AttestationReport) -> bytes:
        """
        Sign attestation report with HMAC.

        Args:
            report: Report to sign

        Returns:
            HMAC signature
        """
        # Create message to sign
        message = (
            report.enclave_id.encode() +
            report.measurement +
            report.nonce +
            str(report.timestamp).encode() +
            report.user_data
        )

        # Compute HMAC
        signature = hmac.new(
            self.signing_key,
            message,
            hashlib.sha256
        ).digest()

        return signature

    def verify_report(
        self,
        report: AttestationReport,
        expected_measurement: bytes = None,
        max_age_seconds: float = 300,  # 5 minutes
    ) -> AttestationResult:
        """
        Verify attestation report.

        Args:
            report: Report to verify
            expected_measurement: Expected measurement (overrides registered)
            max_age_seconds: Maximum age of report (for freshness)

        Returns:
            AttestationResult with verification details
        """
        # Get expected measurement
        if expected_measurement is None:
            expected_measurement = self._expected_measurements.get(
                report.enclave_id,
                None
            )

        if expected_measurement is None:
            return AttestationResult(
                valid=False,
                reason=f"No expected measurement registered for {report.enclave_id}",
                enclave_id=report.enclave_id,
                measurement_match=False,
                signature_valid=False,
                timestamp_recent=False,
            )

        # Verify measurement
        measurement_match = (report.measurement == expected_measurement)

        # Verify signature
        signature_valid = self._verify_signature(report)

        # Verify timestamp (freshness check)
        age = datetime.now().timestamp() - report.timestamp
        timestamp_recent = age <= max_age_seconds

        # Overall validity
        valid = measurement_match and signature_valid and timestamp_recent

        # Build reason
        reasons = []
        if not measurement_match:
            reasons.append("measurement mismatch")
        if not signature_valid:
            reasons.append("invalid signature")
        if not timestamp_recent:
            reasons.append(f"report too old ({age:.1f}s)")

        reason = "; ".join(reasons) if reasons else "valid"

        return AttestationResult(
            valid=valid,
            reason=reason,
            enclave_id=report.enclave_id,
            measurement_match=measurement_match,
            signature_valid=signature_valid,
            timestamp_recent=timestamp_recent,
        )

    def _verify_signature(self, report: AttestationReport) -> bool:
        """
        Verify HMAC signature on report.

        Args:
            report: Report to verify

        Returns:
            True if signature is valid
        """
        # Recreate expected signature
        expected_signature = self._sign_report(report)

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(report.signature, expected_signature)

    def remote_attestation(
        self,
        enclave: Enclave,
        challenger_id: str,
        nonce: bytes = None,
    ) -> bool:
        """
        Perform complete remote attestation protocol.

        Simulates:
        1. Challenger generates nonce
        2. Enclave generates quote with nonce
        3. Verifier checks quote
        4. Return attestation result

        Args:
            enclave: Enclave to attest
            challenger_id: ID of entity requesting attestation
            nonce: Nonce from challenger (auto-generated if None)

        Returns:
            True if attestation succeeded
        """
        # Generate nonce
        if nonce is None:
            nonce = self.generate_nonce()

        # Enclave generates report
        report = self.generate_report(enclave, nonce=nonce)

        # Verifier checks report
        result = self.verify_report(report)

        return result.valid

    def get_attestation_history(self) -> List[AttestationReport]:
        """
        Get history of attestation reports.

        Returns:
            List of all generated reports
        """
        return self._attestation_history.copy()

    def clear_history(self) -> None:
        """Clear attestation history."""
        self._attestation_history.clear()


def simulate_remote_attestation(
    enclave: Enclave,
    challenger_id: str = "remote_client",
) -> Dict[str, any]:
    """
    Simulate complete remote attestation flow.

    Args:
        enclave: Enclave to attest
        challenger_id: ID of challenger

    Returns:
        Dictionary with attestation details
    """
    service = AttestationService()

    # Register enclave measurement
    service.register_enclave(enclave.enclave_id, enclave.get_measurement())

    # Perform attestation
    nonce = service.generate_nonce()
    report = service.generate_report(enclave, nonce=nonce)
    result = service.verify_report(report)

    return {
        "challenger_id": challenger_id,
        "enclave_id": enclave.enclave_id,
        "nonce": nonce.hex(),
        "report": report.to_dict(),
        "result": {
            "valid": result.valid,
            "reason": result.reason,
            "measurement_match": result.measurement_match,
            "signature_valid": result.signature_valid,
            "timestamp_recent": result.timestamp_recent,
        },
    }


def create_mock_ias() -> AttestationService:
    """
    Create mock Intel Attestation Service.

    Returns:
        AttestationService instance
    """
    return AttestationService()


class EnclaveIdentity:
    """
    Identity information for enclave attestation.

    In real SGX, this includes:
    - MRENCLAVE: Measurement of enclave code
    - MRSIGNER: Measurement of signer's public key
    - Attributes: Enclave attributes (DEBUG, MODE64-bit, etc.)
    """

    def __init__(
        self,
        mrenclave: bytes,
        mrsigner: bytes = None,
        attributes: int = 0x0000000000000002,  # MODE64-bit
    ):
        """
        Initialize enclave identity.

        Args:
            mrenclave: Measurement of enclave code
            mrsigner: Measurement of signer's public key
            attributes: Enclave attributes bitmask
        """
        self.mrenclave = mrenclave
        self.mrsigner = mrsigner or hashlib.sha256(b"mock_signer").digest()
        self.attributes = attributes

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "mrenclave": self.mrenclave.hex(),
            "mrsigner": self.mrsigner.hex(),
            "attributes": f"0x{self.attributes:016x}",
        }
