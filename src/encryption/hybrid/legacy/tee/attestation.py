"""
TEE Attestation Service for HT2ML
==================================

Manages remote attestation for TEE enclaves.
Verifies enclave integrity before secure computation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import time


class AttestationStatus(Enum):
    """Attestation status."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class AttestationPolicy:
    """
    Attestation verification policy.

    Defines requirements for valid attestation.
    """
    max_age_seconds: float = 3600.0  # 1 hour
    require_nonce: bool = True
    require_measurement_match: bool = True
    allow_cached_attestation: bool = True
    cache_duration_seconds: float = 600.0  # 10 minutes

    def is_attestation_valid(self, age_seconds: float) -> bool:
        """
        Check if attestation is within validity period.

        Args:
            age_seconds: Age of attestation in seconds

        Returns:
            True if valid
        """
        return age_seconds < self.max_age_seconds

    def can_use_cached(self, cached_age_seconds: float) -> bool:
        """
        Check if cached attestation can be used.

        Args:
            cached_age_seconds: Age of cached attestation

        Returns:
            True if cache is valid
        """
        return self.allow_cached_attestation and cached_age_seconds < self.cache_duration_seconds


@dataclass
class AttestationRecord:
    """
    Record of attestation verification.

    Stores attestation results for caching.
    """
    enclave_id: str
    measurement: bytes
    nonce: bytes
    status: AttestationStatus
    timestamp: float
    report_data: Dict[str, Any] = field(default_factory=dict)

    def get_age_seconds(self) -> float:
        """Get age of attestation in seconds."""
        return time.time() - self.timestamp

    def is_valid(self, policy: AttestationPolicy) -> bool:
        """
        Check if attestation is valid according to policy.

        Args:
            policy: Attestation policy

        Returns:
            True if valid
        """
        if self.status != AttestationStatus.VALID:
            return False

        if not policy.is_attestation_valid(self.get_age_seconds()):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enclave_id': self.enclave_id,
            'measurement': self.measurement.hex(),
            'nonce': self.nonce.hex(),
            'status': self.status.value,
            'timestamp': self.timestamp,
            'age_seconds': self.get_age_seconds(),
        }


class AttestationService:
    """
    Service for managing TEE attestation.

    Handles attestation generation, verification, and caching.
    """

    def __init__(self, policy: Optional[AttestationPolicy] = None):
        """
        Initialize attestation service.

        Args:
            policy: Attestation policy (uses default if None)
        """
        self.policy = policy or AttestationPolicy()
        self.attestation_cache: Dict[str, AttestationRecord] = {}
        self.expected_measurements: Dict[str, bytes] = {}

    def register_expected_measurement(
        self,
        enclave_id: str,
        measurement: bytes
    ) -> None:
        """
        Register expected measurement for an enclave.

        Args:
            enclave_id: Enclave identifier
            measurement: Expected measurement hash
        """
        self.expected_measurements[enclave_id] = measurement
        print(f"Registered expected measurement for {enclave_id}: {measurement.hex()[:40]}...")

    def generate_challenge(self) -> bytes:
        """
        Generate attestation challenge (nonce).

        Returns:
            Fresh nonce for attestation
        """
        nonce = secrets.token_bytes(32)
        return nonce

    def verify_attestation(
        self,
        report: Dict[str, Any],
        enclave_id: str,
        expected_measurement: Optional[bytes] = None
    ) -> AttestationStatus:
        """
        Verify attestation report.

        Args:
            report: Attestation report dictionary
            enclave_id: Enclave identifier
            expected_measurement: Expected measurement (uses registered if None)

        Returns:
            AttestationStatus

        Raises:
            ValueError: If report is malformed
        """
        try:
            # Extract report fields
            report_data = bytes.fromhex(report.get('report_data', ''))
            nonce = bytes.fromhex(report.get('nonce', ''))
            signature = bytes.fromhex(report.get('signature', ''))
            measurement = bytes.fromhex(report.get('enclave_measurement', ''))
            timestamp = report.get('timestamp', 0.0)

            # Check timestamp (not too old)
            age_seconds = time.time() - timestamp
            if age_seconds > self.policy.max_age_seconds:
                print(f"Attestation expired: {age_seconds:.1f}s old")
                return AttestationStatus.EXPIRED

            # Check nonce (must be present if required)
            if self.policy.require_nonce and not nonce:
                print("Attestation missing required nonce")
                return AttestationStatus.INVALID

            # Check measurement matches expected
            if self.policy.require_measurement_match:
                if expected_measurement is None:
                    expected_measurement = self.expected_measurements.get(enclave_id)

                if expected_measurement is None:
                    print(f"No expected measurement for {enclave_id}")
                    return AttestationStatus.INVALID

                if measurement != expected_measurement:
                    print(
                        f"Measurement mismatch for {enclave_id}.\n"
                        f"Expected: {expected_measurement.hex()[:40]}...\n"
                        f"Got: {measurement.hex()[:40]}..."
                    )
                    return AttestationStatus.INVALID

            # In production, would verify TEE signature here
            # For simulation, we've done basic checks

            # Create attestation record
            record = AttestationRecord(
                enclave_id=enclave_id,
                measurement=measurement,
                nonce=nonce,
                status=AttestationStatus.VALID,
                timestamp=timestamp,
                report_data=report
            )

            # Cache attestation
            self.attestation_cache[enclave_id] = record

            print(f"Attestation verified for {enclave_id}")
            return AttestationStatus.VALID

        except Exception as e:
            print(f"Attestation verification error: {e}")
            return AttestationStatus.INVALID

    def get_cached_attestation(
        self,
        enclave_id: str
    ) -> Optional[AttestationRecord]:
        """
        Get cached attestation for enclave.

        Args:
            enclave_id: Enclave identifier

        Returns:
            Cached attestation record if valid, None otherwise
        """
        record = self.attestation_cache.get(enclave_id)

        if record is None:
            return None

        # Check if cached attestation is still valid
        if not record.is_valid(self.policy):
            # Remove stale cache entry
            del self.attestation_cache[enclave_id]
            return None

        return record

    def verify_or_use_cached(
        self,
        report: Optional[Dict[str, Any]],
        enclave_id: str,
        expected_measurement: Optional[bytes] = None
    ) -> AttestationStatus:
        """
        Verify attestation or use cached if valid.

        Args:
            report: New attestation report (None to use cache only)
            enclave_id: Enclave identifier
            expected_measurement: Expected measurement

        Returns:
            AttestationStatus
        """
        # Try to use cached attestation
        cached = self.get_cached_attestation(enclave_id)

        if cached is not None:
            print(f"Using cached attestation for {enclave_id} (age: {cached.get_age_seconds():.1f}s)")
            return cached.status

        # No valid cache, verify new report
        if report is None:
            print(f"No cached attestation and no new report for {enclave_id}")
            return AttestationStatus.INVALID

        return self.verify_attestation(report, enclave_id, expected_measurement)

    def revoke_attestation(self, enclave_id: str) -> None:
        """
        Revoke attestation for an enclave.

        Args:
            enclave_id: Enclave identifier
        """
        if enclave_id in self.attestation_cache:
            record = self.attestation_cache[enclave_id]
            record.status = AttestationStatus.REVOKED
            del self.attestation_cache[enclave_id]
            print(f"Revoked attestation for {enclave_id}")

    def clear_cache(self) -> None:
        """Clear attestation cache."""
        self.attestation_cache.clear()
        print("Attestation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        valid_count = sum(
            1 for record in self.attestation_cache.values()
            if record.status == AttestationStatus.VALID
        )

        return {
            'cached_attestations': len(self.attestation_cache),
            'valid_attestations': valid_count,
            'registered_measurements': len(self.expected_measurements),
        }


def create_attestation_service(
    policy: Optional[AttestationPolicy] = None
) -> AttestationService:
    """
    Factory function to create attestation service.

    Args:
        policy: Attestation policy (uses default if None)

    Returns:
        AttestationService instance
    """
    return AttestationService(policy)


def verify_enclave_measurement(
    report: Dict[str, Any],
    expected_measurement: bytes
) -> bool:
    """
    Verify enclave measurement in attestation report.

    Args:
        report: Attestation report
        expected_measurement: Expected measurement hash

    Returns:
        True if measurement matches
    """
    measurement_hex = report.get('enclave_measurement', '')

    if not measurement_hex:
        return False

    measurement = bytes.fromhex(measurement_hex)
    return measurement == expected_measurement


def create_default_policy() -> AttestationPolicy:
    """
    Create default attestation policy.

    Returns:
        AttestationPolicy with default settings
    """
    return AttestationPolicy()
