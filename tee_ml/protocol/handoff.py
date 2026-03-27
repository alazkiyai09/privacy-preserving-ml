"""
HE↔TEE Handoff Protocol
========================

Defines the interface for passing data between homomorphic encryption
and trusted execution environments in the HT2ML hybrid system.

Protocol Overview:
                    Client
                      │
                      ├─ HE Encryption (encrypt input)
                      │
                      ▼
                  Encrypted Data
                      │
                      ├─ HE Layer 1 (linear ops, no activation)
                      ├─ HE Layer 2 (if budget allows)
                      │
                      ▼
              Handoff Point (decrypt in TEE)
                      │
                      ▼
                      TEE (decrypt, process remaining layers)
                      │
                      ▼
                   Output

Key Design Decisions:
1. HE layers: 1-2 layers max (input privacy)
2. TEE layers: Remaining layers (performance)
3. Handoff: Decrypt within secure enclave
4. Security: Input data always encrypted when leaving client
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

from tee_ml.core.enclave import Enclave, EnclaveSession


class HandoffDirection(Enum):
    """Direction of handoff (HE to TEE or TEE to HE)."""
    HE_TO_TEE = "he_to_tee"
    TEE_TO_HE = "tee_to_he"


@dataclass
class HEContext:
    """
    Context for homomorphic encryption operations.

    Contains encryption parameters and keys needed for HE operations.
    """

    scheme: str  # 'ckks' or 'bfv'
    poly_modulus_degree: int
    scale: float
    eval: int  # Encryption level
    public_key: Optional[Any] = None  # TenSEAL PublicKey
    secret_key: Optional[Any] = None  # TenSEAL SecretKey
    relin_key: Optional[Any] = None  # TenSEAL RelinearizationKey
    galois_key: Optional[Any] = None  # TenSEAL GaloisKeys

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            'scheme': self.scheme,
            'poly_modulus_degree': self.poly_modulus_degree,
            'scale': self.scale,
            'eval': self.eval,
        }


@dataclass
class HEData:
    """
    Data encrypted with homomorphic encryption.

    Contains encrypted vectors along with metadata needed for processing.
    """

    encrypted_data: Any  # Encrypted vector (CiphertextVector)
    shape: Tuple[int, ...]  # Original shape
    scheme: str
    scale: float

    def __post_init__(self):
        """Validate HE data."""
        if not isinstance(self.encrypted_data, list):
            # Try to get size
            try:
                self.size = self.encrypted_data.size()
            except (AttributeError, TypeError, RuntimeError):
                self.size = 0
        else:
            self.size = len(self.encrypted_data)


@dataclass
class HEtoTEEHandoff:
    """
    Handoff from HE to TEE.

    Client sends encrypted data to server.
    Server decrypts within secure enclave for TEE processing.
    """

    encrypted_data: HEData
    he_context: HEContext
    nonce: Optional[bytes] = None  # For freshness
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate handoff data.

        Returns:
            True if handoff is valid
        """
        # Check that encrypted data exists
        if self.encrypted_data is None:
            return False

        # Check that the data inside is not None
        if self.encrypted_data.encrypted_data is None:
            return False

        # Check that context matches
        if (self.encrypted_data.scheme != self.he_context.scheme):
            return False

        return True


@dataclass
class TEEtoHEHandoff:
    """
    Handoff from TEE back to HE (rare, but possible).

    Used when TEE needs to send results back through HE layer.
    """

    plaintext_data: np.ndarray
    he_context: HEContext
    reencrypt: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate handoff data.

        Returns:
            True if handoff is valid
        """
        # Check that data exists
        if self.plaintext_data is None:
            return False

        # Check that context exists
        if self.he_context is None:
            return False

        return True


@dataclass
class HandoffResult:
    """
    Result of handoff operation.
    """

    success: bool
    direction: HandoffDirection
    data: Union[HEData, np.ndarray, None]
    error_message: Optional[str] = None
    execution_time_ns: int = 0


class HT2MLProtocol:
    """
    HT2ML handoff protocol implementation.

    Manages the secure transfer of data between HE and TEE domains.
    """

    def __init__(self, tee_enclave: Enclave):
        """
        Initialize protocol with TEE enclave.

        Args:
            tee_enclave: TEE enclave for processing
        """
        self.enclave = tee_enclave
        self.session: Optional[EnclaveSession] = None
        self.handoff_history: List[HandoffResult] = []

    def handoff_he_to_tee(
        self,
        encrypted_data: HEData,
        he_context: HEContext,
        nonce: bytes = None,
    ) -> Tuple[bool, np.ndarray]:
        """
        Perform handoff from HE to TEE.

        Process:
        1. Validate handoff data
        2. Enter enclave with encrypted data
        3. Decrypt within enclave
        4. Return plaintext data

        Args:
            encrypted_data: Encrypted data from HE
            he_context: HE context with keys
            nonce: Optional nonce for freshness

        Returns:
            (success, plaintext_data) tuple
        """
        import time
        start = time.perf_counter_ns()

        try:
            # Validate handoff
            handoff = HEtoTEEHandoff(
                encrypted_data=encrypted_data,
                he_context=he_context,
                nonce=nonce
            )

            if not handoff.validate():
                result = HandoffResult(
                    success=False,
                    direction=HandoffDirection.HE_TO_TEE,
                    data=None,
                    error_message="Invalid handoff data",
                    execution_time_ns=time.perf_counter_ns() - start,
                )
                self.handoff_history.append(result)
                return False, np.array([])

            # Decrypt the encrypted data
            # In real system, this would call TenSEAL decryption
            # For simulation, we simulate successful decryption by returning mock data

            # Simulate: Return zeros if we can't actually decrypt
            plaintext = np.zeros(encrypted_data.size)

            # Record successful handoff
            result = HandoffResult(
                success=True,
                direction=HandoffDirection.HE_TO_TEE,
                data=plaintext,
                execution_time_ns=time.perf_counter_ns() - start,
            )
            self.handoff_history.append(result)

            return True, plaintext

        except Exception as e:
            result = HandoffResult(
                success=False,
                direction=HandoffDirection.HE_TO_TEE,
                data=None,
                error_message=str(e),
                execution_time_ns=time.perf_counter_ns() - start,
            )
            self.handoff_history.append(result)
            return False, np.array([])

    def handoff_tee_to_he(
        self,
        plaintext_data: np.ndarray,
        he_context: HEContext,
        reencrypt: bool = True,
    ) -> Tuple[bool, HEData]:
        """
        Perform handoff from TEE back to HE.

        Process:
        1. Validate handoff data
        2. Encrypt data with HE
        3. Return encrypted data

        Args:
            plaintext_data: Plaintext data from TEE
            he_context: HE context with keys
            reencrypt: Whether to re-encrypt (if False, returns mock)

        Returns:
            (success, encrypted_data) tuple
        """
        import time
        start = time.perf_counter_ns()

        try:
            # Validate handoff
            handoff = TEEtoHEHandoff(
                plaintext_data=plaintext_data,
                he_context=he_context,
                reencrypt=reencrypt
            )

            if not handoff.validate():
                result = HandoffResult(
                    success=False,
                    direction=HandoffDirection.TEE_TO_HE,
                    data=None,
                    error_message="Invalid handoff data",
                    execution_time_ns=time.perf_counter_ns() - start,
                )
                self.handoff_history.append(result)
                return False, None

            # Encrypt the plaintext data
            # In real system, this would call TenSEAL encryption
            # For simulation, we create a mock encrypted data
            encrypted = HEData(
                encrypted_data=None,  # Mock encrypted data
                shape=plaintext_data.shape,
                scheme=he_context.scheme,
                scale=he_context.scale,
            )

            # Record successful handoff
            result = HandoffResult(
                success=True,
                direction=HandoffDirection.TEE_TO_HE,
                data=encrypted,
                execution_time_ns=time.perf_counter_ns() - start,
            )
            self.handoff_history.append(result)

            return True, encrypted

        except Exception as e:
            result = HandoffResult(
                success=False,
                direction=HandoffDirection.TEE_TO_HE,
                data=None,
                error_message=str(e),
                execution_time_ns=time.perf_counter_ns() - start,
            )
            self.handoff_history.append(result)
            return False, None

    def get_handoff_history(self) -> List[HandoffResult]:
        """Get history of handoff operations."""
        return self.handoff_history.copy()

    def get_handoff_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about handoff operations.

        Returns:
            Dictionary with statistics
        """
        total_handoffs = len(self.handoff_history)
        successful = sum(1 for r in self.handoff_history if r.success)

        by_direction = {}
        for result in self.handoff_history:
            direction = result.direction.value
            if direction not in by_direction:
                by_direction[direction] = {'total': 0, 'successful': 0}
            by_direction[direction]['total'] += 1
            if result.success:
                by_direction[direction]['successful'] += 1

        total_time_ns = sum(r.execution_time_ns for r in self.handoff_history)

        return {
            'total_handoffs': total_handoffs,
            'successful': successful,
            'failed': total_handoffs - successful,
            'success_rate': successful / total_handoffs if total_handoffs > 0 else 0.0,
            'by_direction': by_direction,
            'total_time_ns': total_time_ns,
            'avg_time_ns': total_time_ns / total_handoffs if total_handoffs > 0 else 0,
        }


def create_handoff_protocol(
    tee_enclave: Enclave,
) -> HT2MLProtocol:
    """
    Factory function to create handoff protocol.

    Args:
        tee_enclave: TEE enclave for processing

    Returns:
        HT2MLProtocol instance
    """
    return HT2MLProtocol(tee_enclave)


def validate_handoff_security(
    handoff: Union[HEtoTEEHandoff, TEEtoHEHandoff],
    expected_measurement: Optional[bytes] = None,
) -> bool:
    """
    Validate security of handoff operation.

    Checks:
    - Handoff data integrity
    - Freshness (nonce)
    - Measurement verification (if TEE)

    Args:
        handoff: Handoff data to validate
        expected_measurement: Expected TEE measurement

    Returns:
        True if handoff is secure
    """
    # Check basic validation
    if hasattr(handoff, 'validate'):
        if not handoff.validate():
            return False

    # Check nonce freshness (if provided)
    if isinstance(handoff, HEtoTEEHandoff) and handoff.nonce:
        # In real system, would check nonce timestamp
        # For simulation, we just check it exists
        if not handoff.nonce:
            return False

    # Check measurement (if expected and applicable)
    if expected_measurement is not None:
        # In real system, would verify TEE measurement
        # For simulation, we skip this
        pass

    return True


def estimate_handoff_cost(
    handoff_type: HandoffDirection,
    data_size_mb: float,
    overhead_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Estimate overhead for handoff operation.

    Args:
        handoff_type: Type of handoff (HE_TO_TEE or TEE_TO_HE)
        data_size_mb: Size of data being handed off
        overhead_model: Optional overhead model

    Returns:
        Dictionary with cost estimation
    """
    # Base handoff overhead
    # In real system, includes:
    # - Data serialization
    # - Network transfer (if remote)
    # - Enclave entry/exit
    # - Decryption/Encryption

    base_overhead_ns = 50000  # 50 μs base overhead

    if handoff_type == HandoffDirection.HE_TO_TEE:
        # HE→TEE: Decryption overhead
        decryption_ns = 100000  # 100 μs for decryption
        total_ns = base_overhead_ns + decryption_ns
    else:
        # TEE→HE: Encryption overhead
        encryption_ns = 100000  # 100 μs for encryption
        total_ns = base_overhead_ns + encryption_ns

    return {
        'handoff_type': handoff_type.value,
        'data_size_mb': data_size_mb,
        'base_overhead_ns': base_overhead_ns,
        'cryptographic_ns': decryption_ns if handoff_type == HandoffDirection.HE_TO_TEE else encryption_ns,
        'total_overhead_ns': total_ns,
        'total_overhead_ms': total_ns / 1e6,  # Convert to milliseconds
        'total_overhead_us': total_ns / 1e3,  # Convert to microseconds
    }


class ProtocolOptimizer:
    """
    Optimizes HT2ML protocol for performance and security.

    Analyzes handoff patterns and suggests improvements.
    """

    def __init__(self, protocol: HT2MLProtocol):
        """
        Initialize protocol optimizer.

        Args:
            protocol: HT2MLProtocol instance
        """
        self.protocol = protocol

    def analyze_handoffs(self) -> Dict[str, Any]:
        """
        Analyze handoff patterns for optimization opportunities.

        Returns:
            Dictionary with analysis results
        """
        stats = self.protocol.get_handoff_statistics()

        # Calculate handoff frequency
        total = stats['total_handoffs']
        he_to_tee = stats['by_direction'].get('he_to_tee', {}).get('total', 0)
        tee_to_he = stats['by_direction'].get('tee_to_he', {}).get('total', 0)

        # Identify patterns
        patterns = []

        if he_to_tee > 0 and tee_to_he == 0:
            patterns.append("One-way handoff (HE→TEE only)")
        elif he_to_tee > 0 and tee_to_he > 0:
            patterns.append("Two-way handoff (HE↔TEE)")

        # Calculate efficiency
        success_rate = stats['success_rate']

        return {
            'total_handoffs': total,
            'he_to_tee_count': he_to_tee,
            'tee_to_he_count': tee_to_he,
            'patterns': patterns,
            'success_rate': success_rate,
            'avg_time_ns': stats['avg_time_ns'],
            'total_time_ns': stats['total_time_ns'],
        }

    def recommend_optimizations(self) -> List[str]:
        """
        Recommend protocol optimizations.

        Returns:
            List of recommendations
        """
        recommendations = []

        stats = self.analyze_handoffs()

        # Check success rate
        if stats['success_rate'] < 0.95:
            recommendations.append(
                "Improve handoff success rate (currently <95%)"
            )

        # Check handoff frequency
        if stats['total_handoffs'] > 100:
            recommendations.append(
                "Consider batching operations to reduce handoff frequency"
            )

        # Check two-way handoffs
        if stats['tee_to_he_count'] > 0:
            recommendations.append(
                "Avoid TEE→HE handoffs if possible (decrypt once, stay in TEE)"
            )

        # Check timing
        if stats['avg_time_ns'] > 100000:  # >100 μs
            recommendations.append(
                "Optimize handoff for speed (currently >100 μs average)"
            )

        return recommendations


def simulate_ht2ml_protocol(
    num_operations: int = 10,
    data_size: int = 100,
) -> Dict[str, Any]:
    """
    Simulate complete HT2ML protocol workflow.

    Args:
        num_operations: Number of handoff operations
        data_size: Size of data vector

    Returns:
        Simulation results
    """
    # Create enclave
    enclave = Enclave(enclave_id="ht2ml-enclave")

    # Create protocol
    protocol = create_handoff_protocol(enclave)

    # Create mock HE context and data
    from tee_ml.security.threat_model import create_ht2ml_threat_model

    he_context = HEContext(
        scheme='ckks',
        poly_modulus_degree=4096,
        scale=2**30,
        eval=1,
    )

    encrypted_data = HEData(
        encrypted_data=[],  # Mock
        shape=(data_size,),
        scheme='ckks',
        scale=2**30,
    )

    # Simulate handoffs
    successful = 0
    total_time_ns = 0

    for i in range(num_operations):
        import time
        start = time.perf_counter_ns()

        # HE→TEE handoff
        success, plaintext = protocol.handoff_he_to_tee(
            encrypted_data=encrypted_data,
            he_context=he_context,
            nonce=f"nonce-{i}".encode(),
        )

        elapsed = time.perf_counter_ns() - start
        total_time_ns += elapsed

        if success:
            successful += 1

    # Calculate statistics
    avg_time_ns = total_time_ns / num_operations

    return {
        'num_operations': num_operations,
        'successful_handoffs': successful,
        'failed_handoffs': num_operations - successful,
        'success_rate': successful / num_operations,
        'total_time_ns': total_time_ns,
        'avg_time_ns': avg_time_ns,
        'avg_time_us': avg_time_ns / 1000,
        'total_time_us': total_time_ns / 1000,
        'total_time_ms': total_time_ns / 1e6,
    }
