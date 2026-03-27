"""
Secure Enclave Abstraction
==========================

Provides a software simulation of a Trusted Execution Environment (TEE)
similar to Intel SGX or ARM TrustZone.

Key Concepts:
- Enclave: Isolated execution environment
- EnclaveSession: Active session within an enclave
- SecureMemory: Isolated memory region within enclave
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from threading import Lock
import uuid
import numpy as np

from tee_ml.simulation.overhead import OverheadModel, estimate_inference_overhead


class EnclaveError(Exception):
    """Base exception for enclave operations."""
    pass


class EnclaveMemoryError(EnclaveError):
    """Exception raised when enclave memory limit is exceeded."""
    pass


class EnclaveSecurityError(EnclaveError):
    """Exception raised when security violation is detected."""
    pass


@dataclass
class SecureMemory:
    """
    Isolated memory region within enclave.

    Simulates Enclave Page Cache (EPC) in Intel SGX.
    Memory is encrypted and isolated from non-enclave code.
    """

    size_bytes: int
    _data: Dict[int, np.ndarray] = field(default_factory=dict)
    _allocated_bytes: int = field(default=0)
    _lock: Lock = field(default_factory=Lock)

    def allocate(self, data: np.ndarray) -> int:
        """
        Allocate memory for data in secure region.

        Args:
            data: Data to store

        Returns:
            Offset (memory address) within secure memory

        Raises:
            EnclaveMemoryError: If allocation exceeds size limit
        """
        with self._lock:
            data_size = data.nbytes

            if self._allocated_bytes + data_size > self.size_bytes:
                raise EnclaveMemoryError(
                    f"Cannot allocate {data_size} bytes: "
                    f"only {self.size_bytes - self._allocated_bytes} bytes available"
                )

            # Generate unique offset
            offset = len(self._data)
            self._data[offset] = data.copy()
            self._allocated_bytes += data_size

            return offset

    def read(self, offset: int) -> np.ndarray:
        """
        Read data from secure memory.

        Args:
            offset: Memory offset returned by allocate()

        Returns:
            Copy of data at offset

        Raises:
            KeyError: If offset is invalid
        """
        with self._lock:
            if offset not in self._data:
                raise KeyError(f"Invalid memory offset: {offset}")

            return self._data[offset].copy()

    def free(self, offset: int) -> None:
        """
        Free memory at offset.

        Args:
            offset: Memory offset to free

        Raises:
            KeyError: If offset is invalid
        """
        with self._lock:
            if offset not in self._data:
                raise KeyError(f"Invalid memory offset: {offset}")

            data_size = self._data[offset].nbytes
            del self._data[offset]
            self._allocated_bytes -= data_size

    def get_usage(self) -> float:
        """
        Get current memory usage in bytes.

        Returns:
            Memory usage in bytes
        """
        return self._allocated_bytes

    def get_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        return self._allocated_bytes / (1024 * 1024)

    def get_utilization(self) -> float:
        """
        Get memory utilization as fraction of total.

        Returns:
            Utilization (0.0 to 1.0)
        """
        return self._allocated_bytes / self.size_bytes


@dataclass
class EnclaveSession:
    """
    Active session within an enclave.

    A session represents a single enclave entry/exit cycle.
    Data enters the enclave, operations are performed, then data exits.
    """

    enclave: 'Enclave'
    input_data: np.ndarray
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _output_data: Optional[np.ndarray] = None
    _is_active: bool = True
    _entry_time_ns: int = 0
    _exit_time_ns: int = 0

    def __post_init__(self):
        """Initialize session with entry time."""
        import time
        self._entry_time_ns = time.perf_counter_ns()

    def execute(
        self,
        operation: Callable[[np.ndarray], Any],
        **kwargs
    ) -> Any:
        """
        Execute operation within enclave.

        Args:
            operation: Function to execute on input_data
            **kwargs: Additional arguments for operation

        Returns:
            Result of operation

        Raises:
            EnclaveSecurityError: If session is not active
        """
        if not self._is_active:
            raise EnclaveSecurityError("Cannot execute operation: session is not active")

        try:
            # Execute operation on input data
            result = operation(self.input_data, **kwargs)
            return result
        except Exception as e:
            # Re-raise as enclave error
            raise EnclaveError(f"Operation failed in enclave: {e}")

    def get_memory_usage(self) -> float:
        """
        Get memory usage for this session in MB.

        Returns:
            Memory usage in MB
        """
        return self.input_data.nbytes / (1024 * 1024)

    def get_session_duration_ns(self) -> int:
        """
        Get session duration in nanoseconds.

        Returns:
            Duration from entry to now (or exit if exited)
        """
        import time

        if self._exit_time_ns > 0:
            return self._exit_time_ns - self._entry_time_ns
        else:
            return time.perf_counter_ns() - self._entry_time_ns

    def is_active(self) -> bool:
        """
        Check if session is still active.

        Returns:
            True if session is active, False if exited
        """
        return self._is_active

    def close(self) -> None:
        """
        Close session (mark as exited).

        Sets exit time and marks session as inactive.
        """
        import time
        self._exit_time_ns = time.perf_counter_ns()
        self._is_active = False


@dataclass
class EnclaveMeasurement:
    """
    Measurement (hash) of enclave code for attestation.

    In real SGX, this is SHA256(MRENCLAVE) which hashes:
    - Enclave code
    - Initialized data
    - SSA frame size
    - Stack size

    For simulation, we use a simplified hash.
    """

    code_hash: bytes
    data_hash: bytes
    stack_size: int
    heap_size: int

    def get_measurement(self) -> bytes:
        """
        Get combined measurement hash.

        Returns:
            SHA256 hash of all components
        """
        import hashlib

        h = hashlib.sha256()
        h.update(self.code_hash)
        h.update(self.data_hash)
        h.update(self.stack_size.to_bytes(8, 'big'))
        h.update(self.heap_size.to_bytes(8, 'big'))

        return h.digest()


class Enclave:
    """
    Trusted Execution Environment (TEE) abstraction.

    Simulates an enclave similar to Intel SGX or ARM TrustZone.

    Key Features:
    - Memory isolation via SecureMemory
    - Session management via enter()/exit()
    - Measurement generation for attestation
    - Overhead tracking for realistic simulation

    Example:
        >>> enclave = Enclave(enclave_id="my-enclave", memory_limit_mb=128)
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> session = enclave.enter(data)
        >>> result = session.execute(lambda x: x * 2)
        >>> enclave.exit(session)
    """

    def __init__(
        self,
        enclave_id: str,
        memory_limit_mb: int = 128,
        overhead_model: OverheadModel = None,
    ):
        """
        Initialize enclave.

        Args:
            enclave_id: Unique identifier for this enclave
            memory_limit_mb: Secure memory limit in MB
            overhead_model: OverheadModel for simulation (default: standard model)
        """
        self.enclave_id = enclave_id
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.overhead_model = overhead_model or OverheadModel()

        # Secure memory region
        self.secure_memory = SecureMemory(size_bytes=self.memory_limit_bytes)

        # Session tracking
        self._active_sessions: Dict[str, EnclaveSession] = {}
        self._session_history: List[EnclaveSession] = []
        self._session_lock = Lock()

        # Measurement for attestation
        self._measurement = self._generate_measurement()

        # Statistics
        self._total_entries = 0
        self._total_exits = 0

    def _generate_measurement(self) -> EnclaveMeasurement:
        """
        Generate measurement hash for attestation.

        In real SGX, this would be SHA256 of enclave code.
        For simulation, we generate a hash based on enclave properties.

        Returns:
            EnclaveMeasurement
        """
        import hashlib

        # Hash enclave ID and memory limit
        code_hash = hashlib.sha256(
            f"{self.enclave_id}:{self.memory_limit_bytes}".encode()
        ).digest()

        # Hash some random "data"
        data_hash = hashlib.sha256(b"enclave_initialization_data").digest()

        # Stack and heap sizes (simulated)
        stack_size = 8 * 1024 * 1024  # 8 MB
        heap_size = self.memory_limit_bytes - stack_size

        return EnclaveMeasurement(
            code_hash=code_hash,
            data_hash=data_hash,
            stack_size=stack_size,
            heap_size=heap_size,
        )

    def get_measurement(self) -> bytes:
        """
        Get enclave measurement for attestation.

        Returns:
            Measurement hash
        """
        return self._measurement.get_measurement()

    def enter(self, data: np.ndarray) -> EnclaveSession:
        """
        Enter enclave with data.

        Creates a new session and simulates enclave entry overhead.

        Args:
            data: Data to bring into enclave

        Returns:
            EnclaveSession for this entry

        Raises:
            EnclaveMemoryError: If data exceeds memory limit
        """
        # Simulate entry overhead
        # In real TEE, this involves context switching and mode transition
        import time
        start = time.perf_counter_ns()

        # Check memory limit
        data_size = data.nbytes
        if data_size > self.memory_limit_bytes:
            raise EnclaveMemoryError(
                f"Data size ({data_size} bytes) exceeds enclave memory limit "
                f"({self.memory_limit_bytes} bytes)"
            )

        # Create session
        session = EnclaveSession(enclave=self, input_data=data.copy())

        with self._session_lock:
            self._active_sessions[session.session_id] = session
            self._total_entries += 1

        # Simulate entry time
        entry_time = self.overhead_model.entry_ns
        if entry_time > 0:
            time.sleep(entry_time / 1e9)

        return session

    def exit(self, session: EnclaveSession) -> np.ndarray:
        """
        Exit enclave and retrieve result.

        Closes session and simulates enclave exit overhead.

        Args:
            session: Session to exit

        Returns:
            Output data from session

        Raises:
            EnclaveSecurityError: If session is not active
        """
        if not session.is_active():
            raise EnclaveSecurityError("Cannot exit inactive session")

        # Close session
        session.close()

        # Simulate exit overhead
        import time
        exit_time = self.overhead_model.exit_ns
        if exit_time > 0:
            time.sleep(exit_time / 1e9)

        with self._session_lock:
            # Remove from active sessions
            if session.session_id in self._active_sessions:
                del self._active_sessions[session.session_id]

            # Add to history
            self._session_history.append(session)
            self._total_exits += 1

        # Return input data (or output if set)
        return session.input_data if session._output_data is None else session._output_data

    def is_isolated(self) -> bool:
        """
        Check if enclave provides memory isolation.

        In simulation, this always returns True.
        In real TEE, this depends on hardware.

        Returns:
            True (simulated isolation)
        """
        return True

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        return self.secure_memory.get_usage_mb()

    def get_memory_utilization(self) -> float:
        """
        Get memory utilization as fraction.

        Returns:
            Utilization (0.0 to 1.0)
        """
        return self.secure_memory.get_utilization()

    def get_active_session_count(self) -> int:
        """
        Get number of active sessions.

        Returns:
            Number of active sessions
        """
        with self._session_lock:
            return len(self._active_sessions)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get enclave statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "enclave_id": self.enclave_id,
            "memory_limit_mb": self.memory_limit_bytes / (1024 * 1024),
            "memory_used_mb": self.get_memory_usage(),
            "memory_utilization": self.get_memory_utilization(),
            "active_sessions": self.get_active_session_count(),
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "total_sessions": len(self._session_history),
        }

    def estimate_overhead_for_inference(
        self,
        model_layers: int,
    ) -> Dict[str, Any]:
        """
        Estimate overhead for neural network inference.

        Args:
            model_layers: Number of layers in model

        Returns:
            Overhead estimation
        """
        return estimate_inference_overhead(
            model_layers=model_layers,
            data_size_mb=self.get_memory_usage(),
            overhead_model=self.overhead_model,
        )


def create_enclave(
    enclave_id: str = None,
    memory_limit_mb: int = 128,
) -> Enclave:
    """
    Factory function to create an enclave.

    Args:
        enclave_id: Unique identifier (auto-generated if None)
        memory_limit_mb: Memory limit in MB

    Returns:
        Enclave instance
    """
    if enclave_id is None:
        enclave_id = f"enclave-{uuid.uuid4()}"

    return Enclave(
        enclave_id=enclave_id,
        memory_limit_mb=memory_limit_mb,
    )
