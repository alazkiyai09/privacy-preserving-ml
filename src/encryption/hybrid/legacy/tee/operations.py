"""
TEE Operations for HT2ML
========================

Implements non-linear operations executed in TEE domain.
Uses TEE simulation from Day 7 work.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TEEOperationError(Exception):
    """TEE operation error."""
    pass


class TEEComputationError(TEEOperationError):
    """TEE computation failed."""
    pass


@dataclass
class TEEOperationResult:
    """
    Result of TEE operation execution.

    Contains output data and execution metrics.
    """
    output: np.ndarray
    execution_time_ms: float
    operation: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'output': self.output.tolist() if self.output is not None else None,
            'execution_time_ms': self.execution_time_ms,
            'operation': self.operation,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }


class TEEOperationEngine:
    """
    TEE computation engine for HT2ML.

    Executes non-linear operations in trusted environment.
    """

    def __init__(self):
        """Initialize TEE operation engine."""
        self.operation_count = 0
        self.total_execution_time_ms = 0.0

    def execute_relu(
        self,
        data: np.ndarray
    ) -> TEEOperationResult:
        """
        Execute ReLU activation in TEE.

        ReLU(x) = max(0, x)

        Args:
            data: Input data

        Returns:
            TEEOperationResult with ReLU output

        Raises:
            TEEComputationError: If computation fails
        """
        import time

        start_time = time.time()

        try:
            # Execute ReLU in TEE
            output = np.maximum(0, data)

            execution_time_ms = (time.time() - start_time) * 1000

            self.operation_count += 1
            self.total_execution_time_ms += execution_time_ms

            return TEEOperationResult(
                output=output,
                execution_time_ms=execution_time_ms,
                operation="relu",
                input_shape=data.shape,
                output_shape=output.shape
            )

        except Exception as e:
            raise TEEComputationError(f"ReLU computation failed: {e}")

    def execute_softmax(
        self,
        data: np.ndarray
    ) -> TEEOperationResult:
        """
        Execute Softmax activation in TEE.

        Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

        Args:
            data: Input logits

        Returns:
            TEEOperationResult with Softmax output

        Raises:
            TEEComputationError: If computation fails
        """
        import time

        start_time = time.time()

        try:
            # Stable softmax implementation
            # Subtract max for numerical stability
            max_val = np.max(data)
            exp_x = np.exp(data - max_val)
            output = exp_x / np.sum(exp_x)

            execution_time_ms = (time.time() - start_time) * 1000

            self.operation_count += 1
            self.total_execution_time_ms += execution_time_ms

            return TEEOperationResult(
                output=output,
                execution_time_ms=execution_time_ms,
                operation="softmax",
                input_shape=data.shape,
                output_shape=output.shape
            )

        except Exception as e:
            raise TEEComputationError(f"Softmax computation failed: {e}")

    def execute_argmax(
        self,
        data: np.ndarray
    ) -> TEEOperationResult:
        """
        Execute Argmax in TEE.

        Argmax(x) = index of maximum value

        Args:
            data: Input data (typically softmax probabilities)

        Returns:
            TEEOperationResult with argmax output

        Raises:
            TEEComputationError: If computation fails
        """
        import time

        start_time = time.time()

        try:
            # Execute argmax
            output = np.argmax(data)

            execution_time_ms = (time.time() - start_time) * 1000

            self.operation_count += 1
            self.total_execution_time_ms += execution_time_ms

            # Convert to array for consistency
            output_array = np.array([output])

            return TEEOperationResult(
                output=output_array,
                execution_time_ms=execution_time_ms,
                operation="argmax",
                input_shape=data.shape,
                output_shape=output_array.shape
            )

        except Exception as e:
            raise TEEComputationError(f"Argmax computation failed: {e}")

    def execute_batch(
        self,
        operations: List[Tuple[str, np.ndarray]]
    ) -> List[TEEOperationResult]:
        """
        Execute multiple TEE operations in batch.

        Args:
            operations: List of (operation_name, data) tuples

        Returns:
            List of TEEOperationResult objects

        Raises:
            TEEComputationError: If any operation fails
        """
        results = []

        for op_name, data in operations:
            if op_name == "relu":
                result = self.execute_relu(data)
            elif op_name == "softmax":
                result = self.execute_softmax(data)
            elif op_name == "argmax":
                result = self.execute_argmax(data)
            else:
                raise TEEComputationError(f"Unknown operation: {op_name}")

            results.append(result)

        return results

    def get_average_execution_time(self) -> float:
        """
        Get average execution time per operation.

        Returns:
            Average time in milliseconds
        """
        if self.operation_count == 0:
            return 0.0

        return self.total_execution_time_ms / self.operation_count

    def get_stats(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'operation_count': self.operation_count,
            'total_execution_time_ms': self.total_execution_time_ms,
            'average_execution_time_ms': self.get_average_execution_time(),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.operation_count = 0
        self.total_execution_time_ms = 0.0


class TEEHandoffManager:
    """
    Manages TEE handoff operations.

    Handles data transfer between HE and TEE domains.
    """

    def __init__(self):
        """Initialize handoff manager."""
        self.handoff_count = 0
        self.total_handoff_time_ms = 0.0

    def receive_from_he(
        self,
        encrypted_data: Any,
        nonce: bytes,
        attestation_report: Any
    ) -> np.ndarray:
        """
        Receive data from HE domain (decrypt in TEE).

        Args:
            encrypted_data: CKKS-encrypted data
            nonce: Freshness nonce
            attestation_report: TEE attestation report

        Returns:
            Decrypted plaintext data

        Raises:
            TEEOperationError: If handoff fails

        Note:
            In production, would decrypt in TEE using secret key.
            For simulation, assumes data is already plaintext.
        """
        import time

        start_time = time.time()

        try:
            # Verify attestation
            if attestation_report is None:
                raise TEEOperationError("Missing attestation report")

            # In production, would decrypt CKKS ciphertext here
            # For simulation, extract plaintext from encrypted data
            if hasattr(encrypted_data, 'data'):
                # Assume encrypted_data contains plaintext in simulation
                if isinstance(encrypted_data.data, list):
                    # Convert placeholder to actual array
                    plaintext = np.zeros(len(encrypted_data.data))
                else:
                    plaintext = encrypted_data.data
            else:
                plaintext = np.array(encrypted_data)

            handoff_time_ms = (time.time() - start_time) * 1000

            self.handoff_count += 1
            self.total_handoff_time_ms += handoff_time_ms

            return plaintext

        except Exception as e:
            raise TEEOperationError(f"HE→TEE handoff failed: {e}")

    def send_to_he(
        self,
        plaintext_data: np.ndarray,
        public_key: Any
    ) -> Any:
        """
        Send data to HE domain (re-encrypt with CKKS).

        Args:
            plaintext_data: Plaintext data from TEE
            public_key: HE public key

        Returns:
            Re-encrypted data

        Raises:
            TEEOperationError: If handoff fails

        Note:
            In production, would re-encrypt with CKKS.
            For simulation, returns placeholder.
        """
        import time

        start_time = time.time()

        try:
            # In production, would encrypt with CKKS here
            # For simulation, create placeholder encrypted data

            from src.encryption.hybrid.legacy.he.encryption import CiphertextVector
            from config.he_config import CKKSParams

            encrypted = CiphertextVector(
                data=[f"reencrypted_{i}" for i in range(len(plaintext_data))],
                size=len(plaintext_data),
                shape=plaintext_data.shape,
                scale=2**40,
                scheme="CKKS"
            )
            encrypted.params = CKKSParams()

            handoff_time_ms = (time.time() - start_time) * 1000

            self.handoff_count += 1
            self.total_handoff_time_ms += handoff_time_ms

            return encrypted

        except Exception as e:
            raise TEEOperationError(f"TEE→HE handoff failed: {e}")

    def get_average_handoff_time(self) -> float:
        """
        Get average handoff time.

        Returns:
            Average time in milliseconds
        """
        if self.handoff_count == 0:
            return 0.0

        return self.total_handoff_time_ms / self.handoff_count

    def get_stats(self) -> dict:
        """
        Get handoff statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'handoff_count': self.handoff_count,
            'total_handoff_time_ms': self.total_handoff_time_ms,
            'average_handoff_time_ms': self.get_average_handoff_time(),
        }

    def reset_stats(self) -> None:
        """Reset handoff statistics."""
        self.handoff_count = 0
        self.total_handoff_time_ms = 0.0


def create_tee_engine() -> TEEOperationEngine:
    """
    Factory function to create TEE operation engine.

    Returns:
        TEEOperationEngine instance
    """
    return TEEOperationEngine()


def create_handoff_manager() -> TEEHandoffManager:
    """
    Factory function to create handoff manager.

    Returns:
        TEEHandoffManager instance
    """
    return TEEHandoffManager()
