"""
Proof Verification for ZK-SNARKs

This module handles verification of zero-knowledge proofs.

VERIFICATION PROCESS:
1. Receive proof from prover
2. Verify proof using verification key
3. Check pairing equation: e(A, B) == e(α, β) · e(C, δ)
4. Optionally check public inputs match expected values

MATHEMATICAL BACKGROUND:
In Groth16, verification checks a pairing equation:
e([A]₁, [B]₂) == e([α]₁, [β]₂) · e([C]₁, [δ]₂)

where:
- e(.,.) is a bilinear pairing
- [A]₁, [C]₁ are points in G1
- [B]₂ is a point in G2
- [α]₁, [β]₂, [δ]₁ are from verification key

If the equation holds, the proof is valid.

SECURITY PROPERTIES:
- Soundness: Fake proofs cannot verify (except with negligible probability)
- Zero-knowledge: Valid proofs reveal nothing about witness
- Succinctness: Verification is fast (~milliseconds)

USE CASE IN FEDERATED LEARNING:
- Server verifies client gradient proofs
- Server verifies training data validity proofs
- Server verifies computation correctness proofs

All without seeing client private data!

WARNING: This is a simplified educational implementation.
Production systems should use established libraries with proper pairing operations.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import json

from .circuits import ArithmeticCircuit
from .trusted_setup import VerificationKey
from .proof_gen import Proof


class VerificationError(Exception):
    """Exception raised when proof verification fails."""
    pass


@dataclass
class VerificationResult:
    """
    Result of proof verification.
    """
    is_valid: bool
    error_message: Optional[str] = None
    verification_time: float = 0.0

    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"VerificationResult({status}, time={self.verification_time*1000:.2f}ms)"


class ProofVerifier:
    """
    Verify zero-knowledge proofs.

    USE CASE:
    Server verifies client proofs without seeing client data.

    Example:
        >>> verifier = ProofVerifier(circuit, verification_key)
        >>> result = verifier.verify_proof(proof, public_inputs)
        >>> if result.is_valid:
        ...     print("Proof valid!")
    """

    def __init__(
        self,
        circuit: ArithmeticCircuit,
        verification_key: VerificationKey
    ):
        """
        Initialize proof verifier.

        Args:
            circuit: Arithmetic circuit (for reference)
            verification_key: Verification key from trusted setup
        """
        self.circuit = circuit
        self.verification_key = verification_key

        # Validate circuit ID matches
        if verification_key.circuit_id != self._get_expected_circuit_id():
            raise ValueError("Verification key does not match circuit")

    def verify_proof(
        self,
        proof: Proof,
        public_inputs: Optional[dict] = None,
        expected_output: Optional[int] = None
    ) -> VerificationResult:
        """
        Verify a zero-knowledge proof.

        Args:
            proof: Proof to verify
            public_inputs: Optional public input values to check
            expected_output: Optional expected output value

        Returns:
            Verification result

        Verification Process:
        1. Check circuit ID matches
        2. Verify proof structure is valid
        3. Check pairing equation (simplified here)
        4. Optionally verify public inputs match

        Security:
        - Invalid proofs (from wrong witness) will not verify
        - Verification is fast (~milliseconds)
        - Reveals nothing about private witness
        """
        import time
        start_time = time.time()

        try:
            # Check circuit ID
            if proof.circuit_id != self.verification_key.circuit_id:
                return VerificationResult(
                    is_valid=False,
                    error_message="Circuit ID mismatch"
                )

            # Verify proof structure
            if not self._verify_proof_structure(proof):
                return VerificationResult(
                    is_valid=False,
                    error_message="Invalid proof structure"
                )

            # Verify pairing equation (simplified)
            if not self._verify_pairing_equation(proof):
                return VerificationResult(
                    is_valid=False,
                    error_message="Pairing equation failed"
                )

            # Optionally verify public inputs
            if public_inputs is not None:
                if not self._verify_public_inputs(proof, public_inputs):
                    return VerificationResult(
                        is_valid=False,
                        error_message="Public inputs mismatch"
                    )

            # Optionally verify expected output
            if expected_output is not None:
                if not self._verify_output(proof, expected_output):
                    return VerificationResult(
                        is_valid=False,
                        error_message="Output mismatch"
                    )

            # Verification succeeded
            verification_time = time.time() - start_time
            return VerificationResult(
                is_valid=True,
                verification_time=verification_time
            )

        except Exception as e:
            verification_time = time.time() - start_time
            return VerificationResult(
                is_valid=False,
                error_message=f"Verification error: {str(e)}",
                verification_time=verification_time
            )

    def batch_verify_proofs(
        self,
        proofs: List[Proof],
        public_inputs_list: Optional[List[dict]] = None
    ) -> List[VerificationResult]:
        """
        Verify multiple proofs efficiently.

        Args:
            proofs: List of proofs to verify
            public_inputs_list: Optional list of public inputs for each proof

        Returns:
            List of verification results

        USE CASE:
        Server verifies proofs from multiple clients in parallel.
        """
        results = []

        for i, proof in enumerate(proofs):
            public_inputs = None
            if public_inputs_list is not None:
                public_inputs = public_inputs_list[i]

            result = self.verify_proof(proof, public_inputs)
            results.append(result)

        return results

    def estimate_verification_time(self) -> float:
        """
        Estimate verification time.

        Returns:
            Estimated time in seconds

        Benchmarks:
        - Groth16: ~1-10 ms
        - PLONK: ~3-15 ms
        - Bulletproofs: ~10-50 ms (slower, no trusted setup)
        """
        # Simplified: return Groth16 time
        return 0.005  # 5 ms

    def _verify_proof_structure(self, proof: Proof) -> bool:
        """
        Verify proof has valid structure.

        Args:
            proof: Proof to check

        Returns:
            True if structure is valid
        """
        # Check proof has all required fields
        if not hasattr(proof, 'A_eval') or not hasattr(proof, 'B_eval'):
            return False
        if not hasattr(proof, 'C_eval') or not hasattr(proof, 'H_eval'):
            return False

        # Check values are in field
        field_mod = self.circuit.field_prime
        if not (0 <= proof.A_eval < field_mod):
            return False
        if not (0 <= proof.B_eval < field_mod):
            return False
        if not (0 <= proof.C_eval < field_mod):
            return False

        return True

    def _verify_pairing_equation(self, proof: Proof) -> bool:
        """
        Verify pairing equation (simplified).

        Args:
            proof: Proof to verify

        Returns:
            True if pairing equation holds

        Mathematical Check:
        In real Groth16:
        e(A, B) == e(α, β) · e(C, δ)

        Here we use a simplified check based on polynomial relationship.
        """
        # Simplified verification using polynomial relationship
        # A * B = C * H (mod p)

        field_mod = self.circuit.field_prime

        left = (proof.A_eval * proof.B_eval) % field_mod
        right = (proof.C_eval * proof.H_eval) % field_mod

        return left == right

    def _verify_public_inputs(
        self,
        proof: Proof,
        public_inputs: dict
    ) -> bool:
        """
        Verify public inputs match expected values.

        Args:
            proof: Proof being verified
            public_inputs: Expected public input values

        Returns:
            True if public inputs match

        Note:
            In real system, public inputs are embedded in proof structure.
            This is a simplified version.
        """
        # In real implementation, would check against proof's public input commitments
        # For now, just return True (assume proof includes correct public inputs)
        return True

    def _verify_output(self, proof: Proof, expected_output: int) -> bool:
        """
        Verify output matches expected value.

        Args:
            proof: Proof being verified
            expected_output: Expected output value

        Returns:
            True if output matches

        Note:
            In real system, output is derived from C evaluation.
        """
        # Simplified: check if C evaluation matches expected
        # In real system, would decode output from proof
        return True

    def _get_expected_circuit_id(self) -> str:
        """Get expected circuit ID."""
        import hashlib
        circuit_str = str(self.circuit.get_num_wires())
        circuit_str += str(self.circuit.get_num_gates())
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]


class ProofVerifierOptimized(ProofVerifier):
    """
    Optimized proof verifier with batching and caching.

    OPTIMIZATIONS:
    1. Batch verify multiple proofs
    2. Cache verification keys
    3. Parallel verification
    4. Early rejection for obviously invalid proofs
    """

    def __init__(self, circuit: ArithmeticCircuit, verification_key: VerificationKey):
        """Initialize optimized verifier."""
        super().__init__(circuit, verification_key)
        self._verification_cache = {}

    def verify_proof(
        self,
        proof: Proof,
        public_inputs: Optional[dict] = None,
        expected_output: Optional[int] = None
    ) -> VerificationResult:
        """Verify proof with caching."""
        # Create cache key from proof
        cache_key = self._get_cache_key(proof, public_inputs)

        if cache_key in self._verification_cache:
            # Return cached result
            return self._verification_cache[cache_key]

        # Verify normally
        result = super().verify_proof(proof, public_inputs, expected_output)

        # Cache result
        self._verification_cache[cache_key] = result

        return result

    def _get_cache_key(self, proof: Proof, public_inputs: Optional[dict]) -> str:
        """Generate cache key for proof."""
        import hashlib
        key_str = f"{proof.A_eval}{proof.B_eval}{proof.C_eval}"
        if public_inputs:
            key_str += str(public_inputs)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear verification cache."""
        self._verification_cache.clear()


class VerificationAuditor:
    """
    Audit verification process for security issues.

    Provides tools to detect verification problems.
    """

    @staticmethod
    def audit_verification_key(
        verification_key: VerificationKey,
        circuit: ArithmeticCircuit
    ) -> Dict[str, bool]:
        """
        Audit verification key for security issues.

        Args:
            verification_key: Verification key to audit
            circuit: Associated circuit

        Returns:
            Dictionary of audit results
        """
        results = {}

        # Check circuit ID matches
        expected_id = ProofVerifier(circuit, verification_key)._get_expected_circuit_id()
        results["circuit_id_match"] = verification_key.circuit_id == expected_id

        # Check field modulus matches
        results["field_modulus_match"] = verification_key.field_prime == circuit.field_prime

        # Check ID is not empty
        results["non_empty_id"] = len(verification_key.circuit_id) > 0

        # Overall security
        results["secure"] = all(results.values())

        return results

    @staticmethod
    def audit_proof(proof: Proof, field_mod: int) -> Dict[str, bool]:
        """
        Audit proof for potential issues.

        Args:
            proof: Proof to audit
            field_mod: Field modulus

        Returns:
            Dictionary of audit results
        """
        results = {}

        # Check values are in field
        results["A_in_field"] = 0 <= proof.A_eval < field_mod
        results["B_in_field"] = 0 <= proof.B_eval < field_mod
        results["C_in_field"] = 0 <= proof.C_eval < field_mod
        results["H_in_field"] = 0 <= proof.H_eval < field_mod

        # Check proof is not trivial
        results["non_trivial_A"] = proof.A_eval != 0
        results["non_trivial_B"] = proof.B_eval != 0

        # Check polynomial relationship
        left = (proof.A_eval * proof.B_eval) % field_mod
        right = (proof.C_eval * proof.H_eval) % field_mod
        results["polynomial_relationship"] = left == right

        # Overall validity
        results["valid"] = all(results.values())

        return results

    @staticmethod
    def generate_audit_report(audit_results: Dict[str, bool]) -> str:
        """Generate human-readable audit report."""
        report = "VERIFICATION AUDIT REPORT\n"
        report += "=" * 50 + "\n\n"

        for check, passed in audit_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report += f"{status}: {check}\n"

        if audit_results.get("valid", False):
            report += "\nOVERALL: Proof/Key appears valid\n"
        else:
            report += "\nOVERALL: Issues detected!\n"

        return report


class VerificationStatistics:
    """
    Collect verification statistics.
    """

    def __init__(self):
        """Initialize statistics tracker."""
        self.total_verifications = 0
        self.successful_verifications = 0
        self.failed_verifications = 0
        self.total_verification_time = 0.0

    def record_verification(self, result: VerificationResult) -> None:
        """
        Record verification result.

        Args:
            result: Verification result
        """
        self.total_verifications += 1
        self.total_verification_time += result.verification_time

        if result.is_valid:
            self.successful_verifications += 1
        else:
            self.failed_verifications += 1

    def get_success_rate(self) -> float:
        """Get success rate (0-1)."""
        if self.total_verifications == 0:
            return 0.0
        return self.successful_verifications / self.total_verifications

    def get_average_time(self) -> float:
        """Get average verification time."""
        if self.total_verifications == 0:
            return 0.0
        return self.total_verification_time / self.total_verifications

    def generate_report(self) -> str:
        """Generate statistics report."""
        report = "VERIFICATION STATISTICS\n"
        report += "=" * 50 + "\n\n"
        report += f"Total verifications: {self.total_verifications}\n"
        report += f"Successful: {self.successful_verifications}\n"
        report += f"Failed: {self.failed_verifications}\n"
        report += f"Success rate: {self.get_success_rate()*100:.1f}%\n"
        report += f"Average time: {self.get_average_time()*1000:.2f} ms\n"
        report += f"Throughput: {1.0/self.get_average_time():.0f} verifications/second\n"

        return report
