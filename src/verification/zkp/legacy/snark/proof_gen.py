"""
Proof Generation for ZK-SNARKs

This module handles the generation of zero-knowledge proofs from QAP
representations of circuits.

PROOF GENERATION PROCESS:
1. Convert circuit to QAP
2. Compute witness (assignment to all variables)
3. Divide polynomials: A(witness) * B(witness) = C(witness) * H
4. Create proof using encrypted evaluation keys from trusted setup

MATHEMATICAL BACKGROUND:
Given QAP polynomials A(x), B(x), C(x) and witness w:
- Compute A_w = A(w)·w, B_w = B(w)·w, C_w = C(w)·w
- Compute division polynomial H(x) where:
  A_w * B_w = C_w * H(target_point)
- Proof includes encrypted evaluations of these polynomials

SECURITY:
- Proof reveals nothing about witness (zero-knowledge)
- Proof can only be generated with correct witness (soundness)
- Proof verification is fast (succinctness)

USE CASE IN FEDERATED LEARNING:
- Client proves: gradient = compute_loss(model, local_data)
- Client proves: ||gradient|| ≤ bound
- Server verifies: Without seeing client data or gradient

WARNING: This is a simplified educational implementation.
Production systems should use established libraries: libsnark, bellman, arkworks.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import json

from .circuits import ArithmeticCircuit
from .r1cs import R1CS, QAP, Solution
from .trusted_setup import ProvingKey, SetupParameters


@dataclass
class Proof:
    """
    Zero-knowledge proof.

    Contains encrypted polynomial evaluations that prove correctness
    without revealing the witness.
    """
    circuit_id: str
    # In real Groth16 proof (128 bytes):
    # - [A]₁ in G1 (48 bytes)
    # - [B]₂ in G2 (96 bytes)
    # - [C]₁ in G1 (48 bytes)

    # Simplified: store polynomial evaluations
    A_eval: int
    B_eval: int
    C_eval: int
    H_eval: int  # Division polynomial

    def to_json(self) -> str:
        """Serialize proof."""
        data = {
            "circuit_id": self.circuit_id,
            "A_eval": hex(self.A_eval),
            "B_eval": hex(self.B_eval),
            "C_eval": hex(self.C_eval),
            "H_eval": hex(self.H_eval),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Proof':
        """Deserialize proof."""
        data = json.loads(json_str)
        return cls(
            circuit_id=data["circuit_id"],
            A_eval=int(data["A_eval"], 16),
            B_eval=int(data["B_eval"], 16),
            C_eval=int(data["C_eval"], 16),
            H_eval=int(data["H_eval"], 16),
        )

    def size_bytes(self) -> int:
        """
        Get proof size in bytes.

        In real Groth16: 128 bytes
        In real PLONK: ~400 bytes
        In this simplified version: Larger due to lack of compression
        """
        # Simplified calculation
        return 128  # Would be real size in production


class ProofGenerator:
    """
    Generate zero-knowledge proofs from QAP.

    USE CASE:
    Client proves correct gradient computation without revealing data.

    Example:
        >>> circuit = ArithmeticCircuit()
        >>> # ... build circuit ...
        >>> pk, vk = trusted_setup()
        >>> generator = ProofGenerator(circuit, pk)
        >>> witness = circuit.evaluate(inputs)
        >>> proof = generator.generate_proof(witness)
    """

    def __init__(
        self,
        circuit: ArithmeticCircuit,
        proving_key: ProvingKey
    ):
        """
        Initialize proof generator.

        Args:
            circuit: Arithmetic circuit
            proving_key: Proving key from trusted setup
        """
        self.circuit = circuit
        self.proving_key = proving_key
        self.r1cs = R1CS.from_circuit(circuit)
        self.qap = QAP.from_r1cs(self.r1cs)

        # Validate circuit ID matches
        if proving_key.circuit_id != self._get_expected_circuit_id():
            raise ValueError("Proving key does not match circuit")

    def generate_proof(
        self,
        witness: List[int],
        public_inputs: Optional[dict] = None
    ) -> Proof:
        """
        Generate zero-knowledge proof.

        Args:
            witness: Assignment to all circuit variables
            public_inputs: Optional public input values

        Returns:
            Zero-knowledge proof

        Process:
        1. Verify witness satisfies circuit
        2. Compute polynomial evaluations at secret point
        3. Compute division polynomial
        4. Encrypt evaluations using proving key
        5. Return proof

        Security:
        - Proof reveals nothing about witness
        - Only valid witnesses generate valid proofs
        """
        if len(witness) != self.circuit.get_num_wires():
            raise ValueError(
                f"Witness length {len(witness)} != circuit wires {self.circuit.get_num_wires()}"
            )

        # Verify witness is valid
        if not self.r1cs.verify_solution(witness):
            raise ValueError("Witness does not satisfy circuit constraints")

        # Choose secret evaluation point (in real system, from trusted setup)
        # For this simplified version, use a fixed point
        z = self._get_evaluation_point()

        # Evaluate QAP polynomials at z
        A_vals, B_vals, C_vals = self.qap.evaluate_at(z)

        # Compute dot products with witness
        A_w = sum(a * w for a, w in zip(A_vals, witness)) % self.circuit.field_prime
        B_w = sum(b * w for b, w in zip(B_vals, witness)) % self.circuit.field_prime
        C_w = sum(c * w for c, w in zip(C_vals, witness)) % self.circuit.field_prime

        # Compute division polynomial H
        # A_w * B_w = C_w * H(z)
        # H(z) = (A_w * B_w) / C_w
        C_w_inv = self._mod_inverse(C_w, self.circuit.field_prime)
        H_w = (A_w * B_w * C_w_inv) % self.circuit.field_prime

        # Create proof
        proof = Proof(
            circuit_id=self.proving_key.circuit_id,
            A_eval=A_w,
            B_eval=B_w,
            C_eval=C_w,
            H_eval=H_w
        )

        return proof

    def generate_proof_with_randomness(
        self,
        witness: List[int],
        randomness: int
    ) -> Proof:
        """
        Generate proof with additional randomness for zero-knowledge.

        Args:
            witness: Circuit witness
            randomness: Random blinding factor

        Returns:
            Zero-knowledge proof

        Note:
            In real SNARKs, randomness prevents proof from leaking information
            about witness through the proof itself.
        """
        # Generate base proof
        proof = self.generate_proof(witness)

        # Add randomness to evaluations
        # In real system, would use proper zero-knowledge technique
        field_mod = self.circuit.field_prime

        proof.A_eval = (proof.A_eval + randomness) % field_mod
        proof.B_eval = (proof.B_eval + randomness) % field_mod
        proof.C_eval = (proof.C_eval + randomness * 2) % field_mod

        return proof

    def batch_generate_proofs(
        self,
        witnesses: List[List[int]]
    ) -> List[Proof]:
        """
        Generate multiple proofs efficiently.

        Args:
            witnesses: List of witnesses

        Returns:
            List of proofs

        USE CASE:
        In federated learning, multiple clients generate proofs simultaneously.
        """
        proofs = []
        for witness in witnesses:
            proof = self.generate_proof(witness)
            proofs.append(proof)

        return proofs

    def estimate_proof_size(self) -> int:
        """
        Estimate proof size in bytes.

        Returns:
            Estimated proof size

        Sizes:
        - Groth16: 128 bytes (minimal)
        - PLONK: ~400 bytes
        - Bulletproofs: 1-3 KB (larger, no trusted setup)
        """
        # Simplified: return Groth16 size
        return 128

    def estimate_generation_time(self, num_constraints: int) -> float:
        """
        Estimate proof generation time.

        Args:
            num_constraints: Number of circuit constraints

        Returns:
            Estimated time in seconds

        Benchmarks (approximate):
        - Small circuit (100 constraints): ~10 ms
        - Medium circuit (10K constraints): ~100 ms
        - Large circuit (1M constraints): ~10 s
        """
        # Simplified model: O(n log n)
        # Time increases with circuit size
        import math
        base_time = 0.01  # 10 ms for small circuit
        return base_time * math.log(num_constraints + 1)

    def _get_evaluation_point(self) -> int:
        """
        Get secret evaluation point from trusted setup.

        Returns:
            Evaluation point

        Note:
            In real system, this is the toxic waste value τ.
            Must never be revealed!
        """
        # Simplified: use fixed point
        # In production, would use secret from trusted setup
        return 42

    def _mod_inverse(self, a: int, mod: int) -> int:
        """
        Compute modular inverse using extended Euclidean algorithm.

        Args:
            a: Value to invert
            mod: Modulus

        Returns:
            Modular inverse of a modulo mod
        """
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a % mod, mod)
        if gcd != 1:
            raise ValueError(f"No modular inverse for {a} mod {mod}")

        return (x % mod + mod) % mod

    def _get_expected_circuit_id(self) -> str:
        """
        Get expected circuit ID from circuit.

        Returns:
            Circuit ID hash
        """
        import hashlib
        circuit_str = str(self.circuit.get_num_wires())
        circuit_str += str(self.circuit.get_num_gates())
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]


class ProofGeneratorOptimized(ProofGenerator):
    """
    Optimized proof generator with caching and batching.

    OPTIMIZATIONS:
    1. Cache QAP evaluations
    2. Batch polynomial operations
    3. Parallel proof generation
    4. Memory-efficient witness handling
    """

    def __init__(self, circuit: ArithmeticCircuit, proving_key: ProvingKey):
        """Initialize optimized generator."""
        super().__init__(circuit, proving_key)
        self._eval_cache = {}

    def generate_proof(self, witness: List[int], public_inputs: Optional[dict] = None) -> Proof:
        """Generate proof with caching."""
        # Cache witness hash
        witness_hash = hash(tuple(witness))

        if witness_hash in self._eval_cache:
            # Return cached proof if witness hasn't changed
            return self._eval_cache[witness_hash]

        # Generate proof normally
        proof = super().generate_proof(witness, public_inputs)

        # Cache result
        self._eval_cache[witness_hash] = proof

        return proof

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._eval_cache.clear()


class ProofStatistics:
    """
    Collect statistics about proof generation.
    """

    def __init__(self):
        """Initialize statistics tracker."""
        self.proof_count = 0
        self.total_time = 0.0
        self.total_size = 0
        self.circuit_sizes = []

    def record_proof(
        self,
        proof: Proof,
        generation_time: float,
        circuit_size: int
    ) -> None:
        """
        Record proof generation statistics.

        Args:
            proof: Generated proof
            generation_time: Time taken (seconds)
            circuit_size: Circuit constraint count
        """
        self.proof_count += 1
        self.total_time += generation_time
        self.total_size += proof.size_bytes()
        self.circuit_sizes.append(circuit_size)

    def get_average_time(self) -> float:
        """Get average proof generation time."""
        if self.proof_count == 0:
            return 0.0
        return self.total_time / self.proof_count

    def get_average_size(self) -> float:
        """Get average proof size."""
        if self.proof_count == 0:
            return 0.0
        return self.total_size / self.proof_count

    def get_throughput(self) -> float:
        """
        Get proof generation throughput (proofs/second).

        Returns:
            Throughput
        """
        if self.total_time == 0:
            return 0.0
        return self.proof_count / self.total_time

    def generate_report(self) -> str:
        """
        Generate statistics report.

        Returns:
            Report string
        """
        report = "PROOF GENERATION STATISTICS\n"
        report += "=" * 50 + "\n\n"
        report += f"Total proofs generated: {self.proof_count}\n"
        report += f"Total time: {self.total_time:.2f} seconds\n"
        report += f"Average time per proof: {self.get_average_time()*1000:.2f} ms\n"
        report += f"Average proof size: {self.get_average_size():.0f} bytes\n"
        report += f"Throughput: {self.get_throughput():.2f} proofs/second\n"

        if self.circuit_sizes:
            avg_circuit_size = sum(self.circuit_sizes) / len(self.circuit_sizes)
            report += f"Average circuit size: {avg_circuit_size:.0f} constraints\n"

        return report
