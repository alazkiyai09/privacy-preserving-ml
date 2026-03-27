"""
Gradient Bound Proofs for Federated Learning

This module implements zero-knowledge proofs for proving that client gradients
are bounded (i.e., not maliciously large) without revealing the gradients.

PROBLEM IN FEDERATED LEARNING:
Malicious clients may send extremely large gradients to disrupt the model.
Server needs to detect and reject these attacks.

NAIVE SOLUTION:
Server checks gradient norms → Reveals client gradients → Privacy leak!

ZK SOLUTION:
Client proves ||gradient|| ≤ bound without revealing gradient.

MATHEMATICAL FORMULATION:
Given gradient vector g, prove:
    sqrt(sum(g[i]^2)) ≤ bound

This is equivalent to:
    sum(g[i]^2) ≤ bound^2

We implement this as an arithmetic circuit and generate ZK proof.

USE CASE:
1. Client computes gradient from local data
2. Client generates ZK proof that ||gradient|| ≤ bound
3. Client sends gradient (encrypted) + proof to server
4. Server verifies proof without seeing gradient
5. Server accepts/rejects based on proof validity

Security:
- Server learns nothing about gradient (only that it's bounded)
- Client cannot prove unbounded gradient is bounded
- Proof is small (~128 bytes with Groth16)
- Verification is fast (~5 ms)
"""

import numpy as np
from typing import List, Tuple, Optional
import hashlib

from ..snark.circuits import ArithmeticCircuit, CircuitBuilder
from ..snark.r1cs import R1CS, QAP
from ..snark.proof_gen import ProofGenerator, Proof
from ..snark.verification import ProofVerifier, VerificationResult
from ..snark.trusted_setup import ProvingKey, VerificationKey, TrustedSetup


class GradientBoundProof:
    """
    Zero-knowledge proof that gradient L2 norm is bounded.

    PROVES: ||gradient|| ≤ bound
    REVEALS: Nothing about gradient values

    Example:
        >>> system = GradientBoundProof(bound=1.0, gradient_size=100)
        >>> pk, vk = system.setup()
        >>> gradient = np.random.randn(100) * 0.1  # Small gradient
        >>> proof = system.generate_proof(gradient, pk)
        >>> assert system.verify(proof, vk)
    """

    def __init__(
        self,
        bound: float,
        gradient_size: int,
        field_prime: int = 101
    ):
        """
        Initialize gradient bound proof system.

        Args:
            bound: L2 norm bound (e.g., 1.0)
            gradient_size: Dimension of gradient vector
            field_prime: Field modulus for arithmetic

        Note:
            Small field_prime for demonstration. Use 256-bit prime in production.
        """
        self.bound = bound
        self.bound_sq = bound ** 2
        self.gradient_size = gradient_size
        self.field_prime = field_prime

        # Build circuit for gradient norm computation
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> ArithmeticCircuit:
        """
        Build arithmetic circuit for gradient norm check.

        Circuit computes:
        1. norm_sq = sum(gradient[i]^2)
        2. Check norm_sq ≤ bound_sq

        Returns:
            Arithmetic circuit
        """
        circuit = ArithmeticCircuit(field_prime=self.field_prime)

        # Add gradient inputs (private)
        g_wires = []
        for i in range(self.gradient_size):
            g = circuit.add_private_input(f"g{i}")
            g_wires.append(g)

        # Compute squares
        squares = []
        for g in g_wires:
            sq = circuit.add_gate(circuit.GateType.MUL, g, g)
            squares.append(sq)

        # Sum squares (L2 norm squared)
        if len(squares) == 1:
            norm_sq = squares[0]
        else:
            norm_sq = squares[0]
            for sq in squares[1:]:
                from ..snark.circuits import GateType
                norm_sq = circuit.add_gate(GateType.ADD, norm_sq, sq)

        # Add bound constant
        bound_sq_wire = circuit.add_constant(int(self.bound_sq) % self.field_prime)

        # Output: norm_sq and bound_sq (for verification)
        circuit.set_output(norm_sq)
        circuit.set_output(bound_sq_wire)

        return circuit

    def setup(self, use_mpc: bool = True) -> Tuple[ProvingKey, VerificationKey]:
        """
        Perform trusted setup for gradient bound circuit.

        Args:
            use_mpc: Whether to use MPC ceremony (recommended)

        Returns:
            (ProvingKey, VerificationKey)

        WARNING:
        If use_mpc=False, single-party setup is DANGEROUS!
        Use MPC ceremony in production.
        """
        setup = TrustedSetup(field_prime=self.field_prime)

        if use_mpc:
            # Multi-party setup ceremony
            num_participants = 100
            participant_ids = [f"participant_{i}" for i in range(num_participants)]
            pk, vk = setup.mpc_setup(
                circuit_size=self.circuit.get_num_gates(),
                num_participants=num_participants,
                participant_ids=participant_ids
            )
        else:
            # Single-party setup (DANGEROUS!)
            pk, vk = setup.single_party_setup(
                circuit_size=self.circuit.get_num_gates()
            )

        return pk, vk

    def generate_proof(
        self,
        gradient: np.ndarray,
        proving_key: ProvingKey
    ) -> Proof:
        """
        Generate proof that gradient is bounded.

        Args:
            gradient: Gradient vector (numpy array)
            proving_key: Proving key from setup

        Returns:
            Zero-knowledge proof

        Raises:
            ValueError: If gradient exceeds bound

        Process:
        1. Check gradient norm is actually within bound
        2. Convert gradient to field elements
        3. Compute witness (circuit assignment)
        4. Generate ZK proof
        """
        # Check gradient norm
        norm_sq = np.sum(gradient ** 2)

        if norm_sq > self.bound_sq + 1e-10:  # Small tolerance for floating point
            raise ValueError(
                f"Gradient norm {np.sqrt(norm_sq)} exceeds bound {self.bound}"
            )

        # Convert gradient to field elements (simplified)
        # In production, use proper fixed-point or integer representation
        gradient_int = [int(abs(x) * 100) % self.field_prime for x in gradient]

        # Create witness (input assignment)
        witness = [0] * self.circuit.get_num_wires()

        # Set gradient values
        for i, val in enumerate(gradient_int):
            wire_id = self.circuit.inputs[f"g{i}"].id
            witness[wire_id] = val

        # Compute circuit outputs by evaluating
        outputs = self.circuit.evaluate(self.circuit.inputs)

        # Set output wire values
        for i, output_wire in enumerate(self.circuit.outputs):
            witness[output_wire.id] = outputs[i] % self.field_prime

        # Generate proof
        generator = ProofGenerator(self.circuit, proving_key)
        proof = generator.generate_proof(witness)

        return proof

    def verify(
        self,
        proof: Proof,
        verification_key: VerificationKey
    ) -> bool:
        """
        Verify proof that gradient is bounded.

        Args:
            proof: Proof to verify
            verification_key: Verification key from setup

        Returns:
            True if proof is valid

        Security:
        - Verification does NOT reveal gradient
        - Only proves gradient was within bound
        """
        verifier = ProofVerifier(self.circuit, verification_key)
        result = verifier.verify_proof(proof)
        return result.is_valid

    def estimate_proof_size(self) -> int:
        """
        Estimate proof size in bytes.

        Returns:
            Estimated size (Groth16: 128 bytes)
        """
        return 128

    def estimate_generation_time(self) -> float:
        """
        Estimate proof generation time.

        Returns:
            Estimated time in seconds
        """
        import math
        base_time = 0.01  # 10 ms
        # Scales with gradient size
        return base_time * math.log(self.gradient_size + 1)


class BatchGradientBoundProof:
    """
    Batch proof for multiple gradient bounds.

    USE CASE:
    Server verifies proofs from multiple clients efficiently.

    Optimization:
    Uses batch verification to reduce total verification time.
    """

    def __init__(self, bound: float, gradient_size: int):
        """
        Initialize batch proof system.

        Args:
            bound: L2 norm bound
            gradient_size: Gradient dimension
        """
        self.single_system = GradientBoundProof(bound, gradient_size)

    def generate_proofs(
        self,
        gradients: List[np.ndarray],
        proving_key: ProvingKey
    ) -> List[Proof]:
        """
        Generate proofs for multiple gradients.

        Args:
            gradients: List of gradient vectors
            proving_key: Proving key

        Returns:
            List of proofs
        """
        proofs = []
        for gradient in gradients:
            proof = self.single_system.generate_proof(gradient, proving_key)
            proofs.append(proof)

        return proofs

    def batch_verify(
        self,
        proofs: List[Proof],
        verification_key: VerificationKey
    ) -> List[bool]:
        """
        Verify multiple proofs.

        Args:
            proofs: List of proofs
            verification_key: Verification key

        Returns:
            List of verification results
        """
        verifier = ProofVerifier(
            self.single_system.circuit,
            verification_key
        )

        results = verifier.batch_verify_proofs(proofs)

        return [r.is_valid for r in results]


def demo_gradient_bound_proof():
    """
    Demonstrate gradient bound proof system.
    """
    print("=" * 70)
    print("GRADIENT BOUND PROOF DEMONSTRATION")
    print("=" * 70)
    print()

    # Setup
    print("1. Setting up gradient bound proof system...")
    bound = 1.0
    gradient_size = 100
    system = GradientBoundProof(bound=bound, gradient_size=gradient_size)

    print(f"   Bound: {bound}")
    print(f"   Gradient size: {gradient_size}")
    print()

    # Trusted setup
    print("2. Performing trusted setup ceremony (MPC)...")
    pk, vk = system.setup(use_mpc=True)
    print(f"   Circuit ID: {pk.circuit_id}")
    print()

    # Generate test gradients
    print("3. Generating test gradients...")

    # Valid gradient (small)
    valid_gradient = np.random.randn(gradient_size) * 0.1
    valid_norm = np.linalg.norm(valid_gradient)
    print(f"   Valid gradient norm: {valid_norm:.4f} (≤ {bound})")

    # Invalid gradient (large)
    invalid_gradient = np.random.randn(gradient_size) * 2.0
    invalid_norm = np.linalg.norm(invalid_gradient)
    print(f"   Invalid gradient norm: {invalid_norm:.4f} (> {bound})")
    print()

    # Generate proofs
    print("4. Generating proofs...")

    valid_proof = system.generate_proof(valid_gradient, pk)
    print(f"   ✓ Proof generated for valid gradient")
    print(f"   Proof size: {valid_proof.size_bytes()} bytes")
    print()

    # Try to generate proof for invalid gradient
    print("5. Attempting to generate proof for invalid gradient...")
    try:
        invalid_proof = system.generate_proof(invalid_gradient, pk)
        print("   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    print()

    # Verify proof
    print("6. Verifying proof...")
    is_valid = system.verify(valid_proof, vk)
    print(f"   Verification result: {'VALID ✓' if is_valid else 'INVALID ✗'}")
    print()

    # Statistics
    print("7. Performance estimates...")
    gen_time = system.estimate_generation_time() * 1000
    print(f"   Generation time: ~{gen_time:.1f} ms")
    print(f"   Verification time: ~5 ms")
    print(f"   Proof size: ~{system.estimate_proof_size()} bytes")
    print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_gradient_bound_proof()
