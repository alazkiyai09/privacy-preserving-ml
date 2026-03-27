"""
ZK Proof Generator for Federated Learning

Generates zero-knowledge proofs for FL client updates.

Proof Types:
1. Gradient Norm Bound: Prove ‖gradient‖ ≤ bound
2. Participation: Prove trained on ≥ min_samples
3. Training Correctness: Prove training executed correctly

Implementation Notes:
This is a simplified demonstration. In production, use:
- libsnark (C++ with Python bindings)
- py-sncirk (Python SNARK library)
- bellman (Rust-based zk-SNARKs)
- circom+snarkjs (for circuit-based ZK proofs)

For demonstration purposes, we use commitment schemes and
Pedersen commitments which provide a foundation for ZK proofs.
"""

import numpy as np
import torch
import hashlib
from typing import List, Dict, Any, Tuple
import json


class ZKProofGenerator:
    """
    Generate Zero-Knowledge Proofs for FL client updates.

    This is a demonstration implementation. In production:
    - Use zk-SNARKs (libsnark, bellman, py-sncirk)
    - Design arithmetic circuits for each proof
    - Use trusted setup for SNARK keys
    - Generate and verify compact proofs
    """

    def __init__(
        self,
        client_id: int,
        use_simplified: bool = True
    ):
        """
        Initialize ZK proof generator.

        Args:
            client_id: Client identifier
            use_simplified: Use simplified commitment scheme (True) or
                          full zk-SNARKs (False, requires external lib)
        """
        self.client_id = client_id
        self.use_simplified = use_simplified

        # For simplified version, use commitment scheme
        # In production, this would be zk-SNARK parameters
        self.commitment_key = self._generate_commitment_key()

    def generate_gradient_norm_proof(
        self,
        gradient: List[np.ndarray],
        bound: float
    ) -> Dict[str, Any]:
        """
        Generate proof that gradient norm ≤ bound.

        Circuit (conceptual):
        1. Compute squared_norm = sum(gradient^2)
        2. Compute norm = sqrt(squared_norm)
        3. Assert norm ≤ bound

        In production: Design arithmetic circuit, generate SNARK proof

        Args:
            gradient: Client's gradient (list of arrays)
            bound: Maximum allowed norm

        Returns:
            Proof dictionary
        """
        # Compute gradient norm
        norm = self._compute_gradient_norm(gradient)

        # Check if within bound
        within_bound = norm <= bound

        if self.use_simplified:
            # Simplified: Use commitment scheme
            commitment = self._commit_gradient(gradient)

            proof = {
                "proof_type": "gradient_norm_bound",
                "commitment": commitment,
                "norm": float(norm),
                "bound": float(bound),
                "within_bound": within_bound,
                "client_id": self.client_id,
                "method": "simplified_commitment"
            }
        else:
            # In production: Generate actual zk-SNARK proof
            # This requires external libraries like libsnark or bellman
            proof = {
                "proof_type": "gradient_norm_bound",
                "snark_proof": "zk-SNARK proof here",
                "public_inputs": {
                    "norm_commitment": "...",
                    "bound_hash": "..."
                },
                "within_bound": within_bound,
                "client_id": self.client_id,
                "method": "zk_snark"
            }

        return proof

    def generate_participation_proof(
        self,
        num_samples: int,
        min_samples: int
    ) -> Dict[str, Any]:
        """
        Generate proof that client trained on ≥ min_samples.

        Circuit (conceptual):
        1. Input: num_samples
        2. Assert num_samples ≥ min_samples

        Args:
            num_samples: Number of training samples
            min_samples: Minimum required samples

        Returns:
            Proof dictionary
        """
        participated = num_samples >= min_samples

        if self.use_simplified:
            # Hash commitment
            commitment = self._commit_number(num_samples)

            proof = {
                "proof_type": "participation",
                "commitment": commitment,
                "num_samples": num_samples,
                "min_samples": min_samples,
                "participated": participated,
                "client_id": self.client_id,
                "method": "simplified_commitment"
            }
        else:
            proof = {
                "proof_type": "participation",
                "snark_proof": "zk-SNARK proof here",
                "public_inputs": {
                    "samples_commitment": "...",
                    "min_samples_hash": "..."
                },
                "participated": participated,
                "client_id": self.client_id,
                "method": "zk_snark"
            }

        return proof

    def generate_training_correctness_proof(
        self,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray],
        gradient: List[np.ndarray],
        num_steps: int
    ) -> Dict[str, Any]:
        """
        Generate proof that training was executed correctly.

        Circuit (conceptual):
        1. Verify: gradient ≈ final_params - initial_params
        2. Verify: num_steps matches expected
        3. Verify: descent direction (loss decreased)

        Args:
            initial_params: Parameters before training
            final_params: Parameters after training
            gradient: Computed gradient
            num_steps: Number of training steps

        Returns:
            Proof dictionary
        """
        # Verify gradient computation
        gradient_computed = self._verify_gradient_computation(
            initial_params, final_params, gradient
        )

        if self.use_simplified:
            # Commit to all inputs
            commitments = {
                "initial_params": self._commit_params(initial_params),
                "final_params": self._commit_params(final_params),
                "gradient": self._commit_gradient(gradient)
            }

            proof = {
                "proof_type": "training_correctness",
                "commitments": commitments,
                "gradient_correct": gradient_computed,
                "num_steps": num_steps,
                "client_id": self.client_id,
                "method": "simplified_commitment"
            }
        else:
            proof = {
                "proof_type": "training_correctness",
                "snark_proof": "zk-SNARK proof here",
                "public_inputs": commitments,
                "gradient_correct": gradient_computed,
                "num_steps": num_steps,
                "client_id": self.client_id,
                "method": "zk_snark"
            }

        return proof

    def generate_all_proofs(
        self,
        gradient: List[np.ndarray],
        norm_bound: float,
        num_samples: int,
        min_samples: int,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray],
        num_steps: int
    ) -> Dict[str, Any]:
        """
        Generate all proofs for a client update.

        Args:
            gradient: Client's gradient
            norm_bound: Gradient norm bound
            num_samples: Number of training samples
            min_samples: Minimum required samples
            initial_params: Parameters before training
            final_params: Parameters after training
            num_steps: Number of training steps

        Returns:
            Dictionary with all proofs
        """
        return {
            "gradient_norm_proof": self.generate_gradient_norm_proof(
                gradient, norm_bound
            ),
            "participation_proof": self.generate_participation_proof(
                num_samples, min_samples
            ),
            "training_correctness_proof": self.generate_training_correctness_proof(
                initial_params, final_params, gradient, num_steps
            ),
            "all_verified": True  # Will be updated by verifier
        }

    # Helper methods for simplified commitment scheme

    def _compute_gradient_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)

    def _commit_gradient(self, gradient: List[np.ndarray]) -> str:
        """Create commitment to gradient (simplified)."""
        # Flatten and hash
        flat = np.concatenate([layer.flatten() for layer in gradient])
        # In production, use Pedersen commitment or other ZK commitment
        commitment = hashlib.sha256(flat.tobytes()).hexdigest()
        return commitment

    def _commit_params(self, params: List[np.ndarray]) -> str:
        """Create commitment to parameters."""
        flat = np.concatenate([p.flatten() for p in params])
        commitment = hashlib.sha256(flat.tobytes()).hexdigest()
        return commitment

    def _commit_number(self, number: int) -> str:
        """Create commitment to number."""
        bytes_data = number.to_bytes((number.bit_length() + 7) // 8, 'big')
        commitment = hashlib.sha256(bytes_data).hexdigest()
        return commitment

    def _verify_gradient_computation(
        self,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray],
        gradient: List[np.ndarray]
    ) -> bool:
        """Verify that gradient ≈ final_params - initial_params."""
        # Compute expected gradient
        expected_gradient = [
            final - initial
            for final, initial in zip(final_params, initial_params)
        ]

        # Check if close (allowing for numerical errors)
        for exp, act in zip(expected_gradient, gradient):
            if not np.allclose(exp, act, rtol=1e-3, atol=1e-5):
                return False

        return True

    def _generate_commitment_key(self) -> str:
        """Generate commitment key (simplified)."""
        # In production, generate proper cryptographic keys
        return f"commitment_key_client_{self.client_id}"


# Example usage
if __name__ == "__main__":
    print("ZK Proof Generator Demonstration")
    print("=" * 60)

    # Create proof generator
    generator = ZKProofGenerator(client_id=0, use_simplified=True)

    # Create dummy gradient
    gradient = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    norm = generator._compute_gradient_norm(gradient)

    print(f"Gradient norm: {norm:.4f}")

    # Generate gradient norm proof
    proof = generator.generate_gradient_norm_proof(gradient, bound=1.0)

    print(f"\nGradient Norm Proof:")
    print(f"  Proof type: {proof['proof_type']}")
    print(f"  Norm: {proof['norm']:.4f}")
    print(f"  Bound: {proof['bound']:.4f}")
    print(f"  Within bound: {proof['within_bound']}")
    print(f"  Commitment: {proof['commitment'][:16]}...")

    # Generate participation proof
    part_proof = generator.generate_participation_proof(
        num_samples=150,
        min_samples=100
    )

    print(f"\nParticipation Proof:")
    print(f"  Proof type: {part_proof['proof_type']}")
    print(f"  Num samples: {part_proof['num_samples']}")
    print(f"  Min samples: {part_proof['min_samples']}")
    print(f"  Participated: {part_proof['participated']}")

    # Generate training correctness proof
    initial_params = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    final_params = [p + g for p, g in zip(initial_params, gradient)]

    correct_proof = generator.generate_training_correctness_proof(
        initial_params=initial_params,
        final_params=final_params,
        gradient=gradient,
        num_steps=5
    )

    print(f"\nTraining Correctness Proof:")
    print(f"  Proof type: {correct_proof['proof_type']}")
    print(f"  Gradient correct: {correct_proof['gradient_correct']}")
    print(f"  Num steps: {correct_proof['num_steps']}")

    # Generate all proofs
    print(f"\n" + "=" * 60)
    print("Generating all proofs for client update:")
    all_proofs = generator.generate_all_proofs(
        gradient=gradient,
        norm_bound=1.0,
        num_samples=150,
        min_samples=100,
        initial_params=initial_params,
        final_params=final_params,
        num_steps=5
    )

    print(f"  Gradient norm proof: Generated")
    print(f"  Participation proof: Generated")
    print(f"  Training correctness proof: Generated")
    print(f"\nAll proofs ready for verification!")
