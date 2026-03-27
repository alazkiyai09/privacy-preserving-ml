"""
ZK Proof Verifier for Federated Learning

Verifies zero-knowledge proofs from FL clients.

Verification:
1. Gradient Norm Bound: Verify ‖gradient‖ ≤ bound
2. Participation: Verify trained on ≥ min_samples
3. Training Correctness: Verify training executed correctly

Security:
- If proofs are valid, client update is accepted
- Invalid proofs result in exclusion from aggregation
- ZK ensures privacy: no gradient information leaked
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class ZKProofVerifier:
    """
    Verify Zero-Knowledge Proofs from FL clients.

    This is a demonstration implementation. In production:
    - Verify zk-SNARK proofs using pairing checks
    - Use trusted setup public parameters
    - Verify cryptographic signatures
    """

    def __init__(self, use_simplified: bool = True):
        """
        Initialize ZK proof verifier.

        Args:
            use_simplified: Use simplified verification (True) or
                          full zk-SNARK verification (False)
        """
        self.use_simplified = use_simplified

        # Track verification statistics
        self.verification_stats = {
            "total_verified": 0,
            "gradient_norm_failures": 0,
            "participation_failures": 0,
            "correctness_failures": 0
        }

    def verify_gradient_norm_proof(
        self,
        proof: Dict[str, Any],
        gradient: List[np.ndarray] = None
    ) -> bool:
        """
        Verify gradient norm bound proof.

        In production (zk-SNARK):
        1. Verify SNARK proof using pairing check
        2. Check public inputs match commitment

        Args:
            proof: Proof dictionary
            gradient: Gradient (for simplified verification)

        Returns:
            True if proof is valid
        """
        if self.use_simplified:
            # Simplified: Re-compute norm and check
            if "norm" in proof and "bound" in proof:
                # In production, would verify commitment instead
                within_bound = proof["norm"] <= proof["bound"]
                return within_bound
            elif gradient is not None:
                # Re-compute norm
                norm = self._compute_gradient_norm(gradient)
                within_bound = norm <= proof.get("bound", float('inf'))
                return within_bound
            else:
                return False
        else:
            # In production: Verify zk-SNARK proof
            # This requires external libraries
            try:
                # Placeholder for SNARK verification
                # snark_proof = proof["snark_proof"]
                # public_inputs = proof["public_inputs"]
                # verified = verify_snark(snark_proof, public_inputs, vk)
                verified = proof.get("within_bound", False)
                return verified
            except Exception as e:
                print(f"SNARK verification error: {e}")
                return False

    def verify_participation_proof(
        self,
        proof: Dict[str, Any]
    ) -> bool:
        """
        Verify participation proof.

        Args:
            proof: Proof dictionary

        Returns:
            True if proof is valid
        """
        if self.use_simplified:
            # Simplified: Check if participated
            if "participated" in proof:
                return proof["participated"]
            elif "num_samples" in proof and "min_samples" in proof:
                return proof["num_samples"] >= proof["min_samples"]
            else:
                return False
        else:
            # In production: Verify zk-SNARK proof
            try:
                verified = proof.get("participated", False)
                return verified
            except Exception:
                return False

    def verify_training_correctness_proof(
        self,
        proof: Dict[str, Any],
        initial_params: List[np.ndarray] = None,
        final_params: List[np.ndarray] = None,
        gradient: List[np.ndarray] = None
    ) -> bool:
        """
        Verify training correctness proof.

        Args:
            proof: Proof dictionary
            initial_params: Parameters before training (for re-verification)
            final_params: Parameters after training
            gradient: Computed gradient

        Returns:
            True if proof is valid
        """
        if self.use_simplified:
            # Simplified: Re-verify gradient computation
            if "gradient_correct" in proof:
                return proof["gradient_correct"]
            elif all([initial_params, final_params, gradient]):
                # Re-compute and verify
                return self._verify_gradient_computation(
                    initial_params, final_params, gradient
                )
            else:
                return False
        else:
            # In production: Verify zk-SNARK proof
            try:
                verified = proof.get("gradient_correct", False)
                return verified
            except Exception:
                return False

    def verify_all_proofs(
        self,
        proofs: Dict[str, Any],
        gradient: List[np.ndarray] = None,
        initial_params: List[np.ndarray] = None,
        final_params: List[np.ndarray] = None
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Verify all proofs from client.

        Args:
            proofs: Dictionary with all proofs
            gradient: Client's gradient (for re-verification)
            initial_params: Parameters before training
            final_params: Parameters after training

        Returns:
            (all_verified, verification_results)
        """
        verification_results = {}

        # Verify gradient norm proof
        if "gradient_norm_proof" in proofs:
            verified = self.verify_gradient_norm_proof(
                proofs["gradient_norm_proof"],
                gradient
            )
            verification_results["gradient_norm"] = verified
            if not verified:
                self.verification_stats["gradient_norm_failures"] += 1

        # Verify participation proof
        if "participation_proof" in proofs:
            verified = self.verify_participation_proof(
                proofs["participation_proof"]
            )
            verification_results["participation"] = verified
            if not verified:
                self.verification_stats["participation_failures"] += 1

        # Verify training correctness proof
        if "training_correctness_proof" in proofs:
            verified = self.verify_training_correctness_proof(
                proofs["training_correctness_proof"],
                initial_params, final_params, gradient
            )
            verification_results["training_correctness"] = verified
            if not verified:
                self.verification_stats["correctness_failures"] += 1

        # Check if all verified
        all_verified = all(verification_results.values())

        if all_verified:
            self.verification_stats["total_verified"] += 1

        return all_verified, verification_results

    def verify_client_update(
        self,
        client_id: int,
        gradient: List[np.ndarray],
        proofs: Dict[str, Any],
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify client's update with all proofs.

        Args:
            client_id: Client identifier
            gradient: Client's gradient
            proofs: All proofs from client
            initial_params: Parameters before training
            final_params: Parameters after training

        Returns:
            (is_valid, verification_details)
        """
        all_verified, verification_results = self.verify_all_proofs(
            proofs=proofs,
            gradient=gradient,
            initial_params=initial_params,
            final_params=final_params
        )

        details = {
            "client_id": client_id,
            "all_verified": all_verified,
            "verification_results": verification_results,
            "gradient_norm": self._compute_gradient_norm(gradient),
            "proofs_present": list(proofs.keys())
        }

        return all_verified, details

    def get_statistics(self) -> Dict[str, int]:
        """Get verification statistics."""
        return self.verification_stats.copy()

    def reset_statistics(self):
        """Reset verification statistics."""
        self.verification_stats = {
            "total_verified": 0,
            "gradient_norm_failures": 0,
            "participation_failures": 0,
            "correctness_failures": 0
        }

    # Helper methods

    def _compute_gradient_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)

    def _verify_gradient_computation(
        self,
        initial_params: List[np.ndarray],
        final_params: List[np.ndarray],
        gradient: List[np.ndarray]
    ) -> bool:
        """Verify that gradient ≈ final_params - initial_params."""
        expected_gradient = [
            final - initial
            for final, initial in zip(final_params, initial_params)
        ]

        for exp, act in zip(expected_gradient, gradient):
            if not np.allclose(exp, act, rtol=1e-3, atol=1e-5):
                return False

        return True


# Example usage
if __name__ == "__main__":
    print("ZK Proof Verifier Demonstration")
    print("=" * 60)

    # Create verifier
    verifier = ZKProofVerifier(use_simplified=True)

    # Create dummy data
    gradient = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    initial_params = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    final_params = [p + g for p, g in zip(initial_params, gradient)]

    # Import generator to create proofs
    from src.verification.robust.legacy.zk_proofs.proof_generator import ZKProofGenerator

    generator = ZKProofGenerator(client_id=0, use_simplified=True)

    # Generate proofs
    proofs = generator.generate_all_proofs(
        gradient=gradient,
        norm_bound=1.0,
        num_samples=150,
        min_samples=100,
        initial_params=initial_params,
        final_params=final_params,
        num_steps=5
    )

    print("Proofs generated:")
    for proof_type in proofs.keys():
        print(f"  - {proof_type}")

    # Verify proofs
    print("\n" + "=" * 60)
    print("Verifying proofs:")

    verified, details = verifier.verify_client_update(
        client_id=0,
        gradient=gradient,
        proofs=proofs,
        initial_params=initial_params,
        final_params=final_params
    )

    print(f"\nAll verified: {verified}")
    print(f"\nVerification results:")
    for proof_type, result in details["verification_results"].items():
        status = "✓ VALID" if result else "✗ INVALID"
        print(f"  {proof_type}: {status}")

    print(f"\nGradient norm: {details['gradient_norm']:.4f}")

    # Test with invalid proof
    print("\n" + "=" * 60)
    print("Testing with malicious gradient (norm > bound):")

    malicious_gradient = [np.random.randn(10, 10) * 5.0 for _ in range(3)]
    malicious_final = [p + mg for p, mg in zip(initial_params, malicious_gradient)]

    malicious_proofs = generator.generate_gradient_norm_proof(
        malicious_gradient,
        bound=1.0
    )

    print(f"Malicious gradient norm: {malicious_proofs['norm']:.4f}")
    print(f"Bound: {malicious_proofs['bound']:.4f}")
    print(f"Within bound: {malicious_proofs['within_bound']}")

    verified_mal = verifier.verify_gradient_norm_proof(malicious_proofs)
    print(f"Verification result: {'✓ VALID' if verified_mal else '✗ INVALID'}")

    # Statistics
    print("\n" + "=" * 60)
    print("Verification Statistics:")
    stats = verifier.get_statistics()
    print(f"  Total verified: {stats['total_verified']}")
    print(f"  Gradient norm failures: {stats['gradient_norm_failures']}")
    print(f"  Participation failures: {stats['participation_failures']}")
    print(f"  Correctness failures: {stats['correctness_failures']}")
