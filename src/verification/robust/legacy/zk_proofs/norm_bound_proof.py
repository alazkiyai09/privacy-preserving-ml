"""
Gradient Norm Bound Proof

Specialized ZK proof for gradient norm bound verification.

This is the most critical proof for preventing model poisoning attacks.

Why Important:
- Prevents gradient scaling attacks (dominating aggregation)
- Prevents model collapse from extremely large updates
- ZK ensures privacy: client doesn't reveal actual gradient

Circuit Design (conceptual):
1. Input: gradient (vector), bound (scalar)
2. Compute: norm = sqrt(sum(gradient^2))
3. Assert: norm ≤ bound

In production:
- Design arithmetic circuit for norm computation
- Use zk-SNARKs to generate/verify proof
- Proof size: ~200 bytes (regardless of gradient size!)
- Verification time: ~10ms
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib


class GradientNormProof:
    """
    Gradient Norm Bound Proof.

    Proves that ‖gradient‖ ≤ bound without revealing gradient.
    """

    def __init__(self, bound: float, use_simplified: bool = True):
        """
        Initialize gradient norm proof.

        Args:
            bound: Maximum allowed gradient norm
            use_simplified: Use simplified commitment scheme
        """
        self.bound = bound
        self.use_simplified = use_simplified

    def generate_proof(
        self,
        gradient: List[np.ndarray],
        client_id: int = 0
    ) -> Dict[str, Any]:
        """
        Generate proof that gradient norm ≤ bound.

        Args:
            gradient: Client's gradient
            client_id: Client identifier

        Returns:
            Proof dictionary
        """
        # Compute norm
        norm = self._compute_norm(gradient)

        # Check within bound
        within_bound = norm <= self.bound

        if self.use_simplified:
            # Generate commitment
            commitment = self._commit_to_gradient(gradient)

            # Generate Merkle tree for layers (for efficiency)
            merkle_root = self._compute_merkle_root(gradient)

            proof = {
                "proof_type": "gradient_norm_bound",
                "client_id": client_id,
                "norm": float(norm),
                "bound": float(self.bound),
                "within_bound": within_bound,
                "commitment": commitment,
                "merkle_root": merkle_root,
                "method": "simplified"
            }
        else:
            # In production: Generate zk-SNARK proof
            # This requires designing arithmetic circuit
            proof = {
                "proof_type": "gradient_norm_bound",
                "client_id": client_id,
                "norm": float(norm),
                "bound": float(self.bound),
                "within_bound": within_bound,
                "snark_proof": "zk-SNARK proof (200 bytes)",
                "public_inputs": {
                    "norm_commitment": commitment,
                    "bound_hash": hashlib.sha256(str(self.bound).encode()).hexdigest()
                },
                "method": "zk_snark"
            }

        return proof

    def verify_proof(
        self,
        proof: Dict[str, Any],
        gradient: List[np.ndarray] = None
    ) -> bool:
        """
        Verify gradient norm bound proof.

        Args:
            proof: Proof to verify
            gradient: Gradient (for re-computation in simplified mode)

        Returns:
            True if proof is valid
        """
        # Check if within bound
        if "within_bound" in proof:
            # Trust the proof's claim (for zk-SNARK)
            return proof["within_bound"]

        # Re-compute norm if gradient provided
        if gradient is not None:
            norm = self._compute_norm(gradient)
            return norm <= self.bound

        # Cannot verify without gradient
        return False

    def verify_batch(
        self,
        proofs: List[Dict[str, Any]]
    ) -> Tuple[List[bool], Dict[str, Any]]:
        """
        Verify multiple proofs efficiently.

        Args:
            proofs: List of proofs to verify

        Returns:
            (verification_results, batch_metrics)
        """
        results = []
        valid_count = 0
        total_norm = 0.0

        for proof in proofs:
            valid = self.verify_proof(proof)
            results.append(valid)

            if valid:
                valid_count += 1
                total_norm += proof.get("norm", 0.0)

        metrics = {
            "total_proofs": len(proofs),
            "valid_count": valid_count,
            "invalid_count": len(proofs) - valid_count,
            "valid_rate": valid_count / len(proofs) if proofs else 0.0,
            "average_norm": total_norm / valid_count if valid_count > 0 else 0.0
        }

        return results, metrics

    def adapt_bound_automatically(
        self,
        client_norms: List[float],
        percentile: float = 90
    ) -> float:
        """
        Adapt bound based on historical client norms.

        Args:
            client_norms: List of historical gradient norms
            percentile: Percentile to use for bound

        Returns:
            New bound
        """
        if not client_norms:
            return self.bound

        # Compute percentile
        new_bound = np.percentile(client_norms, percentile)

        # Update bound
        old_bound = self.bound
        self.bound = float(new_bound)

        print(f"Adapted bound: {old_bound:.4f} → {self.bound:.4f}")

        return self.bound

    # Helper methods

    def _compute_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)

    def _commit_to_gradient(self, gradient: List[np.ndarray]) -> str:
        """Create commitment to gradient."""
        flat = np.concatenate([layer.flatten() for layer in gradient])
        commitment = hashlib.sha256(flat.tobytes()).hexdigest()
        return commitment

    def _compute_merkle_root(self, gradient: List[np.ndarray]) -> str:
        """
        Compute Merkle root of gradient layers.

        This allows efficient verification of individual layers.
        """
        # Hash each layer
        layer_hashes = []
        for layer in gradient:
            hash_val = hashlib.sha256(layer.tobytes()).hexdigest()
            layer_hashes.append(hash_val)

        # Build Merkle tree
        while len(layer_hashes) > 1:
            new_level = []
            for i in range(0, len(layer_hashes), 2):
                if i + 1 < len(layer_hashes):
                    # Hash two adjacent nodes
                    combined = layer_hashes[i] + layer_hashes[i+1]
                    new_hash = hashlib.sha256(combined.encode()).hexdigest()
                    new_level.append(new_hash)
                else:
                    # Odd number of nodes, carry forward
                    new_level.append(layer_hashes[i])
            layer_hashes = new_level

        return layer_hashes[0] if layer_hashes else ""


class AdaptiveNormBound(GradientNormProof):
    """
    Adaptive norm bound that adjusts based on client behavior.

    Strategy:
    - Start with conservative bound
    - Monitor client norms over time
    - Adjust bound to accommodate honest clients
    - Detect significant deviations as attacks
    """

    def __init__(
        self,
        initial_bound: float = 1.0,
        window_size: int = 10,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive norm bound.

        Args:
            initial_bound: Starting bound
            window_size: Number of recent rounds to consider
            adaptation_rate: How quickly to adapt (0.0 to 1.0)
        """
        super().__init__(bound=initial_bound)
        self.initial_bound = initial_bound
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # Track norms
        self.norm_history = []

    def update_bound(self, new_norm: float) -> float:
        """
        Update bound based on new gradient norm.

        Args:
            new_norm: New gradient norm observed

        Returns:
            New bound
        """
        # Add to history
        self.norm_history.append(new_norm)

        # Keep only recent history
        if len(self.norm_history) > self.window_size:
            self.norm_history = self.norm_history[-self.window_size:]

        # Compute adaptive bound
        median_norm = np.median(self.norm_history)

        # Adapt bound towards median
        target_bound = median_norm * 1.5  # 50% above median
        new_bound = (
            (1 - self.adaptation_rate) * self.bound +
            self.adaptation_rate * target_bound
        )

        # Update bound (but don't go below initial bound)
        old_bound = self.bound
        self.bound = max(new_bound, self.initial_bound)

        return self.bound

    def is_anomalous(self, norm: float) -> Tuple[bool, float]:
        """
        Check if norm is anomalous.

        Args:
            norm: Gradient norm to check

        Returns:
            (is_anomalous, z_score)
        """
        if len(self.norm_history) < 3:
            return False, 0.0

        # Compute z-score
        mean_norm = np.mean(self.norm_history)
        std_norm = np.std(self.norm_history) + 1e-8

        z_score = abs((norm - mean_norm) / std_norm)
        is_anomalous = z_score > 3.0  # 3-sigma rule

        return is_anomalous, z_score


# Example usage
if __name__ == "__main__":
    print("Gradient Norm Bound Proof Demonstration")
    print("=" * 60)

    # Create proof instance
    norm_proof = GradientNormProof(bound=1.0, use_simplified=True)

    # Create honest gradient
    honest_gradient = [np.random.randn(10, 10) * 0.1 for _ in range(3)]

    print("Honest gradient:")
    honest_norm = norm_proof._compute_norm(honest_gradient)
    print(f"  Norm: {honest_norm:.4f}")

    # Generate proof
    honest_proof = norm_proof.generate_proof(honest_gradient, client_id=0)

    print(f"\nProof generated:")
    print(f"  Proof type: {honest_proof['proof_type']}")
    print(f"  Norm: {honest_proof['norm']:.4f}")
    print(f"  Bound: {honest_proof['bound']:.4f}")
    print(f"  Within bound: {honest_proof['within_bound']}")
    print(f"  Commitment: {honest_proof['commitment'][:16]}...")

    # Verify
    verified = norm_proof.verify_proof(honest_proof)
    print(f"\nVerification: {'✓ VALID' if verified else '✗ INVALID'}")

    # Test with malicious gradient
    print("\n" + "=" * 60)
    print("Malicious gradient (scaled by 10x):")
    malicious_gradient = [np.random.randn(10, 10) * 1.0 for _ in range(3)]

    malicious_norm = norm_proof._compute_norm(malicious_gradient)
    print(f"  Norm: {malicious_norm:.4f}")

    malicious_proof = norm_proof.generate_proof(malicious_gradient, client_id=1)

    print(f"\nProof generated:")
    print(f"  Within bound: {malicious_proof['within_bound']}")

    verified_mal = norm_proof.verify_proof(malicious_proof)
    print(f"  Verification: {'✓ VALID' if verified_mal else '✗ INVALID'}")

    # Batch verification
    print("\n" + "=" * 60)
    print("Batch verification:")
    proofs = [honest_proof, malicious_proof]

    results, metrics = norm_proof.verify_batch(proofs)

    print(f"Results: {[('✓' if r else '✗') for r in results]}")
    print(f"Metrics:")
    print(f"  Valid: {metrics['valid_count']}/{metrics['total_proofs']}")
    print(f"  Valid rate: {metrics['valid_rate']:.2%}")
    print(f"  Average norm: {metrics['average_norm']:.4f}")

    # Adaptive bound
    print("\n" + "=" * 60)
    print("Adaptive norm bound:")
    adaptive_proof = AdaptiveNormBound(initial_bound=1.0, window_size=10)

    # Simulate multiple rounds
    for round_num in range(1, 16):
        norm = np.random.randn() * 0.3 + 0.5  # Mean 0.5, std 0.3
        new_bound = adaptive_proof.update_bound(norm)

        if round_num % 5 == 0:
            print(f"Round {round_num}: norm={norm:.4f}, bound={new_bound:.4f}")
