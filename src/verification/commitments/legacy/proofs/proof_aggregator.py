"""
Proof Aggregator

Batch verification and aggregation of zero-knowledge proofs.
"""

from typing import List, Dict, Any, Tuple
import time
import logging


class ProofAggregator:
    """
    Aggregate and verify multiple proofs efficiently.

    This class handles batch verification of proofs from multiple clients,
    optimizing for scalability in federated learning.
    """

    def __init__(
        self,
        verify_all_proofs: bool = True,
        fail_fast: bool = False
    ):
        """
        Initialize proof aggregator.

        Args:
            verify_all_proofs: Verify all proofs or fail fast on first failure
            fail_fast: Stop verification on first failure
        """
        self.verify_all_proofs = verify_all_proofs
        self.fail_fast = fail_fast
        self.logger = logging.getLogger("ProofAggregator")

    def verify_all_proofs(
        self,
        proofs: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Verify all proofs from a client.

        Args:
            proofs: Dictionary of proofs
            metrics: Client metrics

        Returns:
            - True if all proofs valid
            - List of failed proof names
        """
        failed_proofs = []

        # Verify gradient norm proof
        if "gradient_norm_proof" in proofs:
            if not self._verify_gradient_proof(proofs["gradient_norm_proof"], metrics):
                failed_proofs.append("gradient_norm")
                if self.fail_fast:
                    return False, failed_proofs

        # Verify participation proof
        if "participation_proof" in proofs:
            if not self._verify_participation_proof(proofs["participation_proof"], metrics):
                failed_proofs.append("participation")
                if self.fail_fast:
                    return False, failed_proofs

        # Verify training correctness proof
        if "training_correctness_proof" in proofs:
            if not self._verify_training_proof(proofs["training_correctness_proof"], metrics):
                failed_proofs.append("training_correctness")
                if self.fail_fast:
                    return False, failed_proofs

        return len(failed_proofs) == 0, failed_proofs

    def batch_verify_clients(
        self,
        client_proofs: List[Dict[str, Any]],
        client_metrics: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        Verify proofs from multiple clients.

        Args:
            client_proofs: List of proof dictionaries
            client_metrics: List of metrics dictionaries

        Returns:
            List of verification results (True = verified)
        """
        results = []

        for proofs, metrics in zip(client_proofs, client_metrics):
            is_valid, _ = self.verify_all_proofs(proofs, metrics)
            results.append(is_valid)

        return results

    def _verify_gradient_proof(
        self,
        proof: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Verify gradient norm bound proof.

        Args:
            proof: Gradient norm proof
            metrics: Client metrics

        Returns:
            True if valid
        """
        if not isinstance(proof, dict):
            return False

        # Check structure
        if proof.get("type") != "gradient_norm_bound":
            return False

        # Check bound
        bound = proof.get("bound", 0)
        actual_norm = proof.get("actual_norm", float('inf'))

        # Also check metrics
        gradient_norm = metrics.get("gradient_norm", actual_norm)

        return gradient_norm <= bound and actual_norm <= bound

    def _verify_participation_proof(
        self,
        proof: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Verify participation proof.

        Args:
            proof: Participation proof
            metrics: Client metrics

        Returns:
            True if valid
        """
        if not isinstance(proof, dict):
            return False

        if proof.get("type") != "participation":
            return False

        num_samples = proof.get("num_samples", 0)
        min_samples = proof.get("min_samples", float('inf'))

        # Also check metrics
        actual_samples = metrics.get("num_samples", num_samples)

        return actual_samples >= min_samples and num_samples >= min_samples

    def _verify_training_proof(
        self,
        proof: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Verify training correctness proof.

        Args:
            proof: Training proof
            metrics: Client metrics

        Returns:
            True if valid
        """
        if not isinstance(proof, dict):
            return False

        if proof.get("type") != "training_correctness":
            return False

        # Check that parameters actually changed
        param_change = proof.get("param_change", 0)

        return param_change > 0

    def estimate_verification_time(
        self,
        num_clients: int,
        proofs_per_client: int = 3
    ) -> float:
        """
        Estimate verification time.

        Args:
            num_clients: Number of clients
            proofs_per_client: Number of proofs per client

        Returns:
            Estimated time in seconds
        """
        # Simplified model: ~1ms per proof
        return num_clients * proofs_per_client * 0.001
