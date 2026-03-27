"""
Verifiable FL Server

Flower server with proof verification capabilities.
"""

import flwr as fl
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
import logging
from dataclasses import dataclass

from ..proofs.proof_aggregator import ProofAggregator


@dataclass
class VerificationResult:
    """Result of proof verification."""
    is_valid: bool
    client_id: int
    verification_time: float
    failed_proofs: List[str]
    details: Dict[str, Any]


class VerifiableFLServer:
    """
    Federated learning server with proof verification.

    This server extends the standard Flower server to verify zero-knowledge
    proofs from clients before aggregating their updates.

    Usage:
        >>> server = VerifiableFLServer(model, verifier, logger)
        >>> history = server.start_server()
    """

    def __init__(
        self,
        strategy,
        proof_verifier: Optional[ProofAggregator] = None,
        min_verified_clients: int = 5,
        log_verification: bool = True
    ):
        """
        Initialize verifiable FL server.

        Args:
            strategy: FL aggregation strategy
            proof_verifier: Proof verification utility
            min_verified_clients: Minimum verified clients for aggregation
            log_verification: Whether to log verification events
        """
        self.strategy = strategy
        self.proof_verifier = proof_verifier
        self.min_verified_clients = min_verified_clients
        self.log_verification = log_verification

        self.logger = logging.getLogger("Server")
        self.verification_history = []

    def configure_server(self) -> fl.server.Server:
        """
        Configure and return Flower server.

        Returns:
            Configured Flower server
        """
        return fl.server.Server(
            client_manager=fl.server.simple.SimpleClientManager(),
            strategy=self.strategy
        )

    def verify_client_update(
        self,
        client_id: int,
        metrics: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify proofs from client update.

        Args:
            client_id: Client identifier
            metrics: Client metrics containing proofs

        Returns:
            Verification result
        """
        start_time = time.time()

        # Extract proofs from metrics
        proofs = {
            "gradient_norm_proof": metrics.get("gradient_norm_proof"),
            "participation_proof": metrics.get("participation_proof"),
            "training_correctness_proof": metrics.get("training_correctness_proof")
        }

        # Verify proofs if verifier available
        if self.proof_verifier:
            is_valid, failed_proofs = self.proof_verifier.verify_all_proofs(
                proofs, metrics
            )
        else:
            # Simple check: verify flags in metrics
            is_valid = all([
                metrics.get("gradient_norm_verified", False),
                metrics.get("participation_verified", False),
                metrics.get("training_correctness_verified", False)
            ])
            failed_proofs = []
            if not metrics.get("gradient_norm_verified", False):
                failed_proofs.append("gradient_norm")
            if not metrics.get("participation_verified", False):
                failed_proofs.append("participation")
            if not metrics.get("training_correctness_verified", False):
                failed_proofs.append("training_correctness")

        verification_time = time.time() - start_time

        result = VerificationResult(
            is_valid=is_valid,
            client_id=client_id,
            verification_time=verification_time,
            failed_proofs=failed_proofs,
            details=metrics
        )

        # Log result
        if self.log_verification:
            if is_valid:
                self.logger.info(
                    f"Client {client_id}: Verification PASSED ({verification_time:.3f}s)"
                )
            else:
                self.logger.warning(
                    f"Client {client_id}: Verification FAILED - {failed_proofs}"
                )

        return result

    def filter_verified_clients(
        self,
        results: List[Tuple[Any, Any]]
    ) -> Tuple[List[Tuple[Any, Any]], List[VerificationResult]]:
        """
        Filter clients with valid proofs.

        Args:
            results: List of (client_proxy, fit_res) tuples

        Returns:
            - Filtered results (only verified clients)
            - List of verification results
        """
        verified_results = []
        verification_results = []

        for client_proxy, fit_res in results:
            # Get client ID and metrics
            client_id = getattr(client_proxy, 'cid', 'unknown')
            metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else {}

            # Verify proofs
            if 'gradient_norm_proof' in metrics or self.proof_verifier is None:
                # Client sent proofs
                verification_result = self.verify_client_update(client_id, metrics)
                verification_results.append(verification_result)

                if verification_result.is_valid:
                    verified_results.append((client_proxy, fit_res))
            else:
                # Client didn't send proofs (baseline mode)
                verified_results.append((client_proxy, fit_res))

        return verified_results, verification_results

    def log_verification_summary(
        self,
        round_num: int,
        verification_results: List[VerificationResult]
    ) -> None:
        """
        Log verification summary for round.

        Args:
            round_num: Current round number
            verification_results: List of verification results
        """
        total = len(verification_results)
        verified = sum(1 for r in verification_results if r.is_valid)
        failed = total - verified

        avg_time = (
            sum(r.verification_time for r in verification_results) / total
            if total > 0 else 0
        )

        self.logger.info(
            f"Round {round_num} Verification Summary: "
            f"{verified}/{total} passed, {failed} failed, "
            f"avg_time={avg_time:.3f}s"
        )

        # Log failures
        for result in verification_results:
            if not result.is_valid:
                self.logger.warning(
                    f"  Client {result.client_id} failed: {result.failed_proofs}"
                )

        # Store history
        self.verification_history.append({
            "round": round_num,
            "total": total,
            "verified": verified,
            "failed": failed,
            "avg_time": avg_time
        })

    def get_verification_statistics(self) -> Dict[str, Any]:
        """
        Get overall verification statistics.

        Returns:
            Statistics dictionary
        """
        if not self.verification_history:
            return {}

        total_rounds = len(self.verification_history)
        total_clients = sum(h["total"] for h in self.verification_history)
        total_verified = sum(h["verified"] for h in self.verification_history)
        total_failed = sum(h["failed"] for h in self.verification_history)
        avg_time = sum(h["avg_time"] for h in self.verification_history) / total_rounds

        return {
            "total_rounds": total_rounds,
            "total_clients": total_clients,
            "total_verified": total_verified,
            "total_failed": total_failed,
            "verification_rate": total_verified / total_clients if total_clients > 0 else 0,
            "average_verification_time": avg_time
        }
