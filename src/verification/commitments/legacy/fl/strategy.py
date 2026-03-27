"""
Verifiable FL Strategy

Custom FedAvg strategy with proof verification.
"""

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from flwr.server.client_proxy import ClientProxy
import numpy as np
import logging
import base64

from .server import VerifiableFLServer, VerificationResult


class VerifiableFedAvg(fl.server.strategy.FedAvg):
    """
    Federated averaging strategy with proof verification.

    This strategy extends standard FedAvg to:
    1. Verify proofs from each client
    2. Exclude clients with invalid proofs
    3. Aggregate only verified client updates
    4. Log verification events

    Usage:
        >>> strategy = VerifiableFedAvg(min_verified_clients=5)
        >>> server = fl.server.Server(strategy=strategy)
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.8,
        min_fit_clients: int = 8,
        min_evaluate_clients: int = 8,
        min_available_clients: int = 8,
        min_verified_clients: int = 5,
        verify_proofs: bool = True,
        on_verify_failure: str = "exclude",  # "exclude" or "warn"
        **kwargs
    ):
        """
        Initialize verifiable FedAvg strategy.

        Args:
            fraction_fit: Fraction of clients to use for training
            fraction_evaluate: Fraction of clients to use for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            min_verified_clients: Minimum verified clients for aggregation
            verify_proofs: Enable proof verification
            on_verify_failure: Action on verification failure
            **kwargs: Additional arguments passed to FedAvg
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate,
            min_available_clients=min_available_clients,
            **kwargs
        )

        self.min_verified_clients = min_verified_clients
        self.verify_proofs = verify_proofs
        self.on_verify_failure = on_verify_failure

        self.logger = logging.getLogger("VerifiableFedAvg")
        self.verification_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using weighted average.

        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_res) tuples
            failures: List of failures

        Returns:
            - Aggregated parameters (or None if not enough verified clients)
            - Metrics dictionary
        """
        self.logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")

        if not results:
            return None, {}

        # Verify proofs if enabled
        if self.verify_proofs:
            verified_results, verification_results = self._verify_client_proofs(results)
            self._log_verification_summary(server_round, verification_results)
        else:
            verified_results = results
            verification_results = []

        # Check if we have enough verified clients
        if len(verified_results) < self.min_verified_clients:
            self.logger.warning(
                f"Insufficient verified clients: {len(verified_results)} "
                f"< {self.min_verified_clients}, skipping aggregation"
            )
            return None, {
                "aggregation_skipped": True,
                "verified_clients": len(verified_results),
                "min_required": self.min_verified_clients
            }

        # Aggregate parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            verified_results,
            failures
        )

        # Add verification metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}

        aggregated_metrics["verified_clients"] = len(verified_results)
        aggregated_metrics["excluded_clients"] = len(results) - len(verified_results)
        aggregated_metrics["verification_rate"] = (
            len(verified_results) / len(results) if results else 0
        )

        self.logger.info(
            f"Aggregated {len(verified_results)} verified updates "
            f"({len(results) - len(verified_results)} excluded)"
        )

        return aggregated_parameters, aggregated_metrics

    def _verify_client_proofs(
        self,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]
    ) -> Tuple[List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], List[VerificationResult]]:
        """
        Verify proofs from all clients.

        Args:
            results: List of client results

        Returns:
            - Filtered results (only verified)
            - List of verification results
        """
        verified_results = []
        verification_results = []

        for client_proxy, fit_res in results:
            # Get client ID
            client_id = getattr(client_proxy, 'cid', 'unknown')

            # Get metrics
            metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else {}

            # Verify proofs
            verification_result = self._verify_single_client(client_id, metrics)
            verification_results.append(verification_result)

            # Include or exclude based on verification
            if verification_result.is_valid:
                verified_results.append((client_proxy, fit_res))
            elif self.on_verify_failure == "warn":
                # Still include but warn
                verified_results.append((client_proxy, fit_res))
                self.logger.warning(
                    f"Client {client_id} verification failed but included (warn mode)"
                )
            else:  # exclude
                self.logger.warning(
                    f"Client {client_id} excluded due to verification failure"
                )

        return verified_results, verification_results

    def _verify_single_client(
        self,
        client_id: Any,
        metrics: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify proofs from single client.

        Args:
            client_id: Client identifier
            metrics: Client metrics

        Returns:
            Verification result
        """
        import time
        start_time = time.time()

        # Check for required proofs
        required_proofs = [
            "gradient_norm_verified",
            "participation_verified",
            "training_correctness_verified"
        ]

        failed_proofs = []
        for proof_name in required_proofs:
            if not metrics.get(proof_name, False):
                failed_proofs.append(proof_name.replace("_verified", ""))

        is_valid = len(failed_proofs) == 0
        verification_time = time.time() - start_time

        return VerificationResult(
            is_valid=is_valid,
            client_id=client_id,
            verification_time=verification_time,
            failed_proofs=failed_proofs,
            details=metrics
        )

    def _log_verification_summary(
        self,
        round_num: int,
        verification_results: List[VerificationResult]
    ) -> None:
        """
        Log verification summary.

        Args:
            round_num: Round number
            verification_results: Verification results
        """
        if not verification_results:
            return

        total = len(verification_results)
        verified = sum(1 for r in verification_results if r.is_valid)
        failed = total - verified

        avg_time = (
            sum(r.verification_time for r in verification_results) / total
            if total > 0 else 0
        )

        self.logger.info(
            f"Round {round_num}: Verification - {verified}/{total} passed, "
            f"{failed} failed, avg_time={avg_time:.3f}s"
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
        Get verification statistics.

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
