"""
Logging Utilities

Setup logging for FL experiments and security events.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_to_file: bool = True
) -> None:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_to_file: Whether to log to file
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Add file handler if requested
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"fl_experiment_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        logging.getLogger().addHandler(file_handler)


class SecurityLogger:
    """
    Logger for security events (verification failures, attacks, etc.)
    """

    def __init__(
        self,
        log_file: str = "logs/security.log",
        audit_log_file: str = "logs/audit.log"
    ):
        """
        Initialize security logger.

        Args:
            log_file: Path to security log
            audit_log_file: Path to audit log
        """
        # Create directories
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname(audit_log_file), exist_ok=True)

        # Setup security logger
        self.security_logger = logging.getLogger("Security")
        self.security_logger.setLevel(logging.INFO)

        security_handler = logging.FileHandler(log_file)
        security_handler.setFormatter(
            logging.Formatter('%(asctime)s - SECURITY - %(message)s')
        )
        self.security_logger.addHandler(security_handler)

        # Setup audit logger
        self.audit_logger = logging.getLogger("Audit")
        self.audit_logger.setLevel(logging.INFO)

        audit_handler = logging.FileHandler(audit_log_file)
        audit_handler.setFormatter(
            logging.Formatter('%(asctime)s - AUDIT - %(message)s')
        )
        self.audit_logger.addHandler(audit_handler)

    def log_verification_failure(
        self,
        client_id: int,
        failed_proofs: List[str],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log verification failure.

        Args:
            client_id: Client ID
            failed_proofs: List of failed proof names
            details: Additional details
        """
        self.security_logger.warning(
            f"Client {client_id} verification failed - "
            f"Failed proofs: {failed_proofs}"
        )

        self.audit_logger.info(
            f"VERIFICATION_FAILURE - Client {client_id} - "
            f"Proofs: {failed_proofs} - Details: {details}"
        )

    def log_attack_detected(
        self,
        client_id: int,
        attack_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log detected attack.

        Args:
            client_id: Malicious client ID
            attack_type: Type of attack
            details: Attack details
        """
        self.security_logger.error(
            f"ATTACK DETECTED - Client {client_id} - "
            f"Type: {attack_type}"
        )

        self.audit_logger.critical(
            f"ATTACK - Client {client_id} - "
            f"Type: {attack_type} - Details: {details}"
        )

    def log_client_excluded(
        self,
        client_id: int,
        reason: str
    ) -> None:
        """
        Log client exclusion.

        Args:
            client_id: Client ID
            reason: Exclusion reason
        """
        self.security_logger.info(
            f"Client {client_id} excluded - Reason: {reason}"
        )

        self.audit_logger.info(
            f"CLIENT_EXCLUDED - Client {client_id} - Reason: {reason}"
        )

    def log_round_summary(
        self,
        round_num: int,
        verified: int,
        excluded: int,
        total: int
    ) -> None:
        """
        Log round verification summary.

        Args:
            round_num: Round number
            verified: Number of verified clients
            excluded: Number of excluded clients
            total: Total number of clients
        """
        self.security_logger.info(
            f"Round {round_num} - {verified}/{total} verified, "
            f"{excluded} excluded"
        )

        self.audit_logger.info(
            f"ROUND_SUMMARY - Round {round_num} - "
            f"Verified: {verified}/{total}, Excluded: {excluded}"
        )
