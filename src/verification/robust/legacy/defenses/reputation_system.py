"""
Client Reputation System for Federated Learning

Tracks client behavior over time to identify and exclude malicious clients.

Features:
- Reputation scoring based on verification results and anomaly detection
- Reputation decay over time (prevents stagnation)
- Client banning for repeated malicious behavior
- Adaptive thresholds for exclusion

Application:
- Prevents repeated attacks from same client
- Catches sophisticated attackers who build reputation slowly
- Complements ZK proofs and Byzantine aggregation
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict


class ClientReputationSystem:
    """
    Reputation system for FL clients.

    Scoring:
    - Initial score: 0.5 (neutral)
    - Good behavior (verified, not anomalous): +reward
    - Bad behavior (not verified, anomalous): -penalty
    - Decay: Scores decay slowly over time (encourages continued good behavior)

    Exclusion:
    - Clients with score < min_reputation are excluded from aggregation
    - Clients with score < ban_threshold are permanently banned
    """

    def __init__(
        self,
        num_clients: int,
        min_reputation: float = 0.3,
        ban_threshold: float = 0.1,
        reward_factor: float = 0.05,
        penalty_factor: float = 0.1,
        decay_rate: float = 0.01
    ):
        """
        Initialize reputation system.

        Args:
            num_clients: Total number of clients
            min_reputation: Minimum reputation to participate
            ban_threshold: Permanently ban if score falls below this
            reward_factor: Reward for good behavior
            penalty_factor: Penalty for bad behavior
            decay_rate: Reputation decay per round
        """
        self.num_clients = num_clients
        self.min_reputation = min_reputation
        self.ban_threshold = ban_threshold
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor
        self.decay_rate = decay_rate

        # Initialize reputation scores
        self.scores = np.ones(num_clients) * 0.5

        # Track banned clients
        self.banned_clients: Set[int] = set()

        # Track history for analysis
        self.history: Dict[int, List[float]] = defaultdict(list)

        # Track consecutive failures
        self.consecutive_failures: Dict[int, int] = defaultdict(int)

        # Track verification history
        self.verification_history: Dict[int, List[bool]] = defaultdict(list)

    def update_reputation(
        self,
        client_id: int,
        verification_passed: bool,
        gradient_anomaly_score: float = 0.0
    ) -> float:
        """
        Update client reputation based on behavior.

        Args:
            client_id: Client identifier
            verification_passed: Did ZK proofs pass?
            gradient_anomaly_score: Anomaly score (0.0 to 1.0)

        Returns:
            New reputation score
        """
        # Check if client is banned
        if client_id in self.banned_clients:
            return self.scores[client_id]

        # Apply decay to all scores
        self._apply_decay()

        # Update this client's score
        current_score = self.scores[client_id]

        if verification_passed and gradient_anomaly_score < 0.5:
            # Good behavior
            new_score = current_score + self.reward_factor
            self.consecutive_failures[client_id] = 0
        else:
            # Bad behavior
            penalty = self.penalty_factor * (1 + gradient_anomaly_score)
            new_score = current_score - penalty
            self.consecutive_failures[client_id] += 1

        # Clip to [0, 1]
        new_score = np.clip(new_score, 0.0, 1.0)

        # Check if should be banned
        if new_score < self.ban_threshold:
            self.banned_clients.add(client_id)
            # Further reduce score
            new_score = 0.0

        # Update score and history
        self.scores[client_id] = new_score
        self.history[client_id].append(new_score)
        self.verification_history[client_id].append(verification_passed)

        return new_score

    def should_exclude(
        self,
        client_id: int
    ) -> bool:
        """
        Check if client should be excluded from aggregation.

        Args:
            client_id: Client identifier

        Returns:
            True if client should be excluded
        """
        # Check if banned
        if client_id in self.banned_clients:
            return True

        # Check if below threshold
        if self.scores[client_id] < self.min_reputation:
            return True

        return False

    def ban_client(
        self,
        client_id: int
    ):
        """
        Permanently ban a client.

        Args:
            client_id: Client identifier
        """
        self.banned_clients.add(client_id)
        self.scores[client_id] = 0.0

    def unban_client(
        self,
        client_id: int,
        reset_score: float = 0.5
    ):
        """
        Unban a client (e.g., for testing).

        Args:
            client_id: Client identifier
            reset_score: Reset reputation to this score
        """
        if client_id in self.banned_clients:
            self.banned_clients.remove(client_id)
            self.scores[client_id] = reset_score

    def get_reputation(
        self,
        client_id: int
    ) -> float:
        """Get client's reputation score."""
        return self.scores[client_id]

    def get_all_reputations(self) -> np.ndarray:
        """Get all reputation scores."""
        return self.scores.copy()

    def get_reputation_statistics(self) -> Dict[str, Any]:
        """
        Get reputation system statistics.

        Returns:
            Dictionary with statistics
        """
        active_clients = sum([1 for i in range(self.num_clients)
                             if i not in self.banned_clients])
        low_reputation_clients = sum([1 for i in range(self.num_clients)
                                    if self.scores[i] < self.min_reputation
                                    and i not in self.banned_clients])

        # Compute verification pass rate
        verification_rates = {}
        for client_id, history in self.verification_history.items():
            if len(history) > 0:
                verification_rates[client_id] = sum(history) / len(history)

        return {
            "num_clients": self.num_clients,
            "num_active": active_clients,
            "num_banned": len(self.banned_clients),
            "num_low_reputation": low_reputation_clients,
            "mean_reputation": float(np.mean(self.scores)),
            "std_reputation": float(np.std(self.scores)),
            "min_reputation": float(np.min(self.scores)),
            "max_reputation": float(np.max(self.scores)),
            "banned_clients": list(self.banned_clients),
            "verification_rates": verification_rates
        }

    def get_client_history(
        self,
        client_id: int
    ) -> List[float]:
        """Get reputation history for a client."""
        return self.history[client_id].copy()

    def _apply_decay(self):
        """Apply decay to all scores (prevent stagnation)."""
        # Decay scores towards 0.5
        self.scores = self.scores - self.decay_rate * (self.scores - 0.5)
        # Clip to [0, 1]
        self.scores = np.clip(self.scores, 0.0, 1.0)

    def reset_all_reputations(self, value: float = 0.5):
        """Reset all reputations (for testing)."""
        self.scores = np.ones(self.num_clients) * value
        self.banned_clients.clear()
        self.history.clear()
        self.consecutive_failures.clear()
        self.verification_history.clear()


class AdaptiveReputationSystem(ClientReputationSystem):
    """
    Adaptive reputation system that adjusts thresholds based on behavior.

    Features:
    - Dynamically adjusts min_reputation based on overall behavior
    - More lenient if many clients are struggling
    - More strict if most clients are well-behaved
    """

    def __init__(
        self,
        num_clients: int,
        min_reputation: float = 0.3,
        ban_threshold: float = 0.1,
        **kwargs
    ):
        super().__init__(
            num_clients=num_clients,
            min_reputation=min_reputation,
            ban_threshold=ban_threshold,
            **kwargs
        )

        self.base_min_reputation = min_reputation

    def update_adaptive_threshold(self):
        """Adjust min_reputation based on overall client behavior."""
        mean_reputation = np.mean(self.scores)

        if mean_reputation < 0.4:
            # Many clients struggling, be more lenient
            self.min_reputation = self.base_min_reputation * 0.8
        elif mean_reputation > 0.7:
            # Most clients well-behaved, be more strict
            self.min_reputation = self.base_min_reputation * 1.2
        else:
            # Normal behavior
            self.min_reputation = self.base_min_reputation

        # Clip to [0.1, 0.8]
        self.min_reputation = np.clip(self.min_reputation, 0.1, 0.8)


# Example usage
if __name__ == "__main__":
    print("Reputation System Demonstration")
    print("=" * 60)

    # Create reputation system
    reputation = ClientReputationSystem(
        num_clients=10,
        min_reputation=0.3,
        ban_threshold=0.1,
        reward_factor=0.05,
        penalty_factor=0.1,
        decay_rate=0.01
    )

    print("Initial reputations:")
    print([f"{s:.2f}" for s in reputation.get_all_reputations()])

    # Simulate 10 rounds
    print("\nSimulating 10 rounds:")
    print("Client 0: Honest (verified, not anomalous)")
    print("Client 1: Malicious (starts attacking at round 5)")

    for round_num in range(1, 11):
        # Client 0: Always honest
        reputation.update_reputation(0, verification_passed=True, gradient_anomaly_score=0.0)

        # Client 1: Honest first 5 rounds, then attacks
        if round_num <= 5:
            reputation.update_reputation(1, verification_passed=True, gradient_anomaly_score=0.1)
        else:
            reputation.update_reputation(1, verification_passed=False, gradient_anomaly_score=0.8)

        # Print every 2 rounds
        if round_num % 2 == 0:
            print(f"\nRound {round_num}:")
            print(f"  Client 0 reputation: {reputation.get_reputation(0):.3f}, "
                  f"excluded: {reputation.should_exclude(0)}")
            print(f"  Client 1 reputation: {reputation.get_reputation(1):.3f}, "
                  f"excluded: {reputation.should_exclude(1)}")

    # Final statistics
    print("\n" + "=" * 60)
    print("Final Statistics:")
    stats = reputation.get_reputation_statistics()
    print(f"  Mean reputation: {stats['mean_reputation']:.3f}")
    print(f"  Active clients: {stats['num_active']}")
    print(f"  Banned clients: {stats['num_banned']}")
    print(f"  Low reputation: {stats['num_low_reputation']}")

    # Client histories
    print("\nClient 0 history (honest):")
    print([f"{s:.2f}" for s in reputation.get_client_history(0)])

    print("\nClient 1 history (malicious):")
    print([f"{s:.2f}" for s in reputation.get_client_history(1)])
