"""
Adaptive Attacker for Federated Phishing Detection

Sophisticated attacker who knows the defense mechanisms and crafts attacks
to bypass them.

Attacker Knowledge:
- Knows ZK norm bound
- Knows Byzantine aggregation method
- Knows anomaly detection threshold
- Knows reputation system

Attack Strategies:
1. Reputation Building: Act honest initially, then attack
2. Bound-Aware Scaling: Scale gradient just below ZK bound
3. Gradient Blending: Blend malicious gradient with honest ones
4. Colluding: Coordinate with other malicious clients

This is the most challenging attack to defend against.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Set


class AdaptiveAttacker:
    """
    Adaptive attacker who knows defense mechanisms.

    Knowledge:
    - ZK norm bound: Stays within bound to avoid detection
    - Byzantine method: Crafts gradient to bypass aggregation
    - Reputation system: Builds reputation before attacking

    Attack Strategy:
    - Early rounds: Act honest to build reputation
    - Later rounds: Launch sophisticated attacks
    """

    def __init__(
        self,
        client_id: int,
        knows_zk_bound: bool = True,
        knows_byzantine: bool = True,
        knows_reputation: bool = True,
        zk_bound: float = 1.0
    ):
        """
        Initialize adaptive attacker.

        Args:
            client_id: Client identifier
            knows_zk_bound: Attacker knows ZK norm bound
            knows_byzantine: Attacker knows Byzantine method
            knows_reputation: Attacker knows about reputation system
            zk_bound: ZK norm bound (if known)
        """
        self.client_id = client_id
        self.knows_zk_bound = knows_zk_bound
        self.knows_byzantine = knows_byzantine
        self.knows_reputation = knows_reputation
        self.zk_bound = zk_bound

        # Reputation building state
        self.reputation_built = False
        self.reputation_build_rounds = 0

    def craft_attack(
        self,
        true_gradient: List[np.ndarray],
        context: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Craft attack based on defense knowledge.

        Args:
            true_gradient: Honest gradient
            context: Attack context including:
                - round: Current round number
                - zk_bound: ZK norm bound (may override self.zk_bound)
                - byzantine_method: Byzantine aggregation method
                - other_gradients: Gradients from other clients (for blending)

        Returns:
            (malicious_gradient, attack_info)
        """
        round_num = context.get("round", 0)
        zk_bound = context.get("zk_bound", self.zk_bound)

        # If building reputation, act honest
        if self.knows_reputation and round_num < 5:
            # Act honest for first 5 rounds
            return self._act_honest(true_gradient, round_num)

        # Launch attack
        if self.knows_zk_bound:
            # Craft attack within ZK bound
            if self.knows_byzantine and "other_gradients" in context:
                # Also blend to avoid Byzantine detection
                return self._craft_bounded_blended(
                    true_gradient,
                    context["other_gradients"],
                    zk_bound
                )
            else:
                # Just stay within ZK bound
                return self._craft_bounded_attack(true_gradient, zk_bound)
        else:
            # Don't know ZK bound, use naive attack
            return self._craft_naive_attack(true_gradient)

    def _act_honest(
        self,
        true_gradient: List[np.ndarray],
        round_num: int
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Act honest to build reputation."""
        # Return honest gradient with tiny modification
        # (to show we're participating)
        honest = [layer.copy() for layer in true_gradient]

        # Add tiny noise to show participation
        for i in range(len(honest)):
            noise = np.random.normal(0, 0.001, honest[i].shape)
            honest[i] += noise

        return honest, {
            "attack_type": "honest",
            "round": round_num,
            "building_reputation": True
        }

    def _craft_bounded_attack(
        self,
        true_gradient: List[np.ndarray],
        zk_bound: float
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Craft attack that stays within ZK bound."""
        # Compute original norm
        original_norm = self._compute_norm(true_gradient)

        # Scale as much as possible without exceeding bound
        if original_norm > 0:
            max_scaling = zk_bound / original_norm
            scaling_factor = max_scaling * 0.95  # Stay 5% below bound
        else:
            scaling_factor = 1.0

        malicious = [layer * scaling_factor for layer in true_gradient]
        new_norm = self._compute_norm(malicious)

        return malicious, {
            "attack_type": "bounded_scaling",
            "original_norm": float(original_norm),
            "zk_bound": zk_bound,
            "scaling_factor": scaling_factor,
            "new_norm": float(new_norm),
            "within_bound": new_norm <= zk_bound
        }

    def _craft_bounded_blended(
        self,
        true_gradient: List[np.ndarray],
        other_gradients: List[List[np.ndarray]],
        zk_bound: float
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Craft attack that blends with honest gradients."""
        # First, scale within bound
        malicious, info = self._craft_bounded_attack(true_gradient, zk_bound)

        # Then, blend towards centroid of other gradients
        # (to avoid Byzantine detection)
        if len(other_gradients) > 0:
            # Compute centroid
            centroid = self._compute_centroid(other_gradients)

            # Blend: 70% malicious, 30% centroid
            blended = []
            for mal, cent in zip(malicious, centroid):
                blended.append(mal * 0.7 + cent * 0.3)

            new_norm = self._compute_norm(blended)

            return blended, {
                "attack_type": "bounded_blended",
                **info,
                "blended_with_centroid": True,
                "blend_ratio": 0.3,
                "final_norm": float(new_norm)
            }

        return malicious, info

    def _craft_naive_attack(
        self,
        true_gradient: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Craft naive attack (doesn't know ZK bound)."""
        # Scale by large factor (will likely be detected)
        malicious = [layer * 10.0 for layer in true_gradient]
        new_norm = self._compute_norm(malicious)

        return malicious, {
            "attack_type": "naive_scaling",
            "scaling_factor": 10.0,
            "new_norm": float(new_norm),
            "within_bound": new_norm <= self.zk_bound
        }

    def _compute_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)

    def _compute_centroid(
        self,
        gradients: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Compute centroid of gradients."""
        if len(gradients) == 0:
            return []

        # Average each layer
        centroid = []
        num_layers = len(gradients[0])

        for layer_idx in range(num_layers):
            # Stack all layers at this index
            layers = [grad[layer_idx] for grad in gradients]
            # Compute average
            centroid.append(np.mean(layers, axis=0))

        return centroid


class SophisticatedAttacker(AdaptiveAttacker):
    """
    Sophisticated attacker with reputation building strategy.

    Strategy:
    1. Act honest for first N rounds (build reputation)
    2. Gradually increase attack strength
    3. If detected, stop attacking for a few rounds
    4. Resume attack with lower profile
    """

    def __init__(
        self,
        client_id: int,
        zk_bound: float = 1.0,
        reputation_build_rounds: int = 5,
        attack_intensity: float = 0.5
    ):
        """
        Initialize sophisticated attacker.

        Args:
            client_id: Client identifier
            zk_bound: ZK norm bound
            reputation_build_rounds: Rounds to build reputation before attacking
            attack_intensity: Attack intensity (0.0 to 1.0)
        """
        super().__init__(
            client_id=client_id,
            knows_zk_bound=True,
            knows_byzantine=True,
            knows_reputation=True,
            zk_bound=zk_bound
        )

        self.reputation_build_rounds = reputation_build_rounds
        self.attack_intensity = attack_intensity
        self.consecutive_detections = 0

    def craft_attack(
        self,
        true_gradient: List[np.ndarray],
        context: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Craft sophisticated attack."""
        round_num = context.get("round", 0)
        was_detected = context.get("was_detected", False)

        # If recently detected, act honest to recover
        if was_detected:
            self.consecutive_detections += 1
            if self.consecutive_detections < 3:
                # Lay low for 3 rounds
                return self._act_honest(true_gradient, round_num)
        else:
            self.consecutive_detections = 0

        # Build reputation first
        if round_num < self.reputation_build_rounds:
            return self._act_honest(true_gradient, round_num)

        # Gradual attack: intensity increases with rounds
        round_since_build = round_num - self.reputation_build_rounds
        current_intensity = min(
            self.attack_intensity * (1 + round_since_build * 0.1),
            0.95  # Cap at 95% of bound
        )

        # Scale gradient based on intensity
        original_norm = self._compute_norm(true_gradient)
        if original_norm > 0:
            max_scaling = self.zk_bound / original_norm
            scaling_factor = max_scaling * current_intensity
        else:
            scaling_factor = 1.0

        malicious = [layer * scaling_factor for layer in true_gradient]
        new_norm = self._compute_norm(malicious)

        return malicious, {
            "attack_type": "sophisticated_scaling",
            "round": round_num,
            "reputation_built": True,
            "attack_intensity": current_intensity,
            "scaling_factor": scaling_factor,
            "new_norm": float(new_norm),
            "within_bound": new_norm <= self.zk_bound
        }


class ColludingAttacker:
    """
    Colluding attacker that coordinates with other malicious clients.

    Strategy:
    - Multiple clients coordinate their attacks
    - Overwhelm Byzantine aggregation (which assumes ≤ f malicious)
    - If n=10 and f=2, 3 colluding clients can break the system
    """

    def __init__(
        self,
        malicious_ids: Set[int],
        zk_bound: float = 1.0,
        knows_byzantine: bool = True,
        byzantine_num_malicious: int = 2
    ):
        """
        Initialize colluding attacker.

        Args:
            malicious_ids: Set of malicious client IDs
            zk_bound: ZK norm bound
            knows_byzantine: Attacker knows Byzantine threshold
            byzantine_num_malicious: System's assumption about max malicious
        """
        self.malicious_ids = malicious_ids
        self.zk_bound = zk_bound
        self.knows_byzantine = knows_byzantine
        self.byzantine_num_malicious = byzantine_num_malicious

    def coordinate_attack(
        self,
        true_gradient: List[np.ndarray],
        context: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Coordinate attack with other malicious clients.

        Args:
            true_gradient: This client's honest gradient
            context:
                - round: Current round
                - num_malicious: Total number of malicious clients
                - my_index: This client's index in malicious set

        Returns:
            (malicious_gradient, attack_info)
        """
        num_malicious = context.get("num_malicious", len(self.malicious_ids))

        # Check if we can overwhelm Byzantine defense
        can_overwhelm = num_malicious > self.byzantine_num_malicious

        if can_overwhelm:
            # We're the majority! Use large scaling
            malicious = [layer * 0.95 for layer in true_gradient]  # Stay within bound
            attack_strategy = "overwhelm"
        else:
            # We're minority, be subtle
            original_norm = self._compute_norm(true_gradient)
            if original_norm > 0:
                max_scaling = self.zk_bound / original_norm
                scaling_factor = max_scaling * 0.9
            else:
                scaling_factor = 1.0

            malicious = [layer * scaling_factor for layer in true_gradient]
            attack_strategy = "subtle"

        new_norm = self._compute_norm(malicious)

        return malicious, {
            "attack_type": "colluding",
            "attack_strategy": attack_strategy,
            "num_malicious": num_malicious,
            "can_overwhelm": can_overwhelm,
            "new_norm": float(new_norm),
            "within_bound": new_norm <= self.zk_bound
        }

    def _compute_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)


# Example usage
if __name__ == "__main__":
    print("Adaptive Attacker Demonstration")
    print("=" * 60)

    # Create honest gradient
    honest_gradient = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    honest_norm = AdaptiveAttacker._compute_norm(None, honest_gradient)

    print(f"Honest gradient norm: {honest_norm:.4f}")

    # Test adaptive attacker
    print("\n--- Adaptive Attacker (knows ZK bound) ---")
    adaptive = AdaptiveAttacker(
        client_id=0,
        knows_zk_bound=True,
        zk_bound=1.0
    )

    context = {"round": 10, "zk_bound": 1.0}
    malicious, info = adaptive.craft_attack(honest_gradient, context)

    print(f"Attack type: {info['attack_type']}")
    print(f"Malicious norm: {info.get('new_norm', 0):.4f}")
    print(f"Within bound: {info.get('within_bound', False)}")
    print(f"Scaling factor: {info.get('scaling_factor', 0):.4f}")

    # Test sophisticated attacker
    print("\n--- Sophisticated Attacker (builds reputation) ---")
    sophisticated = SophisticatedAttacker(
        client_id=1,
        zk_bound=1.0,
        reputation_build_rounds=5
    )

    for round_num in [3, 5, 7, 10]:
        malicious, info = sophisticated.craft_attack(honest_gradient, {"round": round_num})
        print(f"Round {round_num}: {info['attack_type']}, "
              f"norm={info.get('new_norm', 0):.4f}, "
              f"reputation_built={info.get('reputation_built', False)}")

    # Test colluding attacker
    print("\n--- Colluding Attacker (3 clients, system assumes 2) ---")
    colluding = ColludingAttacker(
        malicious_ids={0, 1, 2},
        zk_bound=1.0,
        byzantine_num_malicious=2
    )

    context = {"round": 1, "num_malicious": 3}
    malicious, info = colluding.coordinate_attack(honest_gradient, context)

    print(f"Attack strategy: {info['attack_strategy']}")
    print(f"Can overwhelm: {info['can_overwhelm']}")
    print(f"Malicious norm: {info['new_norm']:.4f}")
