"""
Model Poisoning Attack for Federated Phishing Detection

Attack poisons the model update (gradient) sent to the server.

Attack Types:
1. Gradient Scaling: Scale gradient to dominate aggregation
2. Sign Flip: Flip gradient signs to push model in wrong direction
3. Isotropic Attack: Add random noise to gradient

ZK Proof Detection:
- Gradient Scaling: DETECTED (via norm bound) ✅
- Sign Flip: NOT DETECTED (‖-g‖ = ‖g‖) ❌
- Isotropic: PARTIALLY DETECTED (depends on noise magnitude)

Defenses:
- ZK norm bounds: Prevent scaling
- Byzantine aggregation: Prevent sign flip and isotropic
- Anomaly detection: Detect isotropic with high noise
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any


class ModelPoisoningAttack:
    """
    Model Poisoning Attack for federated learning.

    Attack Strategy:
    - Poison the gradient update before sending to server
    - Goal: Degrade global model performance or insert backdoor

    Detection by ZK Proofs:
    - Scaling: Detected if ‖scaled_gradient‖ > bound
    - Sign flip: NOT detected (‖-g‖ = ‖g‖)
    - Isotropic: Detected if noise is large
    """

    def __init__(
        self,
        attack_type: str = "scaling",
        scaling_factor: float = 10.0,
        noise_std: float = 5.0
    ):
        """
        Initialize model poisoning attack.

        Args:
            attack_type: Type of attack ("scaling", "sign_flip", "isotropic")
            scaling_factor: Factor to scale gradient (for scaling attack)
            noise_std: Standard deviation of noise (for isotropic attack)
        """
        if attack_type not in ["scaling", "sign_flip", "isotropic"]:
            raise ValueError(f"Unknown attack_type: {attack_type}")

        self.attack_type = attack_type
        self.scaling_factor = scaling_factor
        self.noise_std = noise_std

    def poison_gradient(
        self,
        gradient: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Poison gradient before sending to server.

        Args:
            gradient: Original gradient (list of parameter arrays)

        Returns:
            Poisoned gradient
        """
        if self.attack_type == "scaling":
            return self._scale_gradient(gradient)
        elif self.attack_type == "sign_flip":
            return self._sign_flip_gradient(gradient)
        elif self.attack_type == "isotropic":
            return self._isotropic_attack(gradient)
        else:
            raise ValueError(f"Unknown attack_type: {self.attack_type}")

    def _scale_gradient(
        self,
        gradient: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Scale gradient by factor.

        Detection: ZK proofs detect if ‖scaled_gradient‖ > bound
        """
        scaled = [layer * self.scaling_factor for layer in gradient]
        return scaled

    def _sign_flip_gradient(
        self,
        gradient: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Flip gradient signs.

        Detection: ZK proofs CANNOT detect (‖-g‖ = ‖g‖)
        """
        flipped = [-layer for layer in gradient]
        return flipped

    def _isotropic_attack(
        self,
        gradient: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Add isotropic (Gaussian) noise to gradient.

        Detection: ZK proofs detect if noise is large enough
        """
        noisy = []
        for layer in gradient:
            noise = np.random.normal(0, self.noise_std, layer.shape)
            noisy.append(layer + noise)
        return noisy

    def get_attack_info(self) -> Dict[str, Any]:
        """Get attack information."""
        return {
            "attack_type": self.attack_type,
            "scaling_factor": self.scaling_factor if self.attack_type == "scaling" else None,
            "noise_std": self.noise_std if self.attack_type == "isotropic" else None
        }

    def __repr__(self) -> str:
        if self.attack_type == "scaling":
            return f"ModelPoisoningAttack(scaling, factor={self.scaling_factor})"
        elif self.attack_type == "sign_flip":
            return "ModelPoisoningAttack(sign_flip)"
        else:
            return f"ModelPoisoningAttack(isotropic, std={self.noise_std})"


class AdaptiveModelPoisoningAttack(ModelPoisoningAttack):
    """
    Adaptive model poisoning attack that knows ZK bounds.

    This attacker knows the ZK norm bound and crafts attacks that
    stay within the bound to avoid detection.
    """

    def __init__(
        self,
        zk_bound: float = 1.0,
        base_attack_type: str = "scaling"
    ):
        """
        Initialize adaptive attack.

        Args:
            zk_bound: ZK proof norm bound
            base_attack_type: Base attack type for initialization ("scaling", "sign_flip", "isotropic")
        """
        super().__init__(attack_type=base_attack_type)
        self.zk_bound = zk_bound

    def craft_within_bounds(
        self,
        true_gradient: List[np.ndarray],
        attack_type: str = "scaling"
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Craft attack that stays within ZK norm bound.

        Args:
            true_gradient: Honest gradient
            attack_type: "scaling" or "noise"

        Returns:
            (malicious_gradient, attack_info)
        """
        # Compute original norm
        original_norm = self._compute_norm(true_gradient)

        if attack_type == "scaling":
            # Scale as much as possible without exceeding bound
            if original_norm > 0:
                max_scaling = self.zk_bound / original_norm
                # Use 95% of max to stay safely within bound
                scaling_factor = max_scaling * 0.95
            else:
                scaling_factor = 1.0

            malicious_gradient = [layer * scaling_factor for layer in true_gradient]

            attack_info = {
                "attack_type": "adaptive_scaling",
                "original_norm": float(original_norm),
                "zk_bound": self.zk_bound,
                "scaling_factor": scaling_factor,
                "new_norm": float(self._compute_norm(malicious_gradient)),
                "within_bound": self._compute_norm(malicious_gradient) <= self.zk_bound
            }

        elif attack_type == "noise":
            # Add as much noise as possible without exceeding bound
            current_norm = original_norm
            max_additional_norm = self.zk_bound - current_norm

            if max_additional_norm > 0:
                # Add noise
                noise_std = max_additional_norm * 0.5
                malicious_gradient = []
                for layer in true_gradient:
                    noise = np.random.normal(0, noise_std, layer.shape)
                    malicious_gradient.append(layer + noise)
            else:
                malicious_gradient = true_gradient
                noise_std = 0.0

            attack_info = {
                "attack_type": "adaptive_noise",
                "original_norm": float(original_norm),
                "zk_bound": self.zk_bound,
                "noise_std": noise_std,
                "new_norm": float(self._compute_norm(malicious_gradient)),
                "within_bound": self._compute_norm(malicious_gradient) <= self.zk_bound
            }
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

        return malicious_gradient, attack_info

    def _compute_norm(self, gradient: List[np.ndarray]) -> float:
        """Compute L2 norm of gradient."""
        squared_norm = 0.0
        for layer in gradient:
            squared_norm += np.sum(layer ** 2)
        return np.sqrt(squared_norm)


def compute_gradient_norm(gradient: List[np.ndarray]) -> float:
    """
    Compute L2 norm of gradient.

    Args:
        gradient: List of parameter arrays

    Returns:
        L2 norm
    """
    squared_norm = 0.0
    for layer in gradient:
        squared_norm += np.sum(layer ** 2)
    return np.sqrt(squared_norm)


# Example usage
if __name__ == "__main__":
    print("Model Poisoning Attack Demonstration")
    print("=" * 60)

    # Create honest gradient
    honest_gradient = [np.random.randn(10, 10) * 0.1 for _ in range(3)]
    honest_norm = compute_gradient_norm(honest_gradient)

    print(f"Honest gradient norm: {honest_norm:.4f}")

    # Test scaling attack
    print("\n--- Scaling Attack ---")
    scaling_attack = ModelPoisoningAttack(attack_type="scaling", scaling_factor=10.0)
    scaled_gradient = scaling_attack.poison_gradient(honest_gradient)
    scaled_norm = compute_gradient_norm(scaled_gradient)

    print(f"Scaled gradient norm: {scaled_norm:.4f}")
    print(f"Scaling factor: {scaled_norm / honest_norm:.2f}x")
    print(f"ZK detection (bound=1.0): {'DETECTED' if scaled_norm > 1.0 else 'NOT DETECTED'}")

    # Test sign flip attack
    print("\n--- Sign Flip Attack ---")
    sign_flip_attack = ModelPoisoningAttack(attack_type="sign_flip")
    flipped_gradient = sign_flip_attack.poison_gradient(honest_gradient)
    flipped_norm = compute_gradient_norm(flipped_gradient)

    print(f"Flipped gradient norm: {flipped_norm:.4f}")
    print(f"Norm change: {(flipped_norm - honest_norm) / honest_norm * 100:.2f}%")
    print(f"ZK detection (bound=1.0): {'DETECTED' if flipped_norm > 1.0 else 'NOT DETECTED'}")

    # Test adaptive attack
    print("\n--- Adaptive Attack (knows ZK bound) ---")
    adaptive_attack = AdaptiveModelPoisoningAttack(zk_bound=1.0)
    adaptive_gradient, attack_info = adaptive_attack.craft_within_bounds(
        honest_gradient,
        attack_type="scaling"
    )

    print(f"Adaptive scaling factor: {attack_info['scaling_factor']:.4f}")
    print(f"Adaptive gradient norm: {attack_info['new_norm']:.4f}")
    print(f"ZK bound: {attack_info['zk_bound']}")
    print(f"Within bound: {attack_info['within_bound']}")
    print(f"ZK detection: {'BYPASSED' if attack_info['within_bound'] else 'DETECTED'}")
