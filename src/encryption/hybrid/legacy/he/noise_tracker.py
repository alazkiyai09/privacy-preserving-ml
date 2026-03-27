"""
Noise Budget Tracking for Homomorphic Encryption
==============================================

Tracks noise consumption in CKKS scheme to prevent budget exhaustion.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class NoiseWarning(Enum):
    """Warning levels for noise consumption."""
    SAFE = "safe"  # Plenty of budget remaining
    MODERATE = "moderate"  # Should monitor closely
    HIGH = "high"  # Approaching limit
    CRITICAL = "critical"  # Budget nearly exhausted


@dataclass
class NoiseConsumption:
    """Record of a single noise consumption operation."""
    operation: str
    noise_consumed: int
    remaining_budget: int
    timestamp: float
    warning_level: NoiseWarning


@dataclass
class NoiseBudget:
    """
    Noise budget manager for CKKS computations.

    Tracks how much noise has been consumed and warns when approaching limits.
    """

    def __init__(
        self,
        initial_budget: int = 200,
        warning_threshold: float = 0.8  # Warn at 80% consumption
    ):
        """
        Initialize noise budget tracker.

        Args:
            initial_budget: Total noise budget in bits
            warning_threshold: Fraction of budget to warn at (0.0-1.0)
        """
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.warning_threshold = warning_threshold
        self.total_consumed = 0
        self.consumption_history: List[NoiseConsumption] = []

    def consume_noise(
        self,
        amount: int,
        operation: str,
        allow_exceed: bool = False
    ) -> bool:
        """
        Consume noise from budget.

        Args:
            amount: Noise amount to consume in bits
            operation: Description of operation
            allow_exceed: If False, raises exception when budget exceeded

        Returns:
            True if consumption successful

        Raises:
            NoiseBudgetExceededError: If budget would be exceeded
        """
        if amount > self.current_budget and not allow_exceed:
            raise ValueError(
                f"Noise budget would be exceeded! "
                f"Tried to consume {amount} bits but only "
                f"{self.current_budget} bits remaining. "
                f"Operation: {operation}"
            )

        self.current_budget -= amount
        self.total_consumed += amount

        # Determine warning level
        consumption_ratio = self.total_consumed / self.initial_budget

        if consumption_ratio >= 1.0:
            warning = NoiseWarning.CRITICAL
        elif consumption_ratio >= self.warning_threshold:
            warning = NoiseWarning.HIGH
        elif consumption_ratio >= 0.5:
            warning = NoiseWarning.MODERATE
        else:
            warning = NoiseWarning.SAFE

        # Record consumption
        consumption = NoiseConsumption(
            operation=operation,
            noise_consumed=amount,
            remaining_budget=self.current_budget,
            timestamp=time.time(),
            warning_level=warning
        )

        self.consumption_history.append(consumption)

        return True

    def get_remaining_budget(self) -> int:
        """Get remaining noise budget."""
        return self.current_budget

    def get_consumed_budget(self) -> int:
        """Get total consumed budget."""
        return self.total_consumed

    def get_budget_percentage(self) -> float:
        """Get percentage of budget consumed."""
        return (self.total_consumed / self.initial_budget) * 100 if self.initial_budget > 0 else 0.0

    def get_warning_level(self) -> NoiseWarning:
        """Get current warning level based on consumption."""
        consumption_ratio = self.total_consumed / self.initial_budget

        if consumption_ratio >= 1.0:
            return NoiseWarning.CRITICAL
        elif consumption_ratio >= self.warning_threshold:
            return NoiseWarning.HIGH
        elif consumption_ratio >= 0.5:
            return NoiseWarning.MODERATE
        else:
            return NoiseWarning.SAFE

    def can_perform_operation(
        self,
        noise_cost: int
    ) -> bool:
        """
        Check if operation fits in noise budget.

        Args:
            noise_cost: Estimated noise cost of operation

        Returns:
            True if operation fits in budget
        """
        return noise_cost <= self.current_budget

    def estimate_noise_cost(
        self,
        input_size: int,
        output_size: int,
        operation: str = "linear"
    ) -> int:
        """
        Estimate noise cost for an operation.

        Args:
            input_size: Input dimensionality
            output_size: Output dimensionality
            operation: Type of operation

        Returns:
            Estimated noise consumption in bits
        """
        if operation == "linear":
            # Matrix multiplication: input_size * output_size multiplications
            # Each consumes noise_consumption_per_mul bits
            # Simplified model
            mul_cost = (input_size * output_size) // 10  # Reduce for efficiency
            add_cost = output_size  # Bias addition

            # Scale down for realistic estimate
            total_cost = (mul_cost + add_cost) // 2

            return total_cost

        elif operation == "add":
            # Vector addition
            return output_size // 2

        else:
            # Default estimate
            return input_size  # Conservative estimate

    def print_status(self) -> None:
        """Print noise budget status."""
        warning = self.get_warning_level()

        print("\n" + "=" * 70)
        print("Noise Budget Status")
        print("=" * 70)
        print(f"Initial Budget: {self.initial_budget} bits")
        print(f"Consumed: {self.total_consumed} bits")
        print(f"Remaining: {self.current_budget} bits")
        print(f"Percentage Used: {self.get_budget_percentage():.1f}%")
        print(f"Warning Level: {warning.value.upper()}")
        print(f"Operations Performed: {len(self.consumption_history)}")
        print("=" * 70 + "\n")

    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """
        Get detailed consumption history.

        Returns:
            List of consumption records
        """
        return [
            {
                'operation': c.operation,
                'noise_consumed': c.noise_consumed,
                'remaining_budget': c.remaining_budget,
                'timestamp': c.timestamp,
                'warning_level': c.warning_level.value,
            }
            for c in self.consumption_history
        ]

    def reset(self, new_budget: Optional[int] = None) -> None:
        """
        Reset noise budget.

        Args:
            new_budget: New budget (uses initial if None)
        """
        if new_budget is None:
            new_budget = self.initial_budget

        self.initial_budget = new_budget
        self.current_budget = new_budget
        self.total_consumed = 0
        self.consumption_history = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get noise budget summary.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'initial_budget': self.initial_budget,
            'consumed': self.total_consumed,
            'remaining': self.current_budget,
            'percentage_used': self.get_budget_percentage(),
            'warning_level': self.get_warning_level().value,
            'operations_count': len(self.consumption_history),
        }


def create_noise_tracker(initial_budget: int = 200) -> NoiseBudget:
    """
    Factory function to create noise tracker.

    Args:
        initial_budget: Initial noise budget in bits

    Returns:
        NoiseBudget instance
    """
    return NoiseBudget(initial_budget=initial_budget)


def estimate_layer_noise_cost(
    layer_spec,
    scale_bits: int = 40
) -> int:
    """
    Estimate noise cost for a layer.

    Args:
        layer_spec: Layer specification
        scale_bits: CKKS scale parameter

    Returns:
        Estimated noise cost in bits

    Note:
        This is a simplified estimation. Actual noise consumption
        depends on many factors in TenSEAL.
    """
    # Linear layer cost
    if layer_spec.layer_type == "linear":
        # Simplified: output_size * scale_bits / 10
        linear_cost = (layer_spec.output_size * scale_bits) // 10
        return linear_cost

    # For other layers (TEE operations), no HE noise cost
    return 0


def calculate_total_noise_cost(
    layers: list,
    scale_bits: int = 40
) -> int:
    """
    Calculate total noise cost for all HE layers.

    Args:
        layers: List of layer specifications
        scale_bits: CKKS scale parameter

    Returns:
        Total noise cost in bits
    """
    total = 0

    for layer in layers:
        total += estimate_layer_noise_cost(layer, scale_bits)

    return total
