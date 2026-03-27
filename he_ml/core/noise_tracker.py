"""
Noise Budget Tracking and Analysis
===================================
Tracks noise growth through homomorphic operations to determine
maximum circuit depth and optimize computation order.

CRITICAL CONCEPTS:
- Initial noise budget: Determined by coeff_mod_bit_sizes (sum of prime bit sizes)
- Noise consumption: Each operation adds noise
- Decryption fails when noise budget reaches zero
- Multiplication is MUCH more expensive than addition

NOISE GROWTH (approximate):
- Addition: Minimal noise growth (~few bits)
- Ciphertext-Plaintext Multiplication: Moderate (~log(scale) bits)
- Ciphertext-Ciphertext Multiplication: High (~2*log(scale) bits)
- Relinearization: Reduces noise but doesn't eliminate it
"""

from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import tenseal as ts
import warnings


class OperationType(Enum):
    """Types of homomorphic operations for noise tracking."""
    ADD = "addition"
    ADD_PLAIN = "addition_plain"
    MULT = "multiplication"
    MULT_PLAIN = "multiplication_plain"
    RELIN = "relinearization"
    RESCALE = "rescaling"  # CKKS only
    ROTATE = "rotation"     # Requires Galois keys


@dataclass
class NoiseOperation:
    """Record of an operation and its noise cost."""
    op_type: OperationType
    depth: int  # Circuit depth when this operation occurred
    noise_before: int  # Bits remaining before operation
    noise_after: int  # Bits remaining after operation
    noise_consumed: int  # Bits consumed by this operation
    description: str = ""


@dataclass
class NoiseBudget:
    """
    Track noise budget through a computation graph.

    Attributes:
        initial_budget: Initial noise budget in bits
        current_budget: Current remaining budget
        operations: List of operations performed
        failed: Whether budget has been exhausted
    """
    initial_budget: int
    current_budget: int
    operations: List[NoiseOperation] = field(default_factory=list)
    failed: bool = False

    @property
    def depth(self) -> int:
        """Current circuit depth."""
        return len(self.operations)

    @property
    def noise_consumed(self) -> int:
        """Total noise consumed so far."""
        return self.initial_budget - self.current_budget


def estimate_noise_budget(
    ciphertext: Union[ts.CKKSVector, ts.BFVVector],
    context: ts.Context,
) -> int:
    """
    Estimate the current noise budget of a ciphertext.

    WARNING: TenSEAL doesn't expose direct noise budget access.
    This is a placeholder for manual tracking.

    In practice, you must:
    1. Track operations manually with NoiseBudget
    2. Use known noise growth formulas
    3. Test empirically by decrypting

    Args:
        ciphertext: The ciphertext to analyze
        context: TenSEAL context

    Returns:
        Estimated noise budget in bits (placeholder - returns initial budget)
    """
    warnings.warn(
        "Direct noise budget measurement not available in TenSEAL Python. "
        "Use NoiseBudget class to track manually."
    )

    # Return initial budget as approximation
    # This is NOT accurate for ciphertexts that have undergone operations
    coeff_mod_bit_sizes = context.params().coeff_mod_bit_sizes()
    return sum(coeff_mod_bit_sizes)


def get_initial_noise_budget(coeff_mod_bit_sizes: List[int]) -> int:
    """
    Calculate the initial noise budget from coefficient modulus bit sizes.

    The initial noise budget is approximately the sum of the bit sizes
    of the coefficient modulus primes (minus some overhead).

    Args:
        coeff_mod_bit_sizes: List of coefficient modulus bit sizes

    Returns:
        Initial noise budget in bits

    Example:
        >>> budget = get_initial_noise_budget([60, 40, 40, 60])
        >>> print(f"Initial noise budget: {budget} bits")
    """
    return sum(coeff_mod_bit_sizes)


def estimate_addition_noise(
    ciphertext_a: Union[ts.CKKSVector, ts.BFVVector],
    ciphertext_b: Union[ts.CKKSVector, ts.BFVVector, np.ndarray],
) -> int:
    """
    Estimate noise consumption for homomorphic addition.

    Addition consumes minimal noise (typically 1-2 bits).

    Args:
        ciphertext_a: First ciphertext
        ciphertext_b: Second ciphertext or plaintext

    Returns:
        Estimated bits consumed

    Reference:
        BFV: ~1 bit per addition
        CKKS: ~1 bit per addition (before rescaling)
    """
    # Addition is very cheap
    return 2  # Conservative estimate


def estimate_multiplication_noise(
    scale: float = 2**40,
    is_plain: bool = False,
) -> int:
    """
    Estimate noise consumption for homomorphic multiplication.

    Multiplication is EXPENSIVE:
    - Ciphertext-Ciphertext: ~2 * log2(scale) bits
    - Ciphertext-Plaintext: ~log2(scale) bits

    Args:
        scale: CKKS scale parameter
        is_plain: True if second operand is plaintext

    Returns:
        Estimated bits consumed

    Example:
        >>> # CKKS with scale=2^40
        >>> noise = estimate_multiplication_noise(scale=2**40, is_plain=False)
        >>> print(f"Ciphertext-ciphertext mult consumes ~{noise} bits")
    """
    scale_bits = int(np.log2(scale))

    if is_plain:
        # Ciphertext-plaintext multiplication
        return scale_bits + 2
    else:
        # Ciphertext-ciphertext multiplication (MUCH more expensive)
        return 2 * scale_bits + 5


def estimate_rotation_noise(
    scale: float = 2**40,
) -> int:
    """
    Estimate noise consumption for rotation operations.

    Rotation (using Galois keys) consumes moderate noise.
    This is relevant for matrix operations and batch processing.

    Args:
        scale: CKKS scale parameter

    Returns:
        Estimated bits consumed

    Reference:
        ~1 bit per rotation (much cheaper than multiplication)
    """
    return 1


def simulate_circuit_depth(
    operations: List[Tuple[str, bool]],
    scale: float = 2**40,
    initial_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate noise consumption through a sequence of operations.

    Useful for planning circuits before implementation.

    Args:
        operations: List of (operation_type, is_plain) tuples
            operation_type: 'add' or 'mult'
            is_plain: True if operand is plaintext
        scale: CKKS scale parameter
        initial_budget: Starting budget (required)

    Returns:
        Dictionary with simulation results:
        {
            'initial_budget': int,
            'final_budget': int,
            'total_operations': int,
            'budget_exhausted': bool,
            'noise_trace': List[int],  # Budget after each operation
            'max_safe_depth': int  # Depth where budget would hit 0
        }

    Example:
        >>> # Simulate: 3 multiplications followed by 2 additions
        >>> ops = [('mult', True), ('mult', True), ('mult', True),
        ...        ('add', False), ('add', False)]
        >>> result = simulate_circuit_depth(ops, scale=2**40, initial_budget=200)
        >>> print(f"Final budget: {result['final_budget']} bits")
    """
    if initial_budget is None:
        raise ValueError("initial_budget must be provided")

    budget = initial_budget
    noise_trace = [budget]

    for op_type, is_plain in operations:
        if op_type == 'add':
            consumed = estimate_addition_noise(None, None)
        elif op_type == 'mult':
            consumed = estimate_multiplication_noise(scale, is_plain)
        elif op_type == 'rotate':
            consumed = estimate_rotation_noise(scale)
        else:
            warnings.warn(f"Unknown operation type: {op_type}")
            continue

        budget -= consumed
        noise_trace.append(max(0, budget))

        if budget <= 0:
            break

    return {
        'initial_budget': initial_budget,
        'final_budget': max(0, budget),
        'total_operations': len(operations),
        'budget_exhausted': budget <= 0,
        'noise_trace': noise_trace,
        'max_safe_depth': len(noise_trace) - 1 if budget > 0 else len(noise_trace) - 2,
    }


def NoiseTracker(initial_budget: int):
    """
    Factory function to create a noise budget tracker.

    This is a convenience wrapper around the NoiseBudget class.

    Args:
        initial_budget: Starting noise budget in bits

    Returns:
        NoiseBudget instance

    Example:
        >>> ctx = create_ckks_context()
        >>> tracker = NoiseTracker(get_initial_noise_budget(ctx))
        >>> # Now track operations...
    """
    return NoiseBudget(
        initial_budget=initial_budget,
        current_budget=initial_budget,
    )


def track_operation(
    budget: NoiseBudget,
    op_type: OperationType,
    noise_consumed: int,
    description: str = "",
) -> NoiseBudget:
    """
    Record an operation and update the noise budget.

    Args:
        budget: Current NoiseBudget object
        op_type: Type of operation
        noise_consumed: Bits consumed by this operation
        description: Human-readable description

    Returns:
        Updated NoiseBudget object

    Example:
        >>> tracker = NoiseTracker(initial_budget=100)
        >>> tracker = track_operation(
        ...     tracker,
        ...     OperationType.MULT,
        ...     noise_consumed=45,
        ...     description="First layer multiplication"
        ... )
    """
    operation = NoiseOperation(
        op_type=op_type,
        depth=budget.depth,
        noise_before=budget.current_budget,
        noise_after=budget.current_budget - noise_consumed,
        noise_consumed=noise_consumed,
        description=description,
    )

    budget.operations.append(operation)
    budget.current_budget -= noise_consumed

    if budget.current_budget <= 0:
        budget.failed = True
        warnings.warn(
            f"NOISE BUDGET EXHAUSTED at depth {budget.depth}! "
            f"Operation: {description}"
        )

    return budget


def print_noise_report(budget: NoiseBudget, title: str = "Noise Budget Report") -> None:
    """
    Print a detailed noise budget report.

    Args:
        budget: NoiseBudget object to report on
        title: Title for the report
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Initial budget:   {budget.initial_budget} bits")
    print(f"Current budget:   {budget.current_budget} bits")
    print(f"Consumed:         {budget.noise_consumed} bits")
    print(f"Remaining:        {budget.current_budget / budget.initial_budget * 100:.1f}%")
    print(f"Circuit depth:    {budget.depth} operations")
    print(f"Status:           {'EXHAUSTED' if budget.failed else 'OK'}")

    if budget.operations:
        print(f"\nOperation Details:")
        print(f"{'-'*70}")
        print(f"{'Depth':<6} {'Operation':<20} {'Before':<10} {'After':<10} {'Consumed':<10}")
        print(f"{'-'*70}")

        for op in budget.operations:
            op_name = op.op_type.value
            print(f"{op.depth:<6} {op_name:<20} {op.noise_before:<10} "
                  f"{op.noise_after:<10} {op.noise_consumed:<10}")

            if op.description:
                print(f"       └─ {op.description}")

    print(f"{'='*70}\n")


def get_recommended_parameters(
    n_multiplications: int,
    n_additions: int = 0,
    scheme: str = 'ckks',
    scale: float = 2**40,
    safety_margin: float = 1.5,
) -> Dict[str, Any]:
    """
    Recommend security parameters for a given circuit depth.

    Args:
        n_multiplications: Number of ciphertext-ciphertext multiplications
        n_additions: Number of additions (less critical)
        scheme: 'bfv' or 'ckks'
        scale: CKKS scale parameter
        safety_margin: Multiplier for extra safety (default 1.5x)

    Returns:
        Dictionary with recommended parameters:
        {
            'poly_modulus_degree': int,
            'coeff_mod_bit_sizes': List[int],
            'estimated_budget': int,
            'estimated_consumption': int,
            'safe': bool
        }

    Example:
        >>> # Need 3 multiplications
        >>> params = get_recommended_parameters(n_multiplications=3)
        >>> ctx = create_ckks_context(**params)
    """
    # Estimate noise consumption
    scale_bits = int(np.log2(scale))
    mult_cost = 2 * scale_bits + 5
    add_cost = 2

    estimated_consumption = (n_multiplications * mult_cost +
                            n_additions * add_cost)

    # Required budget with safety margin
    required_budget = int(estimated_consumption * safety_margin)

    # Recommend coeff_mod_bit_sizes based on required budget
    if required_budget <= 120:
        poly_modulus_degree = 8192
        # Special primes + matching primes
        n_primes = 4
        special_size = 60
        mid_size = scale_bits

        coeff_mod = [special_size]
        for _ in range(n_primes - 2):
            coeff_mod.append(mid_size)
        coeff_mod.append(special_size)

    elif required_budget <= 200:
        poly_modulus_degree = 16384
        n_primes = 5
        special_size = 60
        mid_size = scale_bits

        coeff_mod = [special_size]
        for _ in range(n_primes - 2):
            coeff_mod.append(mid_size)
        coeff_mod.append(special_size)

    else:
        raise ValueError(
            f"Required budget ({required_budget} bits) too high. "
            "Consider reducing circuit depth or using hybrid HE/TEE approach."
        )

    estimated_budget = sum(coeff_mod)

    return {
        'poly_modulus_degree': poly_modulus_degree,
        'coeff_mod_bit_sizes': coeff_mod,
        'scale': scale if scheme == 'ckks' else None,
        'estimated_budget': estimated_budget,
        'estimated_consumption': estimated_consumption,
        'safe': estimated_budget > required_budget,
    }


def max_multiplications_for_context(
    coeff_mod_bit_sizes: List[int],
    scale: float = 2**40,
    safety_margin: float = 1.5,
) -> int:
    """
    Calculate the maximum safe number of ciphertext-ciphertext multiplications.

    Args:
        coeff_mod_bit_sizes: Coefficient modulus bit sizes
        scale: CKKS scale parameter
        safety_margin: Safety margin (default 1.5x)

    Returns:
        Maximum number of multiplications

    Example:
        >>> max_mult = max_multiplications_for_context([60, 40, 40, 60])
        >>> print(f"Can safely perform {max_mult} ciphertext multiplications")
    """
    budget = get_initial_noise_budget(coeff_mod_bit_sizes)
    scale_bits = int(np.log2(scale))
    mult_cost = 2 * scale_bits + 5

    # Conservative estimate
    safe_consumption = budget / safety_margin
    max_mult = int(safe_consumption / mult_cost)

    return max(0, max_mult - 1)  # -1 for safety
