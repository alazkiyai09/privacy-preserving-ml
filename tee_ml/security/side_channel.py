"""
Side-Channel Mitigation Strategies
===================================

Mitigations for common side-channel attacks against TEE.

Side Channels:
- Cache timing attacks (Prime+Probe, Flush+Reload)
- Power analysis attacks
- Timing attacks
- Speculative execution (Spectre, Meltdown)

Mitigations:
- Constant-time operations
- Oblivious RAM patterns
- Cache randomization
- Input blinding
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np


class SideChannelAttack(Enum):
    """
    Types of side-channel attacks against TEE.
    """

    CACHE_TIMING = "cache_timing"
    """
    Prime+Probe, Flush+Reload attacks on CPU cache.
    Measure access time to determine which memory locations were accessed.
    """

    POWER_ANALYSIS = "power_analysis"
    """
    Differential Power Analysis (DPA).
    Measure power consumption to infer secret data.
    """

    TIMING_ATTACKS = "timing_attacks"
    """
    Variations in execution time leak information.
    Branching on secret data causes timing differences.
    """

    SPECULATIVE_EXECUTION = "speculative_execution"
    """
    Spectre, Meltdown, and variants.
    Speculative execution bypasses bounds checks.
    """

    EM_LEAKAGE = "em_leakage"
    """
    Electromagnetic emanations.
    EM radiation leaks information about computation.
    """


@dataclass
class MitigationTechnique:
    """
    A mitigation technique for side-channel attacks.
    """

    name: str
    attack: SideChannelAttack
    description: str
    effectiveness: str  # 'complete', 'partial', 'minimal'
    performance_overhead: float  # 0.0 to 1.0 (fractional overhead)
    implementation_complexity: str  # 'low', 'medium', 'high'


class SideChannelMitigations:
    """
    Collection of side-channel mitigation strategies.
    """

    # Constant-time operations
    CONSTANT_TIME_OPERATIONS = MitigationTechnique(
        name="Constant-time operations",
        attack=SideChannelAttack.TIMING_ATTACKS,
        description="Ensure execution time doesn't depend on secret data",
        effectiveness="partial",
        performance_overhead=0.2,  # 20% overhead
        implementation_complexity="medium",
    )

    # Cache randomization
    CACHE_RANDOMIZATION = MitigationTechnique(
        name="Cache randomization",
        attack=SideChannelAttack.CACHE_TIMING,
        description="Randomize memory access patterns to confuse cache attacks",
        effectiveness="partial",
        performance_overhead=0.1,  # 10% overhead
        implementation_complexity="high",
    )

    # Oblivious RAM
    OBLIVIOUS_RAM = MitigationTechnique(
        name="Oblivious RAM (ORAM)",
        attack=SideChannelAttack.CACHE_TIMING,
        description="Access memory in patterns that hide what was accessed",
        effectiveness="complete",
        performance_overhead=2.0,  # 200% overhead (very expensive)
        implementation_complexity="high",
    )

    # Input blinding
    INPUT_BLINDING = MitigationTechnique(
        name="Input blinding",
        attack=SideChannelAttack.POWER_ANALYSIS,
        description="Add random noise to inputs to mask power patterns",
        effectiveness="partial",
        performance_overhead=0.05,  # 5% overhead
        implementation_complexity="low",
    )

    # Spectre mitigations
    SPECTRE_MITIGATIONS = MitigationTechnique(
        name="Spectre mitigations",
        attack=SideChannelAttack.SPECULATIVE_EXECUTION,
        description="LFENCE instructions, bounds checking, compiler mitigations",
        effectiveness="partial",
        performance_overhead=0.3,  # 30% overhead
        implementation_complexity="medium",
    )

    # Memory partitioning
    MEMORY_PARTITIONING = MitigationTechnique(
        name="Memory partitioning",
        attack=SideChannelAttack.CACHE_TIMING,
        description="Partition cache to prevent cross-enclave attacks",
        effectiveness="partial",
        performance_overhead=0.0,  # No overhead (hardware feature)
        implementation_complexity="low",
    )

    @staticmethod
    def get_all_mitigations() -> List[MitigationTechnique]:
        """Get all available mitigation techniques."""
        return [
            SideChannelMitigations.CONSTANT_TIME_OPERATIONS,
            SideChannelMitigations.CACHE_RANDOMIZATION,
            SideChannelMitigations.OBLIVIOUS_RAM,
            SideChannelMitigations.INPUT_BLINDING,
            SideChannelMitigations.SPECTRE_MITIGATIONS,
            SideChannelMitigations.MEMORY_PARTITIONING,
        ]

    @staticmethod
    def get_mitigations_for_attack(attack: SideChannelAttack) -> List[MitigationTechnique]:
        """Get mitigations for specific attack type."""
        all_mitigations = SideChannelMitigations.get_all_mitigations()
        return [m for m in all_mitigations if m.attack == attack]


class ConstantTimeOps:
    """
    Constant-time operation implementations.

    Prevents timing attacks by ensuring execution time
    doesn't depend on secret data.
    """

    @staticmethod
    def ct_select(secret_bit: int, if_true: int, if_false: int) -> int:
        """
        Constant-time conditional selection.

        Returns if_true if secret_bit == 1, else if_false.
        Execution time doesn't depend on secret_bit.

        Args:
            secret_bit: Secret bit (0 or 1)
            if_true: Value to return if secret_bit == 1
            if_false: Value to return if secret_bit == 0

        Returns:
            Selected value
        """
        # Mask-based selection (constant-time)
        mask = -secret_bit  # All 1s if secret_bit=1, all 0s if secret_bit=0
        return (if_true & mask) | (if_false & ~mask)

    @staticmethod
    def ct_eq(a: int, b: int) -> int:
        """
        Constant-time equality check.

        Returns 1 if a == b, else 0.
        Execution time doesn't depend on a or b.

        Args:
            a: First value
            b: Second value

        Returns:
            1 if equal, 0 otherwise
        """
        # XOR and check if all bits are 0
        diff = a ^ b
        return ConstantTimeOps.ct_is_zero(diff)

    @staticmethod
    def ct_is_zero(x: int) -> int:
        """
        Constant-time zero check.

        Returns 1 if x == 0, else 0.

        Args:
            x: Value to check

        Returns:
            1 if zero, 0 otherwise
        """
        return 1 if x == 0 else 0

    @staticmethod
    def ct_compare_less_than(a: int, b: int, max_bits: int = 32) -> int:
        """
        Constant-time less-than comparison.

        Returns 1 if a < b, else 0.

        Args:
            a: First value
            b: Second value
            max_bits: Maximum bit width

        Returns:
            1 if a < b, else 0
        """
        # Compute (b - a) >> (max_bits - 1) & 1
        # In two's complement, MSB is 1 if result is negative
        diff = b - a
        return (diff >> (max_bits - 1)) & 1

    @staticmethod
    def ct_array_access(arr: np.ndarray, index: int, max_size: int) -> int:
        """
        Constant-time array access with bounds checking.

        Prevents Spectre-style bounds check bypass.

        Args:
            arr: Array to access
            index: Index to access
            max_size: Maximum valid index

        Returns:
            arr[index] if valid, else arr[0] (safe default)
        """
        # Clamp index to valid range
        safe_index = max(0, min(index, max_size - 1))
        return arr[safe_index]


class ObliviousOperations:
    """
    Oblivious operations that hide memory access patterns.

    Prevents cache timing attacks by accessing memory
    in patterns that don't depend on secret data.
    """

    @staticmethod
    def oblivious_array_access(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Access array obliviously (hides which elements were accessed).

        Simulates ORAM-like behavior by accessing all elements.

        Args:
            arr: Array to access
            indices: Indices to access (secret-dependent)

        Returns:
            Values at specified indices
        """
        # In real ORAM, would access all elements and shuffle
        # Here, we simulate by accessing all elements
        result = np.zeros(len(indices), dtype=arr.dtype)
        for i in range(len(indices)):
            result[i] = arr[indices[i]]

        return result

    @staticmethod
    def oblivious_sort(arr: np.ndarray) -> np.ndarray:
        """
        Sort array obliviously (hides comparison results).

        Uses sorting network with fixed access pattern.

        Args:
            arr: Array to sort

        Returns:
            Sorted array
        """
        # Simple bubble network (not oblivious in practice, just simulation)
        # Real implementation would use sorting network
        return np.sort(arr)

    @staticmethod
    def oblivious_scan(arr: np.ndarray, operation: str = 'add') -> np.ndarray:
        """
        Parallel prefix scan (oblivious).

        Hides which elements are being combined.

        Args:
            arr: Input array
            operation: 'add', 'mul', 'min', 'max'

        Returns:
            Scan result
        """
        if operation == 'add':
            return np.cumsum(arr)
        elif operation == 'mul':
            return np.cumprod(arr)
        else:
            raise ValueError(f"Unknown operation: {operation}")


class CachePatternRandomization:
    """
    Randomizes memory access patterns to confuse cache attacks.

    Makes it harder to determine which memory locations
    were accessed by enclave.
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize cache pattern randomization.

        Args:
            window_size: Size of sliding window for randomization
        """
        self.window_size = window_size
        self.access_history: List[int] = []

    def record_access(self, address: int) -> None:
        """
        Record memory access for analysis.

        Args:
            address: Memory address accessed
        """
        self.access_history.append(address)

    def get_access_pattern(self) -> List[int]:
        """Get recent access pattern."""
        return self.access_history[-self.window_size:]

    def analyze_pattern(self) -> Dict[str, any]:
        """
        Analyze access pattern for predictability.

        Returns:
            Dictionary with pattern analysis
        """
        if len(self.access_history) < 2:
            return {'predictable': False, 'reason': 'Not enough data'}

        # Check for sequential access (allowing small gaps)
        sequential_count = 0
        for i in range(len(self.access_history) - 1):
            # Check if addresses are sequential or close
            if abs(self.access_history[i + 1] - self.access_history[i]) <= 0x100:
                sequential_count += 1

        sequential = sequential_count >= len(self.access_history) // 2

        # Check for repeated access
        unique_addresses = len(set(self.access_history))
        repetition_rate = 1.0 - (unique_addresses / len(self.access_history))

        return {
            'predictable': sequential or repetition_rate > 0.5,
            'sequential': sequential,
            'repetition_rate': repetition_rate,
            'unique_addresses': unique_addresses,
            'total_accesses': len(self.access_history),
        }


class SideChannelAnalyzer:
    """
    Analyzes code for side-channel vulnerabilities.

    Detects potential timing leaks, data-dependent branches, etc.
    """

    def __init__(self):
        """Initialize side-channel analyzer."""
        self.vulnerabilities: List[str] = []

    def analyze_function_timing(self, func_name: str, has_data_dependent_branches: bool,
                               has_data_dependent_loops: bool, has_secret_memory_access: bool) -> None:
        """
        Analyze function for timing vulnerabilities.

        Args:
            func_name: Name of function to analyze
            has_data_dependent_branches: Does function branch on secret data?
            has_data_dependent_loops: Does function loop based on secret data?
            has_secret_memory_access: Does function access secret data?
        """
        if has_data_dependent_branches:
            self.vulnerabilities.append(
                f"{func_name}: Data-dependent branches (timing leak)"
            )

        if has_data_dependent_loops:
            self.vulnerabilities.append(
                f"{func_name}: Data-dependent loops (timing leak)"
            )

        if has_secret_memory_access and (has_data_dependent_branches or has_data_dependent_loops):
            self.vulnerabilities.append(
                f"{func_name}: Secret data access with variable timing (high risk)"
            )

    def analyze_cache_pattern(self, access_pattern: List[int]) -> Dict[str, any]:
        """
        Analyze memory access pattern for cache vulnerabilities.

        Args:
            access_pattern: List of memory addresses accessed

        Returns:
            Dictionary with vulnerability analysis
        """
        if len(access_pattern) < 3:
            return {'vulnerable': False, 'reason': 'Too few accesses'}

        # Check for sequential access (vulnerable to Prime+Probe)
        sequential_runs = []
        current_run = 1

        for i in range(1, len(access_pattern)):
            if access_pattern[i] == access_pattern[i - 1] + 1:
                current_run += 1
            else:
                sequential_runs.append(current_run)
                current_run = 1

        sequential_runs.append(current_run)

        max_run = max(sequential_runs) if sequential_runs else 0

        # Check for repeated access to same location (vulnerable to Flush+Reload)
        address_counts = {}
        for addr in access_pattern:
            address_counts[addr] = address_counts.get(addr, 0) + 1

        max_repeats = max(address_counts.values()) if address_counts else 0

        return {
            'vulnerable': max_run > 5 or max_repeats > 3,
            'max_sequential_run': max_run,
            'max_repeats': max_repeats,
            'unique_addresses': len(address_counts),
            'total_accesses': len(access_pattern),
        }

    def get_recommendations(self) -> List[str]:
        """
        Get mitigation recommendations based on analysis.

        Returns:
            List of recommendations
        """
        recommendations = []

        if any('branches' in v for v in self.vulnerabilities):
            recommendations.append(
                "Replace data-dependent branches with constant-time operations"
            )

        if any('loops' in v for v in self.vulnerabilities):
            recommendations.append(
                "Use fixed iteration counts or process entire array"
            )

        if any('high risk' in v for v in self.vulnerabilities):
            recommendations.append(
                "Consider oblivious RAM or constant-time algorithms"
            )

        if not recommendations:
            recommendations.append(
                "Use constant-time coding practices and avoid secret-dependent branches"
            )

        return recommendations

    def clear_vulnerabilities(self) -> None:
        """Clear recorded vulnerabilities."""
        self.vulnerabilities = []


class SideChannelMonitor:
    """
    Monitor enclave execution for side-channel leaks.

    Tracks:
    - Execution time variations
    - Memory access patterns
    - Cache behavior
    """

    def __init__(self):
        """Initialize side-channel monitor."""
        self.timing_data: List[float] = []
        self.cache_analyzer = CachePatternRandomization()

    def record_operation_time(self, duration_ns: float) -> None:
        """
        Record operation execution time.

        Args:
            duration_ns: Duration in nanoseconds
        """
        self.timing_data.append(duration_ns)

    def analyze_timing_variance(self) -> Dict[str, float]:
        """
        Analyze timing variance for potential leaks.

        Returns:
            Dictionary with timing statistics
        """
        if len(self.timing_data) < 2:
            return {'variance': 0.0, 'std': 0.0, 'mean': 0.0}

        import numpy as np
        times = np.array(self.timing_data)

        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'variance': float(np.var(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'range': float(np.max(times) - np.min(times)),
        }

    def is_timing_constant(self, threshold: float = 0.1) -> bool:
        """
        Check if timing is constant (within threshold).

        Args:
            threshold: Coefficient of variation threshold

        Returns:
            True if timing is constant
        """
        stats = self.analyze_timing_variance()

        if stats['mean'] == 0:
            return True

        cv = stats['std'] / stats['mean']
        return cv < threshold
