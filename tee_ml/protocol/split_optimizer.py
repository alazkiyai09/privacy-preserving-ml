"""
Optimal HE/TEE Split Point Analyzer
====================================

Analyzes neural network architecture to find optimal split point
between HE and TEE layers in HT2ML hybrid system.

Key Considerations:
1. Noise budget constraints (HE limited to 1-2 layers)
2. Performance requirements (TEE faster for complex ops)
3. Privacy requirements (input privacy vs. computation privacy)
4. Trust model (who to trust with what data)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SplitStrategy(Enum):
    """Strategy for determining optimal split point."""

    PRIVACY_MAX = "privacy_max"
    """
    Maximize input privacy (more HE layers).
    Use when input data is highly sensitive.
    """

    PERFORMANCE_MAX = "performance_max"
    """
    Maximize performance (more TEE layers).
    Use when computation efficiency is critical.
    """

    BALANCED = "balanced"
    """
    Balance privacy and performance.
    Recommended for most applications.
    """

    TRUST_MINIMIZED = "trust_minimized"
    """
    Minimize trust requirements.
    Use cryptographic guarantees where possible.
    """


@dataclass
class LayerSpecification:
    """
    Specification of a single layer in the network.
    """

    index: int
    input_size: int
    output_size: int
    activation: str  # 'none', 'relu', 'sigmoid', 'tanh', 'softmax'
    use_bias: bool = True

    def get_noise_cost(self, scale_bits: int = 40) -> int:
        """
        Calculate noise cost for HE layer.

        Args:
            scale_bits: CKKS scale parameter

        Returns:
            Noise cost in bits
        """
        # Linear cost
        linear_cost = self.output_size * scale_bits

        # Activation cost
        activation_degrees = {
            'none': 0,
            'relu': 5, 'sigmoid': 5, 'tanh': 5, 'softmax': 5,
            'leaky_relu': 5, 'elu': 5, 'gelu': 7, 'swish': 5,
        }

        activation_cost = activation_degrees.get(self.activation, 0) * scale_bits

        return linear_cost + activation_cost

    def get_tee_overhead_ns(self, data_size_mb: float) -> float:
        """
        Estimate TEE execution time for this layer.

        Args:
            data_size_mb: Input data size in MB

        Returns:
            Estimated time in nanoseconds
        """
        # Base computation time (rough estimate)
        # Linear layer: ~10 μs per 1000 elements
        elements = self.input_size * self.output_size
        base_time_ns = (elements / 1000.0) * 10000  # 10 μs per 1000 elements

        # TEE overhead (entry + exit)
        overhead_ns = 25000  # 25 μs fixed overhead

        return base_time_ns + overhead_ns


@dataclass
class SplitRecommendation:
    """
    Recommendation for optimal HE/TEE split.
    """

    strategy: SplitStrategy
    he_layers: int  # Number of HE layers (from input)
    tee_layers: int  # Number of TEE layers (after split)
    split_point: int  # Layer index where split occurs
    noise_budget_used: int  # Bits of noise budget used
    noise_budget_remaining: int  # Bits remaining
    estimated_he_time_ns: float  # Estimated time for HE layers
    estimated_tee_time_ns: float  # Estimated time for TEE layers
    total_time_ns: float  # Total estimated time
    privacy_score: float  # 0.0 to 1.0 (higher = more private)
    performance_score: float  # 0.0 to 1.0 (higher = faster)
    rationale: str
    total_noise_budget: int = 200  # Total budget for feasibility check

    def is_feasible(self) -> bool:
        """Check if recommendation is feasible."""
        # Feasible if we stayed within the noise budget
        return self.noise_budget_used <= self.total_noise_budget

    def print_summary(self) -> None:
        """Print split recommendation summary."""
        print(f"\n{'='*70}")
        print(f"HT2ML Split Recommendation")
        print(f"{'='*70}")
        print(f"\nStrategy: {self.strategy.value}")
        print(f"Split Point: After layer {self.split_point}")
        print(f"HE Layers: {self.he_layers}")
        print(f"TEE Layers: {self.tee_layers}")
        print(f"\nNoise Budget:")
        print(f"  Used: {self.noise_budget_used} bits")
        print(f"  Remaining: {self.noise_budget_remaining} bits")
        print(f"\nEstimated Performance:")
        print(f"  HE time: {self.estimated_he_time_ns / 1000:.1f} μs")
        print(f"  TEE time: {self.estimated_tee_time_ns / 1000:.1f} μs")
        print(f"  Total: {self.total_time_ns / 1000:.1f} μs")
        print(f"\nScores:")
        print(f"  Privacy: {self.privacy_score:.2f}/1.00")
        print(f"  Performance: {self.performance_score:.2f}/1.00")
        print(f"\nRationale: {self.rationale}")
        print(f"{'='*70}\n")


class SplitOptimizer:
    """
    Optimizes HE/TEE split point for HT2ML architecture.

    Analyzes network architecture and noise budget to recommend
    optimal split between HE and TEE layers.
    """

    def __init__(
        self,
        noise_budget: int = 200,
        scale_bits: int = 40,
        max_he_layers: int = 2,
    ):
        """
        Initialize split optimizer.

        Args:
            noise_budget: Total noise budget in bits
            scale_bits: CKKS scale parameter
            max_he_layers: Maximum number of HE layers
        """
        self.noise_budget = noise_budget
        self.scale_bits = scale_bits
        self.max_he_layers = max_he_layers

    def analyze_layer_cost(
        self,
        layers: List[LayerSpecification],
    ) -> Dict[int, int]:
        """
        Analyze noise cost for each layer.

        Args:
            layers: List of layer specifications

        Returns:
            Dictionary mapping layer index to noise cost
        """
        costs = {}
        cumulative_cost = 0

        for layer in layers:
            layer_cost = layer.get_noise_cost(self.scale_bits)
            cumulative_cost += layer_cost
            costs[layer.index] = cumulative_cost

        return costs

    def find_feasible_splits(
        self,
        layers: List[LayerSpecification],
    ) -> List[int]:
        """
        Find all feasible split points.

        A split is feasible if HE layers fit within noise budget.

        Args:
            layers: List of layer specifications

        Returns:
            List of feasible split points (layer indices)
        """
        feasible = []

        costs = self.analyze_layer_cost(layers)

        # Check each potential split point
        # Split after layer i means layers 0..i are HE
        for i in range(len(layers) + 1):
            if i == 0:
                continue  # Need at least one layer

            # Cost of layers 0..i-1
            if i - 1 >= 0:
                cost_to_split = costs[i - 1]
            else:
                cost_to_split = 0

            if cost_to_split <= self.noise_budget:
                feasible.append(i)

        return feasible

    def estimate_performance(
        self,
        layers: List[LayerSpecification],
        split_point: int,
    ) -> Dict[str, float]:
        """
        Estimate performance for given split.

        Args:
            layers: List of layer specifications
            split_point: Layer to split after (0..len(layers))

        Returns:
            Dictionary with performance estimates
        """
        # Estimate HE time (very rough approximation)
        he_time_ns = 0
        for i in range(split_point):
            layer = layers[i]
            # HE is ~100-1000x slower
            estimated_time = layer.get_tee_overhead_ns(0.001) * 100
            he_time_ns += estimated_time

        # Estimate TEE time
        tee_time_ns = 0
        for i in range(split_point, len(layers)):
            layer = layers[i]
            tee_time_ns += layer.get_tee_overhead_ns(0.001)

        return {
            'he_time_ns': he_time_ns,
            'tee_time_ns': tee_time_ns,
            'total_time_ns': he_time_ns + tee_time_ns,
            'he_time_us': he_time_ns / 1000,
            'tee_time_us': tee_time_ns / 1000,
        }

    def calculate_scores(
        self,
        layers: List[LayerSpecification],
        split_point: int,
    ) -> Tuple[float, float]:
        """
        Calculate privacy and performance scores.

        Args:
            layers: List of layer specifications
            split_point: Layer to split after

        Returns:
            (privacy_score, performance_score) tuple
        """
        # Privacy score: More HE layers = higher privacy
        max_he = min(len(layers), self.max_he_layers)
        privacy_score = split_point / max_he

        # Performance score: More TEE layers = higher performance
        # TEE is ~100x faster than HE
        tee_layers = len(layers) - split_point
        max_tee = len(layers)
        performance_score = tee_layers / max_tee

        return privacy_score, performance_score

    def recommend_split(
        self,
        layers: List[LayerSpecification],
        strategy: SplitStrategy = SplitStrategy.BALANCED,
    ) -> SplitRecommendation:
        """
        Recommend optimal split point for given architecture.

        Args:
            layers: List of layer specifications
            strategy: Optimization strategy

        Returns:
            SplitRecommendation with recommendation
        """
        # Find feasible splits
        feasible_splits = self.find_feasible_splits(layers)

        if not feasible_splits:
            # No feasible splits with HE
            # Use all TEE
            tee_time = self.estimate_performance(layers, len(layers))
            he_time = self.estimate_performance(layers, 0)

            return SplitRecommendation(
                strategy=SplitStrategy.PERFORMANCE_MAX,
                he_layers=0,
                tee_layers=len(layers),
                split_point=0,
                noise_budget_used=0,
                noise_budget_remaining=self.noise_budget,
                estimated_he_time_ns=he_time['he_time_ns'],
                estimated_tee_time_ns=tee_time['tee_time_ns'],
                total_time_ns=tee_time['total_time_ns'],
                privacy_score=0.0,
                performance_score=1.0,
                rationale="No HE layers fit within noise budget. Using all TEE.",
                total_noise_budget=self.noise_budget,
            )

        # Analyze each feasible split
        best_split = None
        best_score = -1

        for split_point in feasible_splits:
            # Calculate scores based on strategy
            privacy_score, performance_score = self.calculate_scores(
                layers, split_point
            )

            # Combine scores based on strategy
            if strategy == SplitStrategy.PRIVACY_MAX:
                score = privacy_score
            elif strategy == SplitStrategy.PERFORMANCE_MAX:
                score = performance_score
            else:  # BALANCED
                score = (privacy_score + performance_score) / 2

            if score > best_score:
                best_score = score
                best_split = split_point

        # Get performance estimate
        perf = self.estimate_performance(layers, best_split)

        # Calculate noise budget usage
        if best_split > 0:
            noise_used = self.analyze_layer_cost(layers)[best_split - 1]
        else:
            noise_used = 0

        # Create rationale
        he_layers = best_split
        tee_layers = len(layers) - best_split

        if strategy == SplitStrategy.PRIVACY_MAX:
            rationale = (
                f"Maximize input privacy with {he_layers} HE layers. "
                f"Uses {noise_used}/{self.noise_budget} bits noise budget."
            )
        elif strategy == SplitStrategy.PERFORMANCE_MAX:
            rationale = (
                f"Maximize performance with {tee_layers} TEE layers. "
                f"Only {he_layers} HE layers for input privacy."
            )
        else:  # BALANCED
            rationale = (
                f"Balanced approach with {he_layers} HE layers and "
                f"{tee_layers} TEE layers. "
                f"Optimal trade-off between privacy ({privacy_score:.2f}) "
                f"and performance ({performance_score:.2f})."
            )

        return SplitRecommendation(
            strategy=strategy,
            he_layers=he_layers,
            tee_layers=tee_layers,
            split_point=best_split,
            noise_budget_used=noise_used,
            noise_budget_remaining=self.noise_budget - noise_used,
            estimated_he_time_ns=perf['he_time_ns'],
            estimated_tee_time_ns=perf['tee_time_ns'],
            total_time_ns=perf['total_time_ns'],
            privacy_score=privacy_score,
            performance_score=performance_score,
            rationale=rationale,
            total_noise_budget=self.noise_budget,
        )

    def compare_all_strategies(
        self,
        layers: List[LayerSpecification],
    ) -> Dict[str, SplitRecommendation]:
        """
        Compare all split strategies.

        Args:
            layers: List of layer specifications

        Returns:
            Dictionary with recommendations for each strategy
        """
        recommendations = {}

        for strategy in [
            SplitStrategy.PRIVACY_MAX,
            SplitStrategy.PERFORMANCE_MAX,
            SplitStrategy.BALANCED,
        ]:
            recommendations[strategy.value] = self.recommend_split(
                layers, strategy
            )

        return recommendations


def create_layer_specifications(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activations: List[str],
    use_biases: List[bool] = None,
) -> List[LayerSpecification]:
    """
    Create layer specifications from architecture description.

    Args:
        input_size: Input dimensionality
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimensionality
        activations: Activation for each layer
        use_biases: Bias usage for each layer

    Returns:
        List of LayerSpecification
    """
    if use_biases is None:
        use_biases = [True] * (len(hidden_sizes) + 1)

    if len(activations) != len(hidden_sizes) + 1:
        raise ValueError(
            f"Number of activations ({len(activations)}) must match "
            f"number of layers ({len(hidden_sizes) + 1})"
        )

    layers = []
    all_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(all_sizes) - 1):
        layer = LayerSpecification(
            index=i,
            input_size=all_sizes[i],
            output_size=all_sizes[i + 1],
            activation=activations[i],
            use_bias=use_biases[i],
        )
        layers.append(layer)

    return layers


def estimate_optimal_split(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activations: List[str],
    noise_budget: int = 200,
    strategy: SplitStrategy = SplitStrategy.BALANCED,
) -> SplitRecommendation:
    """
    Estimate optimal split point for network architecture.

    This is a convenience function that creates layer specifications
    and calls the optimizer.

    Args:
        input_size: Input dimensionality
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimensionality
        activations: Activation for each layer
        noise_budget: Noise budget in bits
        strategy: Optimization strategy

    Returns:
        SplitRecommendation with optimal split
    """
    # Create layer specifications
    layers = create_layer_specifications(
        input_size, hidden_sizes, output_size, activations
    )

    # Create optimizer
    optimizer = SplitOptimizer(
        noise_budget=noise_budget,
        max_he_layers=2,
    )

    # Get recommendation
    return optimizer.recommend_split(layers, strategy)


def visualize_split(
    recommendation: SplitRecommendation,
    layers: List[LayerSpecification],
) -> str:
    """
    Visualize split recommendation as ASCII art.

    Args:
        recommendation: Split recommendation
        layers: Layer specifications

    Returns:
        ASCII visualization
    """
    lines = []
    lines.append("Network Architecture:")
    lines.append("")

    for i, layer in enumerate(layers):
        # Determine environment
        if i < recommendation.he_layers:
            env = "HE"
        else:
            env = "TEE"

        # Layer info
        lines.append(f"Layer {i}: {layer.input_size} → {layer.output_size}")
        lines.append(f"  Environment: {env}")
        lines.append(f"  Activation: {layer.activation}")
        lines.append("")

    # Add split marker
    lines.append(f"{'='*60}")
    lines.append(f"SPLIT POINT: After layer {recommendation.split_point}")
    lines.append(f"{'='*60}")
    lines.append("")

    # Add summary
    lines.append(f"He Layers: {recommendation.he_layers}")
    lines.append(f"TEE Layers: {recommendation.tee_layers}")
    lines.append(f"Privacy Score: {recommendation.privacy_score:.2f}/1.00")
    lines.append(f"Performance Score: {recommendation.performance_score:.2f}/1.00")
    lines.append("")

    return "\n".join(lines)


def analyze_tradeoffs(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activations: List[str],
    noise_budget: int = 200,
) -> Dict[str, Any]:
    """
    Analyze trade-offs between different split strategies.

    Args:
        input_size: Input dimensionality
        hidden_sizes: Hidden layer sizes
        output_size: Output dimensionality
        activations: Activation for each layer
        noise_budget: Noise budget in bits

    Returns:
        Dictionary with trade-off analysis
    """
    # Create layers
    layers = create_layer_specifications(
        input_size, hidden_sizes, output_size, activations
    )

    # Create optimizer
    optimizer = SplitOptimizer(noise_budget=noise_budget)

    # Get recommendations for all strategies
    recommendations = optimizer.compare_all_strategies(layers)

    # Extract key metrics
    analysis = {
        'num_layers': len(layers),
        'noise_budget': noise_budget,
        'feasible_splits': len(optimizer.find_feasible_splits(layers)),
        'recommendations': {},
    }

    for strategy_name, rec in recommendations.items():
        analysis['recommendations'][strategy_name] = {
            'he_layers': rec.he_layers,
            'tee_layers': rec.tee_layers,
            'privacy_score': rec.privacy_score,
            'performance_score': rec.performance_score,
            'estimated_time_ms': rec.total_time_ns / 1e6,
            'feasible': rec.is_feasible(),
        }

    return analysis
