"""
HT2ML: Hybrid HE/TEE Architecture for Privacy-Preserving ML
============================================================

This module documents the design of a hybrid homomorphic encryption
and trusted execution environment architecture for practical,
privacy-preserving machine learning inference.

Motivation:
- Pure HE: Limited by noise budget (1-2 layers max)
- Pure TEE: Requires trust in hardware/manufacturer
- Hybrid HT2ML: Best of both worlds

Architecture Overview:
                    ┌──────────────────────────────────────┐
                    │   Client (Privacy-Sensitive Data)    │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │  Homomorphic Encryption Layer       │
                    │  (Client-Side or Secure Proxy)       │
                    │                                      │
                    │  1. Encrypt input data              │
                    │  2. Send E(x) to server             │
                    │  3. Privacy: Cryptographic          │
                    └────────────┬─────────────────────────┘
                                 │ E(x)
                    ┌────────────▼─────────────────────────┐
                    │   HE Inference Layer (1-2 layers)    │
                    │  (Server-Side)                      │
                    │                                      │
                    │  1. Compute E(y1) = Layer1(E(x))     │
                    │  2. Compute E(y2) = Layer2(E(y1))   │
                    │  3. Privacy: Fully encrypted         │
                    │  4. Limit: Noise budget              │
                    └────────────┬─────────────────────────┘
                                 │ E(y2)
                    ┌────────────▼─────────────────────────┐
                    │   TEE Decryption & Inference         │
                    │  (Server-Side, SGX Enclave)         │
                    │                                      │
                    │  1. Decrypt E(y2) → y2               │
                    │  2. Compute y3 = Layer3(y2)          │
                    │  3. Compute y4 = Layer4(y3)          │
                    │  ...                                 │
                    │  N. Compute predictions              │
                    │  Privacy: Hardware-enforced          │
                    └────────────┬─────────────────────────┘
                                 │ predictions
                    ┌────────────▼─────────────────────────┐
                    │   Response to Client                │
                    │                                      │
                    │  - Return predictions               │
                    │  - Or store on server                │
                    └──────────────────────────────────────┘

Key Properties:
1. **Input Privacy**: Cryptographically protected (HE)
2. **Model Privacy**: Partially protected (TEE attestation)
3. **Performance**: HE (1-2 layers) + TEE (unlimited depth)
4. **Trust Model**: Trust cryptography OR trust hardware

Why This Works:
- HE processes sensitive input without decryption
- HE layers limited by noise, but only 1-2 needed for privacy
- TEE handles complex inference efficiently
- Combined: Privacy + Performance
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TrustModel(Enum):
    """Trust assumptions for different deployment models."""

    CRYPTOGRAPHIC = "cryptographic"  # Trust math only (pure HE)
    HARDWARE = "hardware"  # Trust hardware (pure TEE)
    HYBRID = "hybrid"  # Trust math OR hardware (HT2ML)


@dataclass
class HT2MLLayer:
    """Definition of a layer in HT2ML architecture."""

    index: int
    input_size: int
    output_size: int
    execution_env: str  # 'HE' or 'TEE'

    # Layer parameters
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    activation: str = 'none'

    def get_noise_cost(self, scale_bits: int = 40) -> int:
        """Get noise cost for HE layers."""
        if self.execution_env == 'HE':
            # Linear cost
            linear_cost = self.output_size * scale_bits

            # Activation cost
            activation_degrees = {
                'relu': 5, 'sigmoid': 5, 'tanh': 5, 'softplus': 5, 'none': 0
            }
            activation_cost = activation_degrees.get(self.activation, 0) * scale_bits

            return linear_cost + activation_cost
        else:
            return 0  # TEE has no noise cost


@dataclass
class HT2MLArchitecture:
    """
    HT2ML hybrid architecture definition.

    Rules:
    1. First 1-2 layers must be HE (for input privacy)
    2. Remaining layers can be TEE (for performance)
    3. Total HE cost must fit within noise budget
    """

    layers: List[HT2MLLayer]
    trust_model: TrustModel = TrustModel.HYBRID
    noise_budget: int = 200

    def __post_init__(self):
        """Validate architecture."""
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")

        # Check that HE layers come first
        he_layer_seen = False
        tee_layer_seen = False

        for layer in self.layers:
            if layer.execution_env == 'TEE':
                tee_layer_seen = True
            elif tee_layer_seen and layer.execution_env == 'HE':
                raise ValueError("HE layers must come before TEE layers")

    def get_total_he_cost(self, scale_bits: int = 40) -> int:
        """Calculate total noise cost of HE layers."""
        return sum(layer.get_noise_cost(scale_bits) for layer in self.layers
                   if layer.execution_env == 'HE')

    def is_feasible(self, scale_bits: int = 40) -> bool:
        """Check if architecture fits within noise budget."""
        return self.get_total_he_cost(scale_bits) <= self.noise_budget

    def get_num_he_layers(self) -> int:
        """Get number of HE layers."""
        return sum(1 for layer in self.layers if layer.execution_env == 'HE')

    def get_num_tee_layers(self) -> int:
        """Get number of TEE layers."""
        return sum(1 for layer in self.layers if layer.execution_env == 'TEE')

    def print_summary(self) -> None:
        """Print architecture summary."""
        print(f"\n{'='*60}")
        print(f"HT2ML Architecture Summary")
        print(f"{'='*60}")
        print(f"Trust Model: {self.trust_model.value}")
        print(f"Noise Budget: {self.noise_budget} bits")
        print(f"Total HE Cost: {self.get_total_he_cost()} bits")
        print(f"Feasible: {'✓ Yes' if self.is_feasible() else '✗ No'}")
        print(f"")

        print(f"Layer Breakdown:")
        for layer in self.layers:
            env = layer.execution_env
            cost = layer.get_noise_cost() if env == 'HE' else 0
            print(f"  Layer {layer.index}: {layer.input_size} → {layer.output_size}")
            print(f"    Environment: {env}")
            print(f"    Activation: {layer.activation}")
            print(f"    Noise Cost: {cost} bits")
        print(f"{'='*60}\n")


def design_ht2ml_architecture(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activations: List[str],
    noise_budget: int = 200,
    scale_bits: int = 40,
    max_he_layers: int = 2,
) -> HT2MLArchitecture:
    """
    Design an HT2ML architecture automatically.

    Strategy:
    1. Make first 1-2 layers HE (for input privacy)
    2. Make remaining layers TEE (for performance)
    3. Verify noise budget constraint

    Args:
        input_size: Input dimensionality
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimensionality
        activations: Activation for each layer
        noise_budget: Total noise budget (bits)
        scale_bits: CKKS scale parameter
        max_he_layers: Maximum HE layers (default: 2)

    Returns:
        HT2MLArchitecture instance

    Example:
        >>> arch = design_ht2ml_architecture(
        ...     input_size=784,
        ...     hidden_sizes=[128, 64],
        ...     output_size=10,
        ...     activations=['relu', 'relu']
        ... )
        >>> arch.print_summary()
    """
    # Build layer list
    all_sizes = [input_size] + hidden_sizes + [output_size]
    num_layers = len(all_sizes) - 1

    if len(activations) != num_layers:
        raise ValueError(
            f"Number of activations ({len(activations)}) must match "
            f"number of layers ({num_layers})"
        )

    layers = []
    he_cost = 0

    for i in range(num_layers):
        in_size = all_sizes[i]
        out_size = all_sizes[i + 1]
        activation = activations[i]

        # Decide execution environment
        # Use HE for first max_he_layers if budget allows
        if i < max_he_layers:
            # Tentatively use HE
            test_layer = HT2MLLayer(
                index=i,
                input_size=in_size,
                output_size=out_size,
                execution_env='HE',
                activation=activation
            )

            # Check if fits in budget
            if he_cost + test_layer.get_noise_cost(scale_bits) <= noise_budget:
                layers.append(test_layer)
                he_cost += test_layer.get_noise_cost(scale_bits)
                continue

        # Use TEE for remaining layers
        layer = HT2MLLayer(
            index=i,
            input_size=in_size,
            output_size=out_size,
            execution_env='TEE',
            activation=activation
        )
        layers.append(layer)

    return HT2MLArchitecture(
        layers=layers,
        trust_model=TrustModel.HYBRID,
        noise_budget=noise_budget,
    )


def compare_architectures(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    activations: List[str],
    noise_budget: int = 200,
) -> Dict[str, Any]:
    """
    Compare pure HE, pure TEE, and HT2ML architectures.

    Args:
        input_size: Input dimensionality
        hidden_sizes: Hidden layer sizes
        output_size: Output dimensionality
        activations: Activations per layer
        noise_budget: Noise budget for HE

    Returns:
        Dictionary with comparison results

    Example:
        >>> comparison = compare_architectures(
        ...     input_size=784,
        ...     hidden_sizes=[128, 64],
        ...     output_size=10,
        ...     activations=['relu', 'relu']
        ... )
        >>> print(comparison['recommendation'])
    """
    results = {}

    # Pure HE (all layers HE)
    all_sizes = [input_size] + hidden_sizes + [output_size]
    he_layers = [
        HT2MLLayer(
            index=i,
            input_size=all_sizes[i],
            output_size=all_sizes[i + 1],
            execution_env='HE',
            activation=activations[i] if i < len(activations) else 'none'
        )
        for i in range(len(all_sizes) - 1)
    ]

    he_arch = HT2MLArchitecture(
        layers=he_layers,
        trust_model=TrustModel.CRYPTOGRAPHIC,
        noise_budget=noise_budget,
    )

    # Pure TEE (all layers TEE)
    tee_layers = [
        HT2MLLayer(
            index=i,
            input_size=all_sizes[i],
            output_size=all_sizes[i + 1],
            execution_env='TEE',
            activation=activations[i] if i < len(activations) else 'none'
        )
        for i in range(len(all_sizes) - 1)
    ]

    tee_arch = HT2MLArchitecture(
        layers=tee_layers,
        trust_model=TrustModel.HARDWARE,
        noise_budget=noise_budget,
    )

    # HT2ML (hybrid)
    ht2ml_arch = design_ht2ml_architecture(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activations=activations,
        noise_budget=noise_budget,
    )

    # Comparison
    results['pure_he'] = {
        'feasible': he_arch.is_feasible(),
        'he_cost': he_arch.get_total_he_cost(),
        'num_he_layers': he_arch.get_num_he_layers(),
        'num_tee_layers': he_arch.get_num_tee_layers(),
        'trust_model': 'cryptographic',
    }

    results['pure_tee'] = {
        'feasible': True,  # TEE always feasible
        'he_cost': 0,
        'num_he_layers': tee_arch.get_num_he_layers(),
        'num_tee_layers': tee_arch.get_num_tee_layers(),
        'trust_model': 'hardware',
    }

    results['ht2ml'] = {
        'feasible': ht2ml_arch.is_feasible(),
        'he_cost': ht2ml_arch.get_total_he_cost(),
        'num_he_layers': ht2ml_arch.get_num_he_layers(),
        'num_tee_layers': ht2ml_arch.get_num_tee_layers(),
        'trust_model': 'hybrid',
    }

    # Recommendation
    if results['pure_he']['feasible']:
        recommendation = "Pure HE is feasible and provides maximum privacy"
    elif results['ht2ml']['feasible']:
        recommendation = "HT2ML hybrid is recommended (input privacy + performance)"
    else:
        recommendation = "Pure TEE is the only feasible option"

    results['recommendation'] = recommendation

    return results


def create_real_world_example() -> HT2MLArchitecture:
    """
    Create a real-world HT2ML architecture for phishing detection.

    Example:
    Email phishing classifier:
    - Input: TF-IDF features (784 dimensions)
    - Hidden: 128 → 64 units
    - Output: Binary classification (phishing vs. legitimate)

    Architecture:
    - Layer 1 (HE): 784 → 128, ReLU
      - Privacy: Encrypt email features
      - Noise cost: 128*40 + 200 = 5320 bits (too high!)

    Let's use a smaller example:
    - Input: PCA-reduced features (50 dimensions)
    - Hidden: 20 units
    - Output: Binary classification

    Architecture:
    - Layer 1 (HE): 50 → 20, ReLU
      - Noise cost: 20*40 + 200 = 1000 bits (still too high)

    Even smaller:
    - Input: 10 key features
    - Hidden: 5 units
    - Output: Binary classification

    Architecture:
    - Layer 1 (HE): 10 → 5, Sigmoid
      - Noise cost: 5*40 + 200 = 400 bits (exceeds 200-bit budget)

    Conclusion: Even with activation, single layer exceeds budget
    Solution: Use HT2ML with NO activation in HE layer
    """
    # Practical HT2ML architecture for phishing detection
    arch = HT2MLArchitecture(
        layers=[
            # Layer 1: HE - Simple feature extraction (no activation to save noise)
            HT2MLLayer(
                index=0,
                input_size=10,
                output_size=5,
                execution_env='HE',
                activation='none'  # No activation to save noise budget
            ),
            # Layer 2: TEE - Classification with activation
            HT2MLLayer(
                index=1,
                input_size=5,
                output_size=2,
                execution_env='TEE',
                activation='sigmoid'
            ),
        ],
        trust_model=TrustModel.HYBRID,
        noise_budget=200,
    )

    return arch


def generate_deployment_guide(
    architecture: HT2MLArchitecture,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate deployment guide for HT2ML architecture.

    Args:
        architecture: HT2MLArchitecture instance
        output_path: Optional path to save guide

    Returns:
        Deployment guide as string
    """
    guide = []
    guide.append("=" * 70)
    guide.append("HT2ML Deployment Guide")
    guide.append("=" * 70)
    guide.append("")

    # Architecture overview
    guide.append("## Architecture Overview")
    guide.append("")
    guide.append(f"Trust Model: {architecture.trust_model.value}")
    guide.append(f"Total Layers: {len(architecture.layers)}")
    guide.append(f"HE Layers: {architecture.get_num_he_layers()}")
    guide.append(f"TEE Layers: {architecture.get_num_tee_layers()}")
    guide.append("")

    # Deployment steps
    guide.append("## Deployment Steps")
    guide.append("")

    # Step 1: Client-side
    guide.append("### Step 1: Client-Side (Data Encryption)")
    guide.append("")
    guide.append("Responsibilities:")
    guide.append("- Collect and preprocess input data")
    guide.append("- Encrypt data using CKKS scheme")
    guide.append("- Send encrypted data to inference server")
    guide.append("")
    guide.append("Code Example:")
    guide.append("```python")
    guide.append("from he_ml.core.key_manager import create_ckks_context, generate_keys")
    guide.append("from he_ml.core.encryptor import encrypt_vector")
    guide.append("")
    guide.append("# Setup HE context")
    guide.append("ctx = create_ckks_context(poly_modulus_degree=8192, scale=2**40)")
    guide.append("keys = generate_keys(ctx)")
    guide.append("")
    guide.append("# Encrypt input data")
    guide.append("encrypted_input = encrypt_vector(user_data, ctx, scheme='ckks')")
    guide.append("")
    guide.append("# Send to server")
    guide.append("response = inference_server.predict(encrypted_input)")
    guide.append("```")
    guide.append("")

    # Step 2: HE Inference Server
    if architecture.get_num_he_layers() > 0:
        guide.append("### Step 2: HE Inference Server")
        guide.append("")
        guide.append("Responsibilities:")
        guide.append("- Receive encrypted data")
        guide.append("- Process initial HE layers")
        guide.append("- Forward partially processed result to TEE")
        guide.append("")
        guide.append("Privacy Properties:")
        guide.append("- ✓ Data remains encrypted throughout")
        guide.append("- ✓ Server learns nothing about input")
        guide.append("- ✓ Cryptographic privacy guarantees")
        guide.append("")

    # Step 3: TEE Inference
    if architecture.get_num_tee_layers() > 0:
        guide.append("### Step 3: TEE Inference (SGX Enclave)")
        guide.append("")
        guide.append("Responsibilities:")
        guide.append("- Receive data from HE layer")
        guide.append("- Decrypt within secure enclave")
        guide.append("- Process remaining layers")
        guide.append("- Return predictions")
        guide.append("")
        guide.append("Privacy Properties:")
        guide.append("- ✓ Hardware-enforced isolation")
        guide.append("- ✓ Attestation ensures correct code")
        guide.append("- ✓ Partial privacy (data decrypted in TEE)")
        guide.append("")

    # Security considerations
    guide.append("## Security Considerations")
    guide.append("")

    if architecture.trust_model == TrustModel.HYBRID:
        guide.append("HT2ML provides layered security:")
        guide.append("")
        guide.append("1. **Input Privacy (HE Layer)**")
        guide.append("   - Cryptographically protected")
        guide.append("   - No trust in server required")
        guide.append("   - Privacy: Maximum")
        guide.append("")
        guide.append("2. **Computation Privacy (TEE Layer)**")
        guide.append("   - Hardware-enforced isolation")
        guide.append("   - Trust in Intel SGX")
        guide.append("   - Privacy: Moderate")
        guide.append("")
        guide.append("**Overall Privacy:** High")
        guide.append("**Trust Required:** Cryptography OR Hardware")
    elif architecture.trust_model == TrustModel.CRYPTOGRAPHIC:
        guide.append("Pure HE provides:")
        guide.append("- ✓ Maximum privacy (cryptographic)")
        guide.append("- ✓ No trust in hardware")
        guide.append("- ✗ Limited by noise budget")
    else:
        guide.append("Pure TEE provides:")
        guide.append("- ✓ Fast computation")
        guide.append("- ✓ Unlimited depth")
        guide.append("- ✗ Trust in hardware manufacturer")

    guide.append("")
    guide.append("=" * 70)

    guide_str = "\n".join(guide)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(guide_str)

    return guide_str
