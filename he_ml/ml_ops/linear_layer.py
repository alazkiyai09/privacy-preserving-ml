"""
Linear Layer for Homomorphic Encryption
========================================
Encrypted implementation of neural network linear layers.

Formula: y = xW^T + b

Where:
- x: Input vector (batch_size, in_features)
- W: Weight matrix (out_features, in_features)
- b: Bias vector (out_features,)
- y: Output vector (batch_size, out_features)
"""

from typing import List, Optional, Any
from dataclasses import dataclass
import numpy as np
import tenseal as ts

# Type aliases
CiphertextVector = Any
SecretKey = Any
RelinKeys = Any
GaloisKeys = Any


@dataclass
class LinearLayerConfig:
    """Configuration for encrypted linear layer."""
    in_features: int
    out_features: int
    use_bias: bool = True

    def __post_init__(self):
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")


class EncryptedLinearLayer:
    """
    Linear layer for encrypted neural network inference.

    Computes: y = xW^T + b

    This is THE building block for neural networks operating on encrypted data.

    Architecture:
        Input (in_features) → Weights (out_features × in_features) → Output (out_features)
                              ↓ Bias (out_features)

    Noise Cost:
    - Each output dimension: ~log2(scale) bits
    - Total: out_features * log2(scale) bits
    - Example: For 100 outputs with scale=2^40: ~100 × 40 = 4000 bits!
    - This is why we need to limit layer size in HE!

    Args:
        in_features: Number of input features
        out_features: Number of output features
        weights: Plaintext weight matrix (out_features × in_features)
        bias: Plaintext bias vector (out_features,) or None
        context: TenSEAL context (for creating encrypted weights if needed)

    Example:
        >>> layer = EncryptedLinearLayer(
        ...     in_features=3,
        ...     out_features=2,
        ...     weights=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ...     bias=np.array([0.5, 1.0])
        ... )
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> encrypted_y = layer.forward(encrypted_x, relin_key)
        >>> # Decrypt to get [1*1+2*2+3*3+0.5, 1*4+2*5+3*6+1.0] = [14.5, 33.0]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
        use_bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Validate shapes
        if weights.shape != (out_features, in_features):
            raise ValueError(
                f"Weight shape must be ({out_features}, {in_features}), "
                f"got {weights.shape}"
            )

        self.weights = weights

        if bias is not None:
            if bias.shape != (out_features,):
                raise ValueError(
                    f"Bias shape must be ({out_features},), got {bias.shape}"
                )
            self.bias = bias
        elif use_bias:
            # Initialize zero bias
            self.bias = np.zeros(out_features)
        else:
            self.bias = None

    @classmethod
    def random(
        cls,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        std: float = 0.1,
        seed: Optional[int] = None,
    ) -> 'EncryptedLinearLayer':
        """
        Create a layer with random weights (for testing).

        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to use bias
            std: Standard deviation for weight initialization
            seed: Random seed

        Returns:
            Initialized EncryptedLinearLayer
        """
        if seed is not None:
            np.random.seed(seed)

        weights = np.random.randn(out_features, in_features) * std
        bias = np.random.randn(out_features) * std if use_bias else None

        return cls(in_features, out_features, weights, bias, use_bias)

    def forward(
        self,
        encrypted_input: CiphertextVector,
        relin_key: RelinKeys,
    ) -> List[CiphertextVector]:
        """
        Forward pass through encrypted linear layer.

        Computes: y = xW^T + b

        Args:
            encrypted_input: Encrypted input vector (size in_features)
            relin_key: Relinearization key

        Returns:
            List of encrypted output values (size out_features)

        Noise Cost:
            out_features * log2(scale) bits

        Example:
            >>> layer = EncryptedLinearLayer(3, 2, weights, bias)
            >>> x = np.array([1.0, 2.0, 3.0])
            >>> encrypted_x = encrypt_vector(x, ctx)
            >>> encrypted_y = layer.forward(encrypted_x, relin_key)
            >>> # Decrypt to get output
        """
        from he_ml.ml_ops.matrix_ops import (
            encrypted_plain_matrix_vector_multiply_with_bias
        )

        # Compute xW^T + b
        result = encrypted_plain_matrix_vector_multiply_with_bias(
            encrypted_input,
            self.weights.T,  # Transpose for correct orientation
            self.bias if self.use_bias else np.zeros(self.out_features),
            relin_key
        )

        return result

    def forward_batch(
        self,
        encrypted_batch: List[CiphertextVector],
        relin_key: RelinKeys,
    ) -> List[List[CiphertextVector]]:
        """
        Forward pass for batch of encrypted inputs.

        Args:
            encrypted_batch: List of encrypted input vectors
            relin_key: Relinearization key

        Returns:
            List of lists of encrypted outputs (batch_size × out_features)

        Example:
            >>> batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
            >>> encrypted_batch = [encrypt_vector(v, ctx) for v in batch]
            >>> results = layer.forward_batch(encrypted_batch, relin_key)
        """
        batch_results = []

        for encrypted_input in encrypted_batch:
            result = self.forward(encrypted_input, relin_key)
            batch_results.append(result)

        return batch_results

    def get_output_size(self) -> int:
        """Get output feature dimension."""
        return self.out_features

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        weight_count = self.in_features * self.out_features
        bias_count = self.out_features if self.use_bias else 0
        return weight_count + bias_count

    def print_layer_info(self, title: str = "Linear Layer") -> None:
        """Print layer information."""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"  Input features:  {self.in_features}")
        print(f"  Output features: {self.out_features}")
        print(f"  Use bias:        {self.use_bias}")
        print(f"  Parameters:      {self.get_parameter_count():,}")
        print(f"  Weight shape:    {self.weights.shape}")
        if self.use_bias:
            print(f"  Bias shape:      {self.bias.shape}")

        # Estimate noise cost
        scale_bits = 40  # Assuming scale=2^40
        noise_cost = self.out_features * scale_bits
        print(f"\n  Estimated noise cost:")
        print(f"    Per forward pass: ~{noise_cost} bits")
        print(f"    With 200-bit budget: ~{200 // noise_cost} passes maximum")
        print(f"{'='*50}\n")


def create_sequential_model(
    layers: List[EncryptedLinearLayer],
) -> 'EncryptedSequential':
    """
    Create a sequential model from a list of layers.

    Args:
        layers: List of EncryptedLinearLayer instances

    Returns:
        EncryptedSequential model

    Example:
        >>> layer1 = EncryptedLinearLayer(10, 5, weights1, bias1)
        >>> layer2 = EncryptedLinearLayer(5, 2, weights2, bias2)
        >>> model = create_sequential_model([layer1, layer2])
        >>> x = np.random.randn(10)
        >>> encrypted_x = encrypt_vector(x, ctx)
        >>> output = model.forward(encrypted_x, relin_key)
    """
    return EncryptedSequential(layers)


class EncryptedSequential:
    """
    Sequential model for encrypted neural network inference.

    Chains multiple linear layers together.

    WARNING: Each layer consumes noise budget!
    With 200-bit budget and typical layers (~80 bits), you can only do ~2 layers.

    This is why HT2ML uses:
    - Layer 1-2: HE (privacy-preserving input processing)
    - Layer 3+: TEE (fast, unlimited depth)

    Args:
        layers: List of EncryptedLinearLayer instances
    """

    def __init__(self, layers: List[EncryptedLinearLayer]):
        self.layers = layers

        # Validate layer connections
        for i in range(len(layers) - 1):
            if layers[i].out_features != layers[i+1].in_features:
                raise ValueError(
                    f"Layer {i} output ({layers[i].out_features}) must match "
                    f"layer {i+1} input ({layers[i+1].in_features})"
                )

    def forward(
        self,
        encrypted_input: CiphertextVector,
        relin_key: RelinKeys,
    ) -> List[CiphertextVector]:
        """
        Forward pass through all layers.

        Args:
            encrypted_input: Encrypted input vector
            relin_key: Relinearization key

        Returns:
            List of encrypted output values from final layer

        Example:
            >>> model = create_sequential_model([layer1, layer2, layer3])
            >>> x = np.random.randn(10)
            >>> encrypted_x = encrypt_vector(x, ctx)
            >>> output = model.forward(encrypted_x, relin_key)
            >>> # Decrypt to get final predictions
        """
        current_input = encrypted_input

        for i, layer in enumerate(self.layers):
            output = layer.forward(current_input, relin_key)

            # Convert list of ciphertexts to single ciphertext for next layer
            # (assuming next layer expects single input)
            if i < len(self.layers) - 1:
                # Combine outputs for next layer
                # This is simplified - actual implementation depends on use case
                current_input = output[0] if len(output) == 1 else output
            else:
                return output

    def get_total_parameters(self) -> int:
        """Get total number of parameters in all layers."""
        return sum(layer.get_parameter_count() for layer in self.layers)

    def get_layer_count(self) -> int:
        """Get number of layers."""
        return len(self.layers)

    def print_model_summary(self, title: str = "Encrypted Model") -> None:
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total layers: {self.get_layer_count()}")
        print(f"Total parameters: {self.get_total_parameters():,}")

        for i, layer in enumerate(self.layers):
            print(f"\nLayer {i+1}:")
            print(f"  Input:  {layer.in_features} features")
            print(f"  Output: {layer.out_features} features")
            print(f"  Params: {layer.get_parameter_count():,}")

        print(f"\n{'='*60}\n")

    def estimate_max_depth(
        self,
        noise_budget: int = 200,
        scale: float = 2**40,
    ) -> int:
        """
        Estimate maximum number of layers that can be computed.

        Args:
            noise_budget: Total noise budget (bits)
            scale: CKKS scale parameter

        Returns:
            Maximum number of layers before noise exhaustion

        Example:
            >>> model = create_sequential_model([layer1, layer2, layer3])
            >>> max_layers = model.estimate_max_depth(noise_budget=200)
            >>> print(f"Can compute {max_layers} layers")
        """
        # Estimate noise per layer (simplified)
        scale_bits = int(np.log2(scale))
        noise_per_layer = self.layers[0].out_features * scale_bits

        max_depth = noise_budget // noise_per_layer

        return max_depth


def validate_linear_layer(
    layer: EncryptedLinearLayer,
    plaintext_input: np.ndarray,
    context: Any,
    secret_key: Any,
    tolerance: float = 1e-3,
) -> bool:
    """
    Validate encrypted linear layer against plaintext computation.

    Args:
        layer: EncryptedLinearLayer instance
        plaintext_input: Test input vector
        context: TenSEAL context
        secret_key: Secret key
        tolerance: Max allowed error

    Returns:
        True if validation passes
    """
    # Generate keys
    context.generate_relin_keys()
    relin_key = context.relin_keys()

    # Encrypt input
    from he_ml.core.encryptor import encrypt_vector
    encrypted_input = encrypt_vector(plaintext_input, context, scheme='ckks')

    # Forward pass
    encrypted_output = layer.forward(encrypted_input, relin_key)

    # Decrypt
    decrypted = np.array([ct.decrypt(secret_key)[0] for ct in encrypted_output])

    # Compute expected
    expected = plaintext_input @ layer.weights.T
    if layer.use_bias:
        expected += layer.bias

    # Compare
    error = np.max(np.abs(expected - decrypted))

    if error > tolerance:
        print(f"Validation failed: max error = {error:.2e}")
        print(f"Expected: {expected}")
        print(f"Got:      {decrypted}")
        return False

    return True
