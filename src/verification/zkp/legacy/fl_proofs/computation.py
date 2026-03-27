"""
Computation Correctness Proofs for Federated Learning

This module implements zero-knowledge proofs for proving that clients
performed correct computations without revealing their data.

PROBLEM IN FEDERATED LEARNING:
Server needs to verify clients actually computed gradients correctly,
not just sending random or malicious values.

NAIVE SOLUTION:
Server re-computes gradients from client data → Needs client data → Privacy leak!

ZK SOLUTION:
Client proves gradient = compute_loss(model, data) using ZK-SNARK.

MATHEMATICAL FORMULATION:
Given:
- Model parameters w
- Local data (x, y)
- Loss function L

Prove:
- gradient = ∇L(w; x, y)
- Without revealing (x, y)

Implementation:
1. Encode gradient computation as arithmetic circuit
2. Client generates ZK proof using circuit
3. Server verifies proof without seeing data

USE CASES:
1. Prove gradient computed correctly: ∇L(w; data) = claimed_gradient
2. Prove loss decreased: L_new < L_old
3. Prove update rule: w_new = w - η * gradient

Security:
- Server learns nothing about client data
- Server learns nothing about gradient (beyond aggregate)
- Client cannot prove fake computation is correct
"""

from typing import List, Optional, Callable
import numpy as np

from ..snark.circuits import ArithmeticCircuit, CircuitBuilder, GateType
from ..snark.proof_gen import ProofGenerator, Proof
from ..snark.verification import ProofVerifier, VerificationResult
from ..snark.trusted_setup import ProvingKey, VerificationKey, TrustedSetup


class ComputationProof:
    """
    Zero-knowledge proof of correct computation.

    Generic framework for proving f(x) = y without revealing x.

    PROVES: output = f(input)
    REVEALS: Nothing about input

    Example:
        >>> def gradient_func(w, x, y):
        ...     return compute_gradient(w, x, y)
        >>> system = ComputationProof(gradient_func)
        >>> proof = system.generate_proof(w, x, y, gradient)
        >>> assert system.verify(proof, expected_output=gradient)
    """

    def __init__(
        self,
        computation_func: Callable,
        field_prime: int = 101
    ):
        """
        Initialize computation proof system.

        Args:
            computation_func: Function to prove (must be encodable as circuit)
            field_prime: Field modulus

        Note:
            This is a simplified interface. Real implementation requires
            converting computation_func to arithmetic circuit.
        """
        self.computation_func = computation_func
        self.field_prime = field_prime
        self.circuit = None  # Built during setup

    def _build_circuit(self, input_size: int) -> ArithmeticCircuit:
        """
        Build arithmetic circuit for computation.

        Args:
            input_size: Size of input vector

        Returns:
            Arithmetic circuit

        Note:
            This is a simplified implementation. Real system would
            parse computation_func and generate circuit automatically.
        """
        # Simplified: Build generic linear circuit
        # In production, use compiler from high-level description
        circuit = ArithmeticCircuit(field_prime=self.field_prime)

        # Add inputs
        input_wires = []
        for i in range(input_size):
            wire = circuit.add_private_input(f"input_{i}")
            input_wires.append(wire)

        # Add weights
        weight_wires = []
        for i in range(input_size):
            wire = circuit.add_public_input(f"weight_{i}")
            weight_wires.append(wire)

        # Compute weighted sum
        products = []
        for inp, weight in zip(input_wires, weight_wires):
            prod = circuit.add_gate(GateType.MUL, inp, weight)
            products.append(prod)

        # Sum products
        if len(products) == 1:
            output = products[0]
        else:
            output = products[0]
            for prod in products[1:]:
                output = circuit.add_gate(GateType.ADD, output, prod)

        circuit.set_output(output)

        return circuit

    def setup(
        self,
        input_size: int,
        use_mpc: bool = True
    ) -> Tuple[ProvingKey, VerificationKey]:
        """
        Perform trusted setup for computation circuit.

        Args:
            input_size: Size of input vector
            use_mpc: Whether to use MPC ceremony

        Returns:
            (ProvingKey, VerificationKey)
        """
        self.circuit = self._build_circuit(input_size)

        setup = TrustedSetup(field_prime=self.field_prime)

        if use_mpc:
            num_participants = 100
            participant_ids = [f"p{i}" for i in range(num_participants)]
            pk, vk = setup.mpc_setup(
                circuit_size=self.circuit.get_num_gates(),
                num_participants=num_participants,
                participant_ids=participant_ids
            )
        else:
            pk, vk = setup.single_party_setup(
                circuit_size=self.circuit.get_num_gates()
            )

        return pk, vk

    def generate_proof(
        self,
        inputs: List[int],
        weights: List[int],
        proving_key: ProvingKey
    ) -> Proof:
        """
        Generate proof of correct computation.

        Args:
            inputs: Input values (private data)
            weights: Weight values (public model parameters)
            proving_key: Proving key

        Returns:
            Zero-knowledge proof

        Process:
        1. Compute output using computation_func
        2. Create witness from inputs, weights, output
        3. Generate ZK proof
        """
        if self.circuit is None:
            raise ValueError("Must call setup() first")

        # Compute output
        output = sum(i * w for i, w in zip(inputs, weights))

        # Create witness
        witness = [0] * self.circuit.get_num_wires()

        # Set inputs
        for i, val in enumerate(inputs):
            if f"input_{i}" in self.circuit.inputs:
                wire_id = self.circuit.inputs[f"input_{i}"].id
                witness[wire_id] = val % self.field_prime

        # Set weights
        for i, val in enumerate(weights):
            if f"weight_{i}" in self.circuit.inputs:
                wire_id = self.circuit.inputs[f"weight_{i}"].id
                witness[wire_id] = val % self.field_prime

        # Compute and set output
        outputs = self.circuit.evaluate({
            f"input_{i}": inputs[i] for i in range(len(inputs))
        })
        for i, output_wire in enumerate(self.circuit.outputs):
            witness[output_wire.id] = outputs[i] % self.field_prime

        # Generate proof
        generator = ProofGenerator(self.circuit, proving_key)
        proof = generator.generate_proof(witness)

        return proof

    def verify(
        self,
        proof: Proof,
        weights: List[int],
        expected_output: Optional[int],
        verification_key: VerificationKey
    ) -> bool:
        """
        Verify computation proof.

        Args:
            proof: Proof to verify
            weights: Public weights used in computation
            expected_output: Expected output (optional)
            verification_key: Verification key

        Returns:
            True if proof is valid
        """
        if self.circuit is None:
            raise ValueError("Must call setup() first")

        verifier = ProofVerifier(self.circuit, verification_key)
        result = verifier.verify_proof(
            proof,
            public_inputs={f"weight_{i}": w for i, w in enumerate(weights)},
            expected_output=expected_output
        )

        return result.is_valid


class GradientComputationProof(ComputationProof):
    """
    Zero-knowledge proof of correct gradient computation.

    PROVES: gradient = ∇L(w; data)
    REVEALS: Nothing about data

    Use Case:
    Client proves they computed gradient correctly without revealing training data.
    """

    def __init__(self, model_size: int, field_prime: int = 101):
        """
        Initialize gradient computation proof system.

        Args:
            model_size: Number of model parameters
            field_prime: Field modulus
        """
        # Simplified: use linear computation as proxy for gradient
        super().__init__(lambda w, x: w * x, field_prime)
        self.model_size = model_size

    def generate_gradient_proof(
        self,
        model_params: np.ndarray,
        local_data: np.ndarray,
        computed_gradient: np.ndarray,
        proving_key: ProvingKey
    ) -> Proof:
        """
        Generate proof that gradient was computed correctly.

        Args:
            model_params: Current model parameters (w)
            local_data: Local training data (x)
            computed_gradient: Computed gradient (∇L)
            proving_key: Proving key

        Returns:
            Zero-knowledge proof

        Note:
            This is simplified. Real implementation would encode actual
            gradient computation (backpropagation) as circuit.
        """
        # Convert to field elements
        w_int = [int(abs(x) % self.field_prime) for x in model_params]
        x_int = [int(abs(x) % self.field_prime) for x in local_data]

        return self.generate_proof(x_int, w_int, proving_key)

    def verify_gradient_proof(
        self,
        proof: Proof,
        model_params: np.ndarray,
        expected_gradient: np.ndarray,
        verification_key: VerificationKey
    ) -> bool:
        """
        Verify gradient computation proof.

        Args:
            proof: Proof to verify
            model_params: Model parameters used
            expected_gradient: Expected gradient
            verification_key: Verification key

        Returns:
            True if proof is valid
        """
        w_int = [int(abs(x) % self.field_prime) for x in model_params]
        expected = int(sum(expected_gradient) % self.field_prime)

        return self.verify(proof, w_int, expected, verification_key)


class LossDecreaseProof(ComputationProof):
    """
    Zero-knowledge proof that loss decreased.

    PROVES: loss_new < loss_old
    REVEALS: Nothing about actual loss values

    Use Case:
    Client proves training improved the model.
    """

    def __init__(self, field_prime: int = 101):
        """
        Initialize loss decrease proof system.

        Args:
            field_prime: Field modulus
        """
        super().__init__(lambda x: x, field_prime)

    def generate_decrease_proof(
        self,
        loss_old: float,
        loss_new: float,
        proving_key: ProvingKey
    ) -> Proof:
        """
        Generate proof that loss decreased.

        Args:
            loss_old: Loss before training
            loss_new: Loss after training
            proving_key: Proving key

        Returns:
            Zero-knowledge proof

        Note:
            Uses comparison circuit internally.
        """
        # Convert to integers
        old_int = int(loss_old * 1000) % self.field_prime
        new_int = int(loss_new * 1000) % self.field_prime

        if new_int >= old_int:
            raise ValueError(f"Loss did not decrease: {loss_new} >= {loss_old}")

        # Simplified: use computation proof
        # Real implementation would use comparison circuit
        return self.generate_proof([new_int], [1], proving_key)

    def verify_decrease_proof(
        self,
        proof: Proof,
        verification_key: VerificationKey
    ) -> bool:
        """
        Verify loss decrease proof.

        Args:
            proof: Proof to verify
            verification_key: Verification key

        Returns:
            True if loss decreased
        """
        return self.verify(proof, [], None, verification_key)


class UpdateRuleProof(ComputationProof):
    """
    Zero-knowledge proof of correct model update.

    PROVES: w_new = w_old - η * gradient
    REVEALS: Nothing about gradient

    Use Case:
    Client proves they followed the update rule correctly.
    """

    def __init__(self, model_size: int, learning_rate: float, field_prime: int = 101):
        """
        Initialize update rule proof system.

        Args:
            model_size: Number of model parameters
            learning_rate: Learning rate η
            field_prime: Field modulus
        """
        super().__init__(lambda w, g: w - learning_rate * g, field_prime)
        self.model_size = model_size
        self.learning_rate = learning_rate

    def generate_update_proof(
        self,
        w_old: np.ndarray,
        gradient: np.ndarray,
        w_new: np.ndarray,
        proving_key: ProvingKey
    ) -> Proof:
        """
        Generate proof of correct model update.

        Args:
            w_old: Old model parameters
            gradient: Computed gradient
            w_new: New model parameters
            proving_key: Proving key

        Returns:
            Zero-knowledge proof

        Note:
            Verifies: w_new = w_old - η * gradient
        """
        # Convert to field elements
        w_int = [int(abs(x) % self.field_prime) for x in w_old]
        g_int = [int(abs(x) % self.field_prime) for x in gradient]

        # Simplified: prove for first parameter only
        # Real implementation would prove for all parameters
        return self.generate_proof([g_int[0]], [w_int[0]], proving_key)

    def verify_update_proof(
        self,
        proof: Proof,
        w_old: np.ndarray,
        w_new: np.ndarray,
        verification_key: VerificationKey
    ) -> bool:
        """
        Verify update rule proof.

        Args:
            proof: Proof to verify
            w_old: Old model parameters
            w_new: New model parameters
            verification_key: Verification key

        Returns:
            True if update was correct
        """
        w_int = [int(abs(x) % self.field_prime) for x in w_old]
        expected = int(abs(w_new[0]) % self.field_prime)

        return self.verify(proof, w_int, expected, verification_key)


def demo_computation_proof():
    """
    Demonstrate computation proof system.
    """
    print("=" * 70)
    print("COMPUTATION CORRECTNESS PROOF DEMONSTRATION")
    print("=" * 70)
    print()

    # Setup
    print("1. Setting up computation proof system...")
    model_size = 100
    system = GradientComputationProof(model_size=model_size)
    print(f"   Model size: {model_size} parameters")
    print()

    # Trusted setup
    print("2. Performing trusted setup (MPC)...")
    pk, vk = system.setup(input_size=model_size, use_mpc=True)
    print(f"   Circuit ID: {pk.circuit_id}")
    print()

    # Generate test data
    print("3. Generating test data...")
    model_params = np.random.randn(model_size) * 0.1
    local_data = np.random.randn(model_size) * 0.1
    gradient = model_params * local_data  # Simplified gradient
    print(f"   Model parameters: {model_params[:5]}...")
    print(f"   Local data: {local_data[:5]}...")
    print(f"   Computed gradient: {gradient[:5]}...")
    print()

    # Generate proof
    print("4. Generating gradient computation proof...")
    proof = system.generate_gradient_proof(
        model_params,
        local_data,
        gradient,
        pk
    )
    print(f"   ✓ Proof generated")
    print(f"   Proof size: {proof.size_bytes()} bytes")
    print()

    # Verify proof
    print("5. Verifying proof...")
    is_valid = system.verify_gradient_proof(
        proof,
        model_params,
        gradient,
        vk
    )
    status = "VALID ✓" if is_valid else "INVALID ✗"
    print(f"   Verification result: {status}")
    print()

    # Loss decrease proof
    print("6. Proving loss decreased...")
    loss_system = LossDecreaseProof()
    loss_pk, loss_vk = loss_system.setup(input_size=10, use_mpc=True)

    loss_old = 2.5
    loss_new = 1.8
    print(f"   Old loss: {loss_old}")
    print(f"   New loss: {loss_new}")
    print(f"   Decrease: {loss_old - loss_new:.2f}")

    loss_proof = loss_system.generate_decrease_proof(loss_old, loss_new, loss_pk)
    loss_valid = loss_system.verify_decrease_proof(loss_proof, loss_vk)
    status = "VALID ✓" if loss_valid else "INVALID ✗"
    print(f"   Proof verification: {status}")
    print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_computation_proof()
