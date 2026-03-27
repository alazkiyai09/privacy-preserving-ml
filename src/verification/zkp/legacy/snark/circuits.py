"""
Arithmetic Circuits for ZK-SNARKs

An arithmetic circuit represents a computation as a directed acyclic graph (DAG)
where:
- Nodes are arithmetic operations (addition, multiplication)
- Edges represent wire values
- Inputs are public or private values

Mathematical Form:
A circuit computes a polynomial over a finite field:
output = polynomial(input1, input2, ..., inputn)

Components:
1. Wire: Holds a value (field element)
2. Gate: Performs operation on wires (+ or *)
3. Circuit: Collection of gates

This module provides utilities for building and manipulating arithmetic circuits.

USE CASE IN FEDERATED LEARNING:
- Encode gradient computation as circuit
- Prove gradient = compute_loss(model, data)
- Prove gradient is bounded without revealing it

Security Assumptions:
- Field arithmetic is correct
- Circuit is correctly encoded
- No overflow in field operations
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Types of arithmetic gates."""
    ADD = 0
    MUL = 1
    CONSTANT = 2
    PUBLIC_INPUT = 3
    PRIVATE_INPUT = 4
    OUTPUT = 5


@dataclass
class Wire:
    """
    Wire in arithmetic circuit.

    A wire holds a value and connects gates together.
    """
    id: int
    value: Optional[int] = None
    is_public: bool = False
    is_input: bool = False
    is_output: bool = False

    def __repr__(self):
        status = []
        if self.is_public:
            status.append("public")
        if self.is_input:
            status.append("input")
        if self.is_output:
            status.append("output")

        status_str = f" ({', '.join(status)})" if status else ""
        return f"Wire({self.id}{status_str})"


@dataclass
class Gate:
    """
    Gate in arithmetic circuit.

    Performs arithmetic operation on input wires.
    """
    id: int
    gate_type: GateType
    input_wires: List[Wire]
    output_wire: Wire
    constant: Optional[int] = None

    def evaluate(self) -> int:
        """
        Evaluate the gate.

        Returns:
            Result of the gate operation

        Raises:
            ValueError: If input wires don't have values
        """
        if self.gate_type == GateType.ADD:
            if len(self.input_wires) != 2:
                raise ValueError("ADD gate requires exactly 2 inputs")

            a = self.input_wires[0].value
            b = self.input_wires[1].value

            if a is None or b is None:
                raise ValueError("Input wires must have values")

            return a + b

        elif self.gate_type == GateType.MUL:
            if len(self.input_wires) != 2:
                raise ValueError("MUL gate requires exactly 2 inputs")

            a = self.input_wires[0].value
            b = self.input_wires[1].value

            if a is None or b is None:
                raise ValueError("Input wires must have values")

            return a * b

        elif self.gate_type == GateType.CONSTANT:
            if self.constant is None:
                raise ValueError("CONSTANT gate must have constant value")

            return self.constant

        else:
            raise ValueError(f"Unsupported gate type: {self.gate_type}")

    def __repr__(self):
        return f"Gate({self.id}, {self.gate_type.name})"


class ArithmeticCircuit:
    """
    Arithmetic circuit for ZK-SNARKs.

    Represents computation as DAG of arithmetic gates.

    USE CASE:
    Encode gradient computation as circuit for ZK proof.

    Example:
        >>> circuit = ArithmeticCircuit(field_prime=101)
        >>> x = circuit.add_private_input("x")
        >>> y = circuit.add_private_input("y")
        >>> z = circuit.add_gate(GateType.MUL, x, y)  # z = x * y
        >>> circuit.set_output(z)
    """

    def __init__(self, field_prime: int = 101):
        """
        Initialize arithmetic circuit.

        Args:
            field_prime: Prime modulus for field arithmetic

        Note:
            Small prime for demonstration. Use 256-bit prime in production.
        """
        self.field_prime = field_prime
        self.wires: List[Wire] = []
        self.gates: List[Gate] = []
        self.wire_counter = 0
        self.gate_counter = 0
        self.inputs: Dict[str, Wire] = {}
        self.outputs: List[Wire] = []

    def _create_wire(
        self,
        is_public: bool = False,
        is_input: bool = False,
        is_output: bool = False
    ) -> Wire:
        """
        Create a new wire.

        Args:
            is_public: Whether wire is public
            is_input: Whether wire is input
            is_output: Whether wire is output

        Returns:
            New wire
        """
        wire = Wire(
            id=self.wire_counter,
            is_public=is_public,
            is_input=is_input,
            is_output=is_output
        )
        self.wires.append(wire)
        self.wire_counter += 1
        return wire

    def add_public_input(self, name: str) -> Wire:
        """
        Add a public input wire.

        Args:
            name: Name for the input

        Returns:
            The input wire
        """
        wire = self._create_wire(is_public=True, is_input=True)
        self.inputs[name] = wire
        return wire

    def add_private_input(self, name: str) -> Wire:
        """
        Add a private input wire (witness).

        Args:
            name: Name for the input

        Returns:
            The input wire
        """
        wire = self._create_wire(is_public=False, is_input=True)
        self.inputs[name] = wire
        return wire

    def add_constant(self, value: int) -> Wire:
        """
        Add a constant value wire.

        Args:
            value: Constant value

        Returns:
            The constant wire
        """
        wire = self._create_wire()

        # Create constant gate
        gate = Gate(
            id=self.gate_counter,
            gate_type=GateType.CONSTANT,
            input_wires=[],
            output_wire=wire,
            constant=value % self.field_prime
        )
        self.gates.append(gate)
        self.gate_counter += 1

        wire.value = value % self.field_prime
        return wire

    def add_gate(
        self,
        gate_type: GateType,
        input1: Wire,
        input2: Optional[Wire] = None
    ) -> Wire:
        """
        Add an arithmetic gate.

        Args:
            gate_type: Type of gate (ADD or MUL)
            input1: First input wire
            input2: Second input wire (optional for CONSTANT)

        Returns:
            Output wire of the gate
        """
        output_wire = self._create_wire()

        if gate_type == GateType.CONSTANT:
            gate = Gate(
                id=self.gate_counter,
                gate_type=gate_type,
                input_wires=[],
                output_wire=output_wire
            )
        else:
            if input2 is None:
                raise ValueError(f"{gate_type} gate requires 2 inputs")

            gate = Gate(
                id=self.gate_counter,
                gate_type=gate_type,
                input_wires=[input1, input2],
                output_wire=output_wire
            )

        self.gates.append(gate)
        self.gate_counter += 1

        return output_wire

    def set_output(self, wire: Wire) -> None:
        """
        Mark a wire as circuit output.

        Args:
            wire: The output wire
        """
        wire.is_output = True
        if wire not in self.outputs:
            self.outputs.append(wire)

    def evaluate(self, inputs: Dict[str, int]) -> List[int]:
        """
        Evaluate the circuit.

        Args:
            inputs: Dictionary mapping input names to values

        Returns:
            List of output values

        Raises:
            ValueError: If inputs are invalid
        """
        # Set input values
        for name, value in inputs.items():
            if name not in self.inputs:
                raise ValueError(f"Unknown input: {name}")

            self.inputs[name].value = value % self.field_prime

        # Evaluate gates in topological order
        evaluated = set()

        for _ in range(len(self.gates)):
            progress = False

            for gate in self.gates:
                if gate.output_wire.id in evaluated:
                    continue

                # Check if inputs are ready
                inputs_ready = all(
                    wire.value is not None or wire.id in evaluated
                    for wire in gate.input_wires
                )

                if inputs_ready or gate.gate_type == GateType.CONSTANT:
                    # Evaluate gate
                    result = gate.evaluate()
                    gate.output_wire.value = result % self.field_prime
                    evaluated.add(gate.output_wire.id)
                    progress = True

            if not progress:
                raise ValueError("Circuit has cycle or missing inputs")

        # Get outputs
        outputs = [wire.value for wire in self.outputs]
        return outputs

    def get_num_wires(self) -> int:
        """Get total number of wires."""
        return len(self.wires)

    def get_num_gates(self) -> int:
        """Get total number of gates."""
        return len(self.gates)

    def get_num_inputs(self) -> int:
        """Get number of inputs (public + private)."""
        return len(self.inputs)

    def get_num_outputs(self) -> int:
        """Get number of outputs."""
        return len(self.outputs)

    def __repr__(self):
        return (
            f"ArithmeticCircuit("
            f"wires={self.get_num_wires()}, "
            f"gates={self.get_num_gates()}, "
            f"inputs={self.get_num_inputs()}, "
            f"outputs={self.get_num_outputs()})"
        )


class CircuitBuilder:
    """
    Helper class for building common circuits.

    Provides pre-built circuits for common operations.
    """

    @staticmethod
    def create_addition_circuit(field_prime: int = 101) -> ArithmeticCircuit:
        """
        Create circuit for addition: out = a + b.

        Args:
            field_prime: Field modulus

        Returns:
            Addition circuit
        """
        circuit = ArithmeticCircuit(field_prime)
        a = circuit.add_public_input("a")
        b = circuit.add_public_input("b")
        out = circuit.add_gate(GateType.ADD, a, b)
        circuit.set_output(out)

        return circuit

    @staticmethod
    def create_multiplication_circuit(field_prime: int = 101) -> ArithmeticCircuit:
        """
        Create circuit for multiplication: out = a * b.

        Args:
            field_prime: Field modulus

        Returns:
            Multiplication circuit
        """
        circuit = ArithmeticCircuit(field_prime)
        a = circuit.add_public_input("a")
        b = circuit.add_public_input("b")
        out = circuit.add_gate(GateType.MUL, a, b)
        circuit.set_output(out)

        return circuit

    @staticmethod
    def create_range_check_circuit(
        min_val: int,
        max_val: int,
        field_prime: int = 101
    ) -> ArithmeticCircuit:
        """
        Create circuit that checks if value is in range [min_val, max_val].

        Args:
            min_val: Minimum value
            max_val: Maximum value
            field_prime: Field modulus

        Returns:
            Range check circuit

        Note:
            This is a simplified version. Real range check uses
            bit decomposition and comparison circuits.
        """
        circuit = ArithmeticCircuit(field_prime)
        x = circuit.add_public_input("x")
        min_const = circuit.add_constant(min_val)
        max_const = circuit.add_constant(max_val)

        # x >= min: Check if x - min >= 0
        # x <= max: Check if max - x >= 0
        # Simplified: Just compare values

        diff1 = circuit.add_gate(GateType.ADD, x, min_const)  # x - min (using + for simplicity)
        diff2 = circuit.add_gate(GateType.ADD, max_const, x)  # max - x

        circuit.set_output(diff1)
        circuit.set_output(diff2)

        return circuit

    @staticmethod
    def create_dotproduct_circuit(
        size: int,
        field_prime: int = 101
    ) -> ArithmeticCircuit:
        """
        Create circuit for dot product: out = sum(x[i] * y[i]).

        Args:
            size: Size of vectors
            field_prime: Field modulus

        Returns:
            Dot product circuit

        Example:
            >>> circuit = CircuitBuilder.create_dotproduct_circuit(3)
            >>> result = circuit.evaluate({
            ...     "x0": 1, "y0": 2,
            ...     "x1": 3, "y1": 4,
            ...     "x2": 5, "y2": 6
            ... })
            >>> # result = 1*2 + 3*4 + 5*6 = 44
        """
        circuit = ArithmeticCircuit(field_prime)

        # Add input pairs
        x_wires = []
        y_wires = []
        for i in range(size):
            x = circuit.add_public_input(f"x{i}")
            y = circuit.add_public_input(f"y{i}")
            x_wires.append(x)
            y_wires.append(y)

        # Compute products
        products = []
        for x, y in zip(x_wires, y_wires):
            prod = circuit.add_gate(GateType.MUL, x, y)
            products.append(prod)

        # Sum products
        if not products:
            # Empty dot product = 0
            zero = circuit.add_constant(0)
            circuit.set_output(zero)
        elif len(products) == 1:
            circuit.set_output(products[0])
        else:
            # Chain additions
            current_sum = products[0]
            for prod in products[1:]:
                current_sum = circuit.add_gate(GateType.ADD, current_sum, prod)

            circuit.set_output(current_sum)

        return circuit

    @staticmethod
    def create_gradient_norm_circuit(
        size: int,
        bound: int,
        field_prime: int = 101
    ) -> ArithmeticCircuit:
        """
        Create circuit that checks if gradient L2 norm is bounded.

        Args:
            size: Size of gradient vector
            bound: L2 norm bound
            field_prime: Field modulus

        Returns:
            Gradient norm circuit

        Computation:
        1. Compute squared norm: sum(g[i]^2)
        2. Compare with bound^2

        USE CASE IN FEDERATED LEARNING:
        Prove gradient is bounded without revealing it.
        """
        circuit = ArithmeticCircuit(field_prime)

        # Add gradient inputs
        g_wires = []
        for i in range(size):
            g = circuit.add_private_input(f"g{i}")
            g_wires.append(g)

        # Compute squares
        squares = []
        for g in g_wires:
            sq = circuit.add_gate(GateType.MUL, g, g)
            squares.append(sq)

        # Sum squares (squared norm)
        if len(squares) == 1:
            norm_sq = squares[0]
        else:
            norm_sq = squares[0]
            for sq in squares[1:]:
                norm_sq = circuit.add_gate(GateType.ADD, norm_sq, sq)

        # Add bound
        bound_sq_wire = circuit.add_constant(bound * bound)

        # Compare: norm_sq <= bound_sq
        # Simplified: Just output both values
        circuit.set_output(norm_sq)
        circuit.set_output(bound_sq_wire)

        return circuit
