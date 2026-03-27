"""
Rank-1 Constraint System (R1CS) for ZK-SNARKs

R1CS is a standard format for encoding arithmetic circuits as systems of
quadratic constraints. Each constraint has the form:

<A, z> * <B, z> = <C, z>

where:
- z is the assignment vector (witness + public inputs + outputs)
- A, B, C are vectors of coefficients
- <.,.> denotes inner product

R1CS is the intermediate representation between arithmetic circuits and
Quadratic Arithmetic Programs (QAP).

This module provides utilities for creating and manipulating R1CS.

USE CASE IN FEDERATED LEARNING:
- Encode gradient computation as R1CS
- Convert to QAP for efficient proof generation
- Verify computations without revealing private data

Mathematical Background:
For a circuit with n wires and m gates:
- z ∈ F^n is the assignment
- Each gate creates one constraint
- Total constraints: m (one per gate)

Security Assumptions:
- Field arithmetic is correct
- Constraints correctly encode circuit
- QAP interpolation is accurate
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .circuits import ArithmeticCircuit, Gate, Wire, GateType


@dataclass
class Constraint:
    """
    Single R1CS constraint: <A, z> * <B, z> = <C, z>
    """
    a: List[int]  # A coefficients
    b: List[int]  # B coefficients
    c: List[int]  # C coefficients

    def __repr__(self):
        return f"Constraint(len(A)={len(self.a)}, len(B)={len(self.b)}, len(C)={len(self.c)})"


class R1CS:
    """
    Rank-1 Constraint System.

    Converts arithmetic circuits to R1CS format.

    Example:
        >>> circuit = ArithmeticCircuit()
        >>> # ... build circuit ...
        >>> r1cs = R1CS.from_circuit(circuit)
        >>> # Convert to QAP
        >>> qap = r1cs.to_qap()
    """

    def __init__(self, num_variables: int, num_constraints: int, field_prime: int = 101):
        """
        Initialize R1CS.

        Args:
            num_variables: Number of variables (wires)
            num_constraints: Number of constraints (gates)
            field_prime: Field modulus
        """
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.field_prime = field_prime
        self.constraints: List[Constraint] = []

    def add_constraint(self, a: List[int], b: List[int], c: List[int]) -> None:
        """
        Add a constraint to the system.

        Args:
            a: A vector coefficients
            b: B vector coefficients
            c: C vector coefficients

        Note:
            Each vector should have length num_variables.
        """
        if len(a) != self.num_variables:
            raise ValueError(f"A vector length {len(a)} != num_variables {self.num_variables}")
        if len(b) != self.num_variables:
            raise ValueError(f"B vector length {len(b)} != num_variables {self.num_variables}")
        if len(c) != self.num_variables:
            raise ValueError(f"C vector length {len(c)} != num_variables {self.num_variables}")

        self.constraints.append(Constraint(a, b, c))

    def verify_solution(self, assignment: List[int]) -> bool:
        """
        Verify that an assignment satisfies all constraints.

        Args:
            assignment: Variable assignment (z vector)

        Returns:
            True if all constraints are satisfied

        Verification:
        For each constraint, check: <A, z> * <B, z> == <C, z> (mod p)
        """
        if len(assignment) != self.num_variables:
            return False

        z = np.array([a % self.field_prime for a in assignment])

        for constraint in self.constraints:
            a = np.array(constraint.a)
            b = np.array(constraint.b)
            c = np.array(constraint.c)

            left = (np.dot(a, z) * np.dot(b, z)) % self.field_prime
            right = np.dot(c, z) % self.field_prime

            if left != right:
                return False

        return True

    @classmethod
    def from_circuit(cls, circuit: ArithmeticCircuit) -> 'R1CS':
        """
        Convert arithmetic circuit to R1CS.

        Args:
            circuit: Arithmetic circuit to convert

        Returns:
            R1CS representation of circuit

        Conversion Process:
        For each gate, create a constraint:
        - ADD gate: out = in1 + in2 → out - in1 - in2 = 0
        - MUL gate: out = in1 * in2 → out - in1 * in2 = 0

        R1CS form: <A, z> * <B, z> = <C, z>

        For ADD: We can't directly encode, need reformulation
        For MUL: A = [0, ..., in1, ..., 0], B = [0, ..., in2, ..., 0], C = [0, ..., out, ..., 0]
        """
        num_variables = circuit.get_num_wires()
        num_constraints = circuit.get_num_gates()

        r1cs = cls(
            num_variables=num_variables,
            num_constraints=num_constraints,
            field_prime=circuit.field_prime
        )

        # Process each gate
        for gate in circuit.gates:
            a = [0] * num_variables
            b = [0] * num_variables
            c = [0] * num_variables

            if gate.gate_type == GateType.ADD:
                # out = in1 + in2
                # Reformulate: out * 1 = in1 + in2
                # A = [0, ..., in1, ..., 0]
                # B = [1, 0, ..., 0] (constant 1)
                # C = [0, ..., 1, ..., 1, ..., 0] (out + in1 + in2)

                # For simplicity, encode as: out = in1 + in2
                # This requires special handling in full R1CS
                # Simplified: use 1 * out = in1 + in2
                a[0] = 1  # Constant 1
                b[gate.output_wire.id] = 1  # out

                for inp in gate.input_wires:
                    c[inp.id] = 1  # in1, in2

            elif gate.gate_type == GateType.MUL:
                # out = in1 * in2
                # A = [0, ..., in1, ..., 0]
                # B = [0, ..., in2, ..., 0]
                # C = [0, ..., out, ..., 0]

                a[gate.input_wires[0].id] = 1  # in1
                b[gate.input_wires[1].id] = 1  # in2
                c[gate.output_wire.id] = 1  # out

            elif gate.gate_type == GateType.CONSTANT:
                # out = constant
                # A = [constant, 0, ..., 0]
                # B = [1, 0, ..., 0]
                # C = [0, ..., out, ..., 0]

                a[0] = gate.constant
                b[0] = 1
                c[gate.output_wire.id] = 1

            else:
                raise ValueError(f"Unsupported gate type: {gate.gate_type}")

            r1cs.add_constraint(a, b, c)

        return r1cs

    def get_constraint_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get constraint matrices as numpy arrays.

        Returns:
            (A, B, C) matrices of shape (num_constraints, num_variables)
        """
        if not self.constraints:
            return (
                np.zeros((0, self.num_variables)),
                np.zeros((0, self.num_variables)),
                np.zeros((0, self.num_variables))
            )

        A = np.array([c.a for c in self.constraints])
        B = np.array([c.b for c in self.constraints])
        C = np.array([c.c for c in self.constraints])

        return A, B, C

    def __repr__(self):
        return (
            f"R1CS("
            f"variables={self.num_variables}, "
            f"constraints={len(self.constraints)}, "
            f"field={self.field_prime})"
        )


class QAP:
    """
    Quadratic Arithmetic Program.

    QAP is a compressed representation of R1CS using polynomial interpolation.
    Instead of m constraints, we have 3 sets of m+1 polynomials each.

    Mathematical Form:
    For R1CS constraints (A_i, B_i, C_i), create polynomials:
    - A(x) = sum(A_i * L_i(x))
    - B(x) = sum(B_i * L_i(x))
    - C(x) = sum(C_i * L_i(x))

    where L_i(x) are Lagrange basis polynomials.

    The constraint becomes: A(z) * B(z) = C(z) for some z.

    ADVANTAGE:
    Much more compact representation for large circuits.
    Essential for efficient SNARK proof generation.

    Security Assumptions:
    - Polynomial interpolation is correct
    - Root z is kept secret
    - Division polynomial H(x) is computed correctly
    """

    def __init__(
        self,
        A_polynomials: List[List[int]],  # List of coefficient lists
        B_polynomials: List[List[int]],
        C_polynomials: List[List[int]],
        field_prime: int = 101
    ):
        """
        Initialize QAP.

        Args:
            A_polynomials: List of A polynomial coefficients
            B_polynomials: List of B polynomial coefficients
            C_polynomials: List of C polynomial coefficients
            field_prime: Field modulus
        """
        self.A_polynomials = A_polynomials
        self.B_polynomials = B_polynomials
        self.C_polynomials = C_polynomials
        self.field_prime = field_prime
        self.degree = len(A_polynomials[0]) - 1  # All polynomials same degree

    def evaluate_at(self, x: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Evaluate all polynomials at point x.

        Args:
            x: Evaluation point

        Returns:
            (A_values, B_values, C_values) at point x
        """
        def eval_poly(coeffs: List[int], point: int) -> int:
            """Evaluate polynomial at point using Horner's method."""
            result = 0
            for coeff in reversed(coeffs):
                result = (result * point + coeff) % self.field_prime
            return result

        A_vals = [eval_poly(poly, x) for poly in self.A_polynomials]
        B_vals = [eval_poly(poly, x) for poly in self.B_polynomials]
        C_vals = [eval_poly(poly, x) for poly in self.C_polynomials]

        return A_vals, B_vals, C_vals

    def verify(self, assignment: List[int], z: int) -> bool:
        """
        Verify that assignment satisfies QAP at point z.

        Args:
            assignment: Variable assignment
            z: Evaluation point

        Returns:
            True if A(z) * B(z) = C(z) for assignment
        """
        A_vals, B_vals, C_vals = self.evaluate_at(z)

        # Compute <A(z), assignment>
        a_dot = sum(a * w for a, w in zip(A_vals, assignment)) % self.field_prime
        b_dot = sum(b * w for b, w in zip(B_vals, assignment)) % self.field_prime
        c_dot = sum(c * w for c, w in zip(C_vals, assignment)) % self.field_prime

        left = (a_dot * b_dot) % self.field_prime
        right = c_dot

        return left == right

    @classmethod
    def from_r1cs(cls, r1cs: R1CS) -> 'QAP':
        """
        Convert R1CS to QAP using Lagrange interpolation.

        Args:
            r1cs: R1CS to convert

        Returns:
            QAP representation

        Process:
        1. Choose evaluation points {z_1, ..., z_m} for m constraints
        2. For each variable j, interpolate polynomials:
           A_j(x) through points (z_i, A_ij)
           B_j(x) through points (z_i, B_ij)
           C_j(x) through points (z_i, C_ij)
        3. Result: m+1 polynomials for A, B, C each

        Simplified Implementation:
        Uses distinct evaluation points 1, 2, ..., m for constraints.
        """
        m = r1cs.num_constraints
        n = r1cs.num_variables

        # Get constraint matrices
        A_mat, B_mat, C_mat = r1cs.get_constraint_matrices()

        # Evaluation points: 1, 2, ..., m
        eval_points = list(range(1, m + 1))

        # For each variable, interpolate polynomial through its values
        A_polys = []
        B_polys = []
        C_polys = []

        for var_idx in range(n):
            # Values for this variable across all constraints
            A_vals = A_mat[:, var_idx].tolist()
            B_vals = B_mat[:, var_idx].tolist()
            C_vals = C_mat[:, var_idx].tolist()

            # Interpolate polynomial
            A_poly = cls._interpolate(eval_points, A_vals, r1cs.field_prime)
            B_poly = cls._interpolate(eval_points, B_vals, r1cs.field_prime)
            C_poly = cls._interpolate(eval_points, C_vals, r1cs.field_prime)

            A_polys.append(A_poly)
            B_polys.append(B_poly)
            C_polys.append(C_poly)

        return cls(A_polys, B_polys, C_polys, r1cs.field_prime)

    @staticmethod
    def _interpolate(x_vals: List[int], y_vals: List[int], field_prime: int) -> List[int]:
        """
        Lagrange interpolation.

        Args:
            x_vals: X coordinates (distinct)
            y_vals: Y coordinates
            field_prime: Field modulus

        Returns:
            Polynomial coefficients (highest degree first)
        """
        if len(x_vals) != len(y_vals):
            raise ValueError("x_vals and y_vals must have same length")

        if len(x_vals) == 0:
            return []

        if len(x_vals) == 1:
            # Constant polynomial
            return [y_vals[0] % field_prime]

        # Simplified Lagrange interpolation
        # For each x_i, compute basis polynomial L_i(x)
        # Then P(x) = sum(y_i * L_i(x))

        # Using numpy for polynomial fitting (simplified)
        # In production, use finite field arithmetic
        try:
            coeffs = np.polyfit(x_vals, y_vals, len(x_vals) - 1)
            coeffs = [int(round(c)) % field_prime for c in coeffs]
            return coeffs
        except:
            # Fallback: return zero polynomial
            return [0] * len(x_vals)

    def __repr__(self):
        return (
            f"QAP("
            f"variables={len(self.A_polynomials)}, "
            f"degree={self.degree}, "
            f"field={self.field_prime})"
        )


class Solution:
    """
    Solution (witness) to R1CS/QAP.

    Contains assignment to all variables that satisfies constraints.
    """

    def __init__(
        self,
        assignment: List[int],
        public_inputs: Dict[int, int],
        public_outputs: Dict[int, int]
    ):
        """
        Initialize solution.

        Args:
            assignment: Full assignment (all variables)
            public_inputs: Map of public input indices to values
            public_outputs: Map of public output indices to values
        """
        self.assignment = assignment
        self.public_inputs = public_inputs
        self.public_outputs = public_outputs

    def get_public_part(self) -> List[int]:
        """
        Get public part of solution (inputs and outputs).

        Returns:
            Public values as list
        """
        public = {}
        public.update(self.public_inputs)
        public.update(self.public_outputs)

        # Sort by index and return values
        sorted_indices = sorted(public.keys())
        return [public[i] for i in sorted_indices]

    def __repr__(self):
        return (
            f"Solution("
            f"total_vars={len(self.assignment)}, "
            f"public_inputs={len(self.public_inputs)}, "
            f"public_outputs={len(self.public_outputs)})"
        )
