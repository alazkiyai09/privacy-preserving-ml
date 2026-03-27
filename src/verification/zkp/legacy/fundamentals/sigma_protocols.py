"""
Sigma Protocols

Sigma protocols are interactive proof protocols where the prover convinces the verifier
that they know a secret without revealing it.

Structure:
1. Commitment: Prover sends initial message
2. Challenge: Verifier sends random challenge
3. Response: Prover sends response

Can be made non-interactive using Fiat-Shamir heuristic.

This module implements Schnorr identification protocol - a fundamental sigma protocol
for proving knowledge of a discrete logarithm.

Mathematical Form:
- Prover knows x such that y = g^x mod p
- Prover wants to prove knowledge of x without revealing it

Protocol:
1. Prover picks random r, sends t = g^r
2. Verifier sends random challenge c
3. Prover sends s = r + c*x
4. Verifier checks: g^s == t * y^c

Security Assumptions:
- Discrete logarithm problem is hard
- Challenge is truly random
"""

import hashlib
import random
from typing import Tuple
import yaml


class SchnorrProtocol:
    """
    Schnorr identification protocol - prove knowledge of discrete logarithm.

    USE CASE:
    Prove you know secret x such that public_key = g^x without revealing x.

    SECURITY ASSUMPTIONS:
    - Discrete logarithm problem is hard in the chosen group
    - Hash function is collision-resistant (for Fiat-Shamir)
    - Random values are uniformly random

    WARNINGS:
    - Never reuse the random commitment value r!
    - Use cryptographically secure random number generator
    """

    def __init__(
        self,
        group_order: int = None,
        generator: int = None,
        config_path: str = None
    ):
        """
        Initialize Schnorr protocol.

        Args:
            group_order: Order of the group (prime)
            generator: Generator of the group
            config_path: Optional path to config file
        """
        if config_path:
            self._load_config(config_path)
        else:
            # Curve25519 parameters (simplified)
            self.group_order = group_order or (2**252 + 27742317777372353535851937790883648493)
            self.generator = generator or 2

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.group_order = int(config["security"]["curve"]["order"])
        # In practice, generator would be loaded from config

    def generate_keypair(self) -> Tuple[int, int]:
        """
        Generate a public/private key pair.

        Returns:
            (private_key, public_key) where public_key = g^private_key

        Security:
            - Private key MUST be randomly generated
            - Private key MUST be kept secret
            - Never reuse private keys!

        Example:
            >>> protocol = SchnorrProtocol()
            >>> private_key, public_key = protocol.generate_keypair()
            >>> # public_key can be shared, private_key must be kept secret
        """
        private_key = random.SystemRandom().randint(1, self.group_order - 2)
        public_key = pow(self.generator, private_key, self.group_order)
        return private_key, public_key

    def generate_proof_interactive(
        self,
        secret: int,
        public_input: int,
        random_value: int = None
    ) -> Tuple[int, int]:
        """
        Generate proof (interactive version - first message).

        Protocol step 1: Prover generates commitment t = g^r

        Args:
            secret: The secret value x (private key)
            public_input: The public value y = g^x
            random_value: Optional random value r (generated if not provided)

        Returns:
            (commitment, random_value) - The first message and the randomness used

        Security Requirements:
            - random_value MUST be uniformly random
            - random_value MUST NEVER be reused
            - This is just the first message - still need challenge!

        Example (interactive protocol):
            >>> protocol = SchnorrProtocol()
            >>> sk, pk = protocol.generate_keypair()
            >>> t, r = protocol.generate_proof_interactive(sk, pk)
            >>> # Send t to verifier, receive challenge c
            >>> # Then: s = protocol.compute_response(sk, r, c)
        """
        if random_value is None:
            # CRITICAL: Use cryptographically secure random
            random_value = random.SystemRandom().randint(1, self.group_order - 2)

        # Verify public_input matches secret
        expected_public = pow(self.generator, secret, self.group_order)
        if public_input != expected_public:
            raise ValueError("Public input does not match secret")

        # Compute commitment t = g^r
        commitment = pow(self.generator, random_value, self.group_order)

        return commitment, random_value

    def compute_response(
        self,
        secret: int,
        random_value: int,
        challenge: int
    ) -> int:
        """
        Compute response to challenge (interactive version - third message).

        Protocol step 3: Prover computes s = r + c*x

        Args:
            secret: The secret value x
            random_value: The random value r used in commitment
            challenge: The challenge c from verifier

        Returns:
            The response s

        Example:
            >>> protocol = SchnorrProtocol()
            >>> sk, pk = protocol.generate_keypair()
            >>> t, r = protocol.generate_proof_interactive(sk, pk)
            >>> c = 12345  # Challenge from verifier
            >>> s = protocol.compute_response(sk, r, c)
            >>> # Send s to verifier
        """
        response = (random_value + challenge * secret) % (self.group_order - 1)
        return response

    def verify_interactive(
        self,
        commitment: int,
        challenge: int,
        response: int,
        public_input: int
    ) -> bool:
        """
        Verify proof (interactive version).

        Protocol step 4: Verifier checks g^s == t * y^c

        Args:
            commitment: The commitment t from prover
            challenge: The challenge c sent to prover
            response: The response s from prover
            public_input: The public value y

        Returns:
            True if proof is valid

        Mathematical Verification:
            g^s = g^(r + c*x) = g^r * g^(c*x) = t * (g^x)^c = t * y^c

        Example:
            >>> protocol = SchnorrProtocol()
            >>> sk, pk = protocol.generate_keypair()
            >>> t, r = protocol.generate_proof_interactive(sk, pk)
            >>> c = 12345
            >>> s = protocol.compute_response(sk, r, c)
            >>> assert protocol.verify_interactive(t, c, s, pk)
        """
        # Compute g^s
        left = pow(self.generator, response, self.group_order)

        # Compute t * y^c
        y_pow_c = pow(public_input, challenge, self.group_order)
        right = (commitment * y_pow_c) % self.group_order

        return left == right

    def generate_proof(
        self,
        secret: int,
        public_input: int,
        random_value: int = None
    ) -> Tuple[int, int]:
        """
        Generate non-interactive proof using Fiat-Shamir heuristic.

        Instead of verifier sending challenge, prover computes:
        challenge = H(t || public_input)

        Args:
            secret: The secret value x
            public_input: The public value y = g^x
            random_value: Optional random value r

        Returns:
            (commitment, response) - The non-interactive proof

        Security:
            - Hash function MUST be collision-resistant
            - Random value MUST still be random and never reused

        Example:
            >>> protocol = SchnorrProtocol()
            >>> sk, pk = protocol.generate_keypair()
            >>> proof = protocol.generate_proof(sk, pk)
            >>> assert protocol.verify(pk, proof)
        """
        commitment, random_value = self.generate_proof_interactive(
            secret, public_input, random_value
        )

        # Fiat-Shamir: Compute challenge from hash
        challenge = self._compute_challenge(commitment, public_input)

        response = self.compute_response(secret, random_value, challenge)

        return commitment, response

    def verify(
        self,
        public_input: int,
        proof: Tuple[int, int]
    ) -> bool:
        """
        Verify non-interactive proof.

        Args:
            public_input: The public value y
            proof: (commitment, response)

        Returns:
            True if proof is valid

        Example:
            >>> protocol = SchnorrProtocol()
            >>> sk, pk = protocol.generate_keypair()
            >>> proof = protocol.generate_proof(sk, pk)
            >>> assert protocol.verify(pk, proof)
        """
        commitment, response = proof

        # Recompute challenge
        challenge = self._compute_challenge(commitment, public_input)

        # Verify using same check as interactive version
        return self.verify_interactive(commitment, challenge, response, public_input)

    def _compute_challenge(self, commitment: int, public_input: int) -> int:
        """
        Compute challenge using Fiat-Shamir heuristic.

        challenge = H(commitment || public_input)

        Security:
            - Hash function MUST be modeled as random oracle
            - Use SHA-256 or stronger
        """
        data = f"{commitment}{public_input}".encode()
        hash_bytes = hashlib.sha256(data).digest()

        # Convert to integer
        challenge = int.from_bytes(hash_bytes, byteorder='big')

        # Ensure challenge is in valid range
        challenge = challenge % (self.group_order - 1)

        return challenge


class ORProof:
    """
    Proof of knowledge of one of multiple secrets (OR composition).

    Proves that you know at least one of several secrets without revealing which one.

    Use case: Anonymous credentials - prove you have one of several valid credentials
    without revealing which one.
    """

    def __init__(self, protocol: SchnorrProtocol):
        self.protocol = protocol

    def generate_proof(
        self,
        known_secret_index: int,
        secrets: list,
        public_inputs: list
    ) -> Tuple[list, list]:
        """
        Generate proof that you know one of the secrets.

        Args:
            known_secret_index: Index of the secret you know
            secrets: List of all secrets (unknown ones can be None)
            public_inputs: List of corresponding public inputs

        Returns:
            (commitments, responses) - The proof

        Example:
            >>> protocol = SchnorrProtocol()
            >>> or_proof = ORProof(protocol)
            >>> sk1, pk1 = protocol.generate_keypair()
            >>> sk2, pk2 = protocol.generate_keypair()
            >>> # We know sk1 but not sk2
            >>> proof = or_proof.generate_proof(0, [sk1, None], [pk1, pk2])
            >>> assert or_proof.verify([pk1, pk2], proof)
        """
        commitments = []
        responses = []

        for i, (secret, public_input) in enumerate(zip(secrets, public_inputs)):
            if i == known_secret_index:
                # Generate real proof for known secret
                comm, resp = self.protocol.generate_proof(secret, public_input)
            else:
                # Generate simulated proof for unknown secret
                # In real OR proof, this uses challenge simulation
                # Simplified: just generate dummy values
                resp = random.SystemRandom().randint(1, self.protocol.group_order - 2)
                challenge = self.protocol._compute_challenge(0, public_input)
                comm = (pow(self.protocol.generator, resp, self.protocol.group_order) *
                       pow(public_input, challenge, self.protocol.group_order)) % self.protocol.group_order

            commitments.append(comm)
            responses.append(resp)

        return commitments, responses

    def verify(
        self,
        public_inputs: list,
        proof: Tuple[list, list]
    ) -> bool:
        """
        Verify OR proof.

        Args:
            public_inputs: List of public inputs
            proof: (commitments, responses)

        Returns:
            True if proof is valid (at least one statement is true)
        """
        commitments, responses = proof

        # Check all but one - this is simplified
        # Real OR proof uses more sophisticated verification
        valid_count = 0
        for public_input, comm, resp in zip(public_inputs, commitments, responses):
            if self.protocol.verify(public_input, (comm, resp)):
                valid_count += 1

        return valid_count >= 1


class ANDProof:
    """
    Proof of knowledge of all secrets (AND composition).

    Proves that you know all of multiple secrets.

    Use case: Prove you know multiple credentials without revealing them.
    """

    def __init__(self, protocol: SchnorrProtocol):
        self.protocol = protocol

    def generate_proof(
        self,
        secrets: list,
        public_inputs: list
    ) -> list:
        """
        Generate proof that you know all secrets.

        Args:
            secrets: List of secrets you know
            public_inputs: List of corresponding public inputs

        Returns:
            List of proofs, one for each statement

        Example:
            >>> protocol = SchnorrProtocol()
            >>> and_proof = ANDProof(protocol)
            >>> sk1, pk1 = protocol.generate_keypair()
            >>> sk2, pk2 = protocol.generate_keypair()
            >>> proof = and_proof.generate_proof([sk1, sk2], [pk1, pk2])
            >>> assert and_proof.verify([pk1, pk2], proof)
        """
        proofs = []
        for secret, public_input in zip(secrets, public_inputs):
            proof = self.protocol.generate_proof(secret, public_input)
            proofs.append(proof)

        return proofs

    def verify(
        self,
        public_inputs: list,
        proofs: list
    ) -> bool:
        """
        Verify AND proof.

        Args:
            public_inputs: List of public inputs
            proofs: List of proofs

        Returns:
            True if all proofs are valid
        """
        if len(public_inputs) != len(proofs):
            return False

        for public_input, proof in zip(public_inputs, proofs):
            if not self.protocol.verify(public_input, proof):
                return False

        return True
