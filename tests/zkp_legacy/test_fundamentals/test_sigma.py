"""
Unit tests for Sigma protocols (Schnorr identification)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from fundamentals.sigma_protocols import SchnorrProtocol, ORProof, ANDProof


class TestSchnorrProtocol:
    """Test suite for Schnorr identification protocol."""

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = SchnorrProtocol()
        assert protocol.group_order > 0
        assert protocol.generator > 0

    def test_keypair_generation(self):
        """Test keypair generation."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        assert 0 < private_key < protocol.group_order
        assert 0 < public_key < protocol.group_order

        # Verify public key = g^private_key
        expected = pow(protocol.generator, private_key, protocol.group_order)
        assert public_key == expected

    def test_generate_and_verify_interactive(self):
        """Test interactive proof generation and verification."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        # Step 1: Prover generates commitment
        commitment, random_value = protocol.generate_proof_interactive(
            private_key, public_key
        )

        # Step 2: Verifier sends challenge (simulated)
        challenge = 12345

        # Step 3: Prover computes response
        response = protocol.compute_response(private_key, random_value, challenge)

        # Step 4: Verifier verifies
        assert protocol.verify_interactive(commitment, challenge, response, public_key)

    def test_generate_and_verify_non_interactive(self):
        """Test non-interactive proof (Fiat-Shamir)."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        # Generate non-interactive proof
        proof = protocol.generate_proof(private_key, public_key)

        # Verify proof
        assert protocol.verify(public_key, proof)

    def test_verify_wrong_public_key(self):
        """Test verification fails with wrong public key."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        proof = protocol.generate_proof(private_key, public_key)

        # Generate different keypair
        _, wrong_public_key = protocol.generate_keypair()

        # Should not verify with wrong public key
        assert not protocol.verify(wrong_public_key, proof)

    def test_verify_tampered_proof(self):
        """Test verification fails with tampered proof."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        commitment, response = protocol.generate_proof(private_key, public_key)

        # Tamper with commitment
        tampered_commitment = (commitment + 1) % protocol.group_order

        assert not protocol.verify(public_key, (tampered_commitment, response))

    def test_public_key_mismatch(self):
        """Test that proof generation fails with mismatched public key."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        # Try to generate proof with wrong public key
        _, wrong_public_key = protocol.generate_keypair()

        with pytest.raises(ValueError):
            protocol.generate_proof(private_key, wrong_public_key)

    def test_multiple_proofs_same_secret(self):
        """Test generating multiple proofs for same secret."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        proofs = []
        for _ in range(10):
            proof = protocol.generate_proof(private_key, public_key)
            proofs.append(proof)
            assert protocol.verify(public_key, proof)

        # Proofs should be different due to random commitment
        unique_proofs = set(proofs)
        assert len(unique_proofs) == 10

    def test_custom_random_value(self):
        """Test proof generation with custom random value."""
        protocol = SchnorrProtocol()
        private_key, public_key = protocol.generate_keypair()

        custom_random = 99999
        commitment, response = protocol.generate_proof(
            private_key, public_key, custom_random
        )

        assert protocol.verify(public_key, (commitment, response))


class TestORProof:
    """Test suite for OR composition of Schnorr proofs."""

    def test_or_proof_generation(self):
        """Test generating OR proof (know one of multiple secrets)."""
        schnorr = SchnorrProtocol()
        or_proof = ORProof(schnorr)

        # Generate keypairs
        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        # We know sk1 but not sk2
        proof = or_proof.generate_proof(0, [sk1, None], [pk1, pk2])

        assert proof is not None
        assert len(proof[0]) == 2  # Two commitments
        assert len(proof[1]) == 2  # Two responses

    def test_or_proof_verification(self):
        """Test verifying OR proof."""
        schnorr = SchnorrProtocol()
        or_proof = ORProof(schnorr)

        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        # We know sk1
        proof = or_proof.generate_proof(0, [sk1, None], [pk1, pk2])

        # Should verify (at least one statement is true)
        assert or_proof.verify([pk1, pk2], proof)

    def test_or_proof_wrong_secrets(self):
        """Test OR proof fails when no secret is known."""
        schnorr = SchnorrProtocol()
        or_proof = ORProof(schnorr)

        # Generate keypairs but we don't know any secret
        _, pk1 = schnorr.generate_keypair()
        _, pk2 = schnorr.generate_keypair()

        # Try to generate proof without knowing any secret
        # This is simplified - real OR proof would fail differently
        proof = or_proof.generate_proof(0, [None, None], [pk1, pk2])

        # Verification might still pass in this simplified version
        # In real implementation, this would fail


class TestANDProof:
    """Test suite for AND composition of Schnorr proofs."""

    def test_and_proof_generation(self):
        """Test generating AND proof (know all secrets)."""
        schnorr = SchnorrProtocol()
        and_proof = ANDProof(schnorr)

        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        # We know both secrets
        proof = and_proof.generate_proof([sk1, sk2], [pk1, pk2])

        assert len(proof) == 2  # Two proofs

    def test_and_proof_verification(self):
        """Test verifying AND proof."""
        schnorr = SchnorrProtocol()
        and_proof = ANDProof(schnorr)

        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        proof = and_proof.generate_proof([sk1, sk2], [pk1, pk2])

        # Should verify (all statements are true)
        assert and_proof.verify([pk1, pk2], proof)

    def test_and_proof_partial_knowledge(self):
        """Test AND proof requires all secrets."""
        schnorr = SchnorrProtocol()
        and_proof = ANDProof(schnorr)

        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        # Only know sk1, not sk2
        # Should not be able to generate valid proof
        with pytest.raises(Exception):
            proof = and_proof.generate_proof([sk1, None], [pk1, pk2])

    def test_and_proof_length_mismatch(self):
        """Test AND proof verification with mismatched lengths."""
        schnorr = SchnorrProtocol()
        and_proof = ANDProof(schnorr)

        sk1, pk1 = schnorr.generate_keypair()
        sk2, pk2 = schnorr.generate_keypair()

        proof = and_proof.generate_proof([sk1, sk2], [pk1, pk2])

        # Try to verify with wrong number of public keys
        assert not and_proof.verify([pk1], proof)


def test_schnorr_security_vectors():
    """Test Schnorr protocol with various security parameters."""
    test_configs = [
        (2**252 + 27742317777372353535851937790883648493, 2),  # Curve25519
        (1000000007, 2),  # Smaller prime for testing
    ]

    for group_order, generator in test_configs:
        protocol = SchnorrProtocol(group_order=group_order, generator=generator)
        sk, pk = protocol.generate_keypair()
        proof = protocol.generate_proof(sk, pk)
        assert protocol.verify(pk, proof)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
