"""
Unit tests for Set membership proofs
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from fundamentals.set_membership import MerkleTree, MerkleProof, SetMembershipProof, RSAAccumulator


class TestMerkleTree:
    """Test suite for Merkle tree operations."""

    def test_initialization_empty(self):
        """Test initializing empty Merkle tree."""
        tree = MerkleTree()

        assert tree.get_root() is None
        assert len(tree.leaves) == 0

    def test_initialization_with_data(self):
        """Test initializing Merkle tree with data."""
        data = [b"item1", b"item2", b"item3"]
        tree = MerkleTree(data)

        assert tree.get_root() is not None
        assert len(tree.leaves) == 3

    def test_add_single_leaf(self):
        """Test adding a single leaf."""
        tree = MerkleTree()
        idx = tree.add_leaf(b"test_data")

        assert idx == 0
        assert len(tree.leaves) == 1
        assert tree.get_root() is not None

    def test_add_multiple_leaves(self):
        """Test adding multiple leaves."""
        tree = MerkleTree()

        indices = []
        for i in range(5):
            idx = tree.add_leaf(f"data{i}".encode())
            indices.append(idx)

        assert indices == [0, 1, 2, 3, 4]
        assert len(tree.leaves) == 5

    def test_add_leaves_batch(self):
        """Test adding leaves in batch."""
        tree = MerkleTree()
        data = [f"item{i}".encode() for i in range(10)]

        tree.add_leaves(data)

        assert len(tree.leaves) == 10
        assert tree.get_root() is not None

    def test_root_deterministic(self):
        """Test that root is deterministic for same data."""
        data = [b"item1", b"item2", b"item3"]

        tree1 = MerkleTree(data)
        tree2 = MerkleTree(data)

        assert tree1.get_root() == tree2.get_root()

    def test_root_changes_with_data(self):
        """Test that root changes when data changes."""
        tree1 = MerkleTree([b"item1", b"item2"])
        tree2 = MerkleTree([b"item1", b"item3"])  # Different second item

        assert tree1.get_root() != tree2.get_root()

    def test_get_proof_valid_index(self):
        """Test getting proof for valid index."""
        tree = MerkleTree()
        tree.add_leaf(b"item1")
        tree.add_leaf(b"item2")

        proof = tree.get_proof(0)

        assert isinstance(proof, MerkleProof)
        assert proof.leaf_index == 0
        assert proof.root == tree.get_root()

    def test_get_proof_invalid_index(self):
        """Test getting proof for invalid index."""
        tree = MerkleTree()
        tree.add_leaf(b"item1")

        with pytest.raises(IndexError):
            tree.get_proof(5)

    def test_verify_proof_valid(self):
        """Test verifying valid Merkle proof."""
        tree = MerkleTree()
        data = b"test_data"
        idx = tree.add_leaf(data)

        proof = tree.get_proof(idx)

        assert tree.verify_proof(proof, data)

    def test_verify_proof_invalid_data(self):
        """Test verifying proof with wrong data."""
        tree = MerkleTree()
        data1 = b"item1"
        data2 = b"item2"

        idx = tree.add_leaf(data1)
        proof = tree.get_proof(idx)

        # Try to verify with different data
        assert not tree.verify_proof(proof, data2)

    def test_verify_proof_wrong_index(self):
        """Test that proof doesn't verify for wrong index."""
        tree = MerkleTree()
        data = [b"item1", b"item2", b"item3"]

        for d in data:
            tree.add_leaf(d)

        proof = tree.get_proof(0)  # Proof for item1

        # Try to verify item2 with proof for item1
        assert not tree.verify_proof(proof, data[1])

    def test_merkle_proof_for_all_leaves(self):
        """Test generating and verifying proofs for all leaves."""
        data = [f"item{i}".encode() for i in range(10)]
        tree = MerkleTree(data)

        for i, item in enumerate(data):
            proof = tree.get_proof(i)
            assert tree.verify_proof(proof, item)

    def test_json_serialization(self):
        """Test serializing and deserializing tree."""
        data = [b"item1", b"item2", b"item3"]
        tree1 = MerkleTree(data)

        json_str = tree1.to_json()
        tree2 = MerkleTree.from_json(json_str)

        assert tree1.get_root() == tree2.get_root()
        assert len(tree1.leaves) == len(tree2.leaves)

    def test_odd_number_of_leaves(self):
        """Test Merkle tree with odd number of leaves."""
        data = [f"item{i}".encode() for i in range(7)]  # Odd number
        tree = MerkleTree(data)

        assert tree.get_root() is not None

        # All leaves should have valid proofs
        for i, item in enumerate(data):
            proof = tree.get_proof(i)
            assert tree.verify_proof(proof, item)

    def test_single_leaf_tree(self):
        """Test Merkle tree with single leaf."""
        tree = MerkleTree()
        tree.add_leaf(b"only_item")

        assert tree.get_root() is not None

        proof = tree.get_proof(0)
        assert tree.verify_proof(proof, b"only_item")

    def test_different_hash_functions(self):
        """Test Merkle tree with different hash functions."""
        data = [b"item1", b"item2"]

        tree_sha256 = MerkleTree(data, hash_function="sha256")
        tree_sha3 = MerkleTree(data, hash_function="sha3_256")

        # Different hash functions should produce different roots
        assert tree_sha256.get_root() != tree_sha3.get_root()


class TestSetMembershipProof:
    """Test suite for set membership proofs."""

    def test_initialization(self):
        """Test set membership proof initialization."""
        tree = MerkleTree([b"item1", b"item2"])
        proof_system = SetMembershipProof(tree)

        assert proof_system.tree is tree

    def test_generate_proof_member(self):
        """Test generating proof for member of set."""
        tree = MerkleTree()
        data = [f"email{i}@domain.com".encode() for i in range(5)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        value = b"email2@domain.com"
        proof = proof_system.generate_proof(value)

        assert 'value' in proof
        assert 'root' in proof
        assert 'leaf_index' in proof

    def test_generate_proof_non_member_fails(self):
        """Test that proof generation fails for non-member."""
        tree = MerkleTree()
        data = [f"email{i}@domain.com".encode() for i in range(5)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        value = b"unknown@domain.com"

        with pytest.raises(ValueError):
            proof_system.generate_proof(value)

    def test_verify_valid_proof(self):
        """Test verifying valid membership proof."""
        tree = MerkleTree()
        data = [f"email{i}@domain.com".encode() for i in range(5)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        value = b"email2@domain.com"
        proof = proof_system.generate_proof(value)

        assert proof_system.verify(proof)

    def test_verify_invalid_proof(self):
        """Test that verification fails for invalid proof."""
        tree = MerkleTree()
        data = [f"email{i}@domain.com".encode() for i in range(5)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        # Generate valid proof
        value = b"email2@domain.com"
        proof = proof_system.generate_proof(value)

        # Tamper with proof
        proof['value'] = "wrong@domain.com"

        assert not proof_system.verify(proof)

    def test_generate_zk_proof(self):
        """Test generating zero-knowledge membership proof."""
        tree = MerkleTree()
        data = [f"email{i}@domain.com".encode() for i in range(5)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        value = b"email2@domain.com"

        # Create commitment
        from fundamentals.commitments import PedersenCommitment
        scheme = PedersenCommitment()
        commitment, r = scheme.commit(123)  # Commit to value ID

        proof = proof_system.generate_zk_proof(value, commitment, r)

        assert 'commitment' in proof
        assert 'merkle_proof' in proof

    def test_batch_membership_proofs(self):
        """Test generating membership proofs for multiple items."""
        tree = MerkleTree()
        data = [f"item{i}".encode() for i in range(100)]
        for item in data:
            tree.add_leaf(item)

        proof_system = SetMembershipProof(tree)

        # Generate proofs for random subset
        import random
        test_indices = random.sample(range(100), 10)

        for idx in test_indices:
            value = data[idx]
            proof = proof_system.generate_proof(value)
            assert proof_system.verify(proof)


class TestRSAAccumulator:
    """Test suite for RSA accumulator."""

    def test_initialization(self):
        """Test RSA accumulator initialization."""
        acc = RSAAccumulator()

        assert acc.n > 0
        assert acc.g > 0
        assert len(acc.elements) == 0

    def test_add_single_element(self):
        """Test adding single element."""
        acc = RSAAccumulator()
        acc.add_element(42)

        assert 42 in acc.elements
        assert acc.get_accumulator_value() > 0

    def test_add_multiple_elements(self):
        """Test adding multiple elements."""
        acc = RSAAccumulator()

        elements = [2, 3, 5, 7, 11]  # Primes
        for elem in elements:
            acc.add_element(elem)

        assert len(acc.elements) == 5

    def test_get_witness(self):
        """Test getting witness for element."""
        acc = RSAAccumulator()
        acc.add_element(42)

        witness = acc.get_witness(42)

        assert witness > 0
        assert witness < acc.n

    def test_get_witness_non_member_fails(self):
        """Test that getting witness fails for non-member."""
        acc = RSAAccumulator()
        acc.add_element(42)

        with pytest.raises(ValueError):
            acc.get_witness(100)

    def test_verify_member(self):
        """Test verifying membership with witness."""
        acc = RSAAccumulator()

        elements = [2, 3, 5, 7, 11]
        for elem in elements:
            acc.add_element(elem)

        # Get witness for element 5
        witness = acc.get_witness(5)

        # Verify
        assert acc.verify_member(5, witness)

    def test_verify_non_member_fails(self):
        """Test that verification fails for non-member."""
        acc = RSAAccumulator()

        elements = [2, 3, 5, 7, 11]
        for elem in elements:
            acc.add_element(elem)

        # Get witness for element 5
        witness = acc.get_witness(5)

        # Try to verify different element
        assert not acc.verify_member(7, witness)

    def test_accumulator_value_changes(self):
        """Test that accumulator value changes with additions."""
        acc = RSAAccumulator()

        value1 = acc.get_accumulator_value()
        acc.add_element(42)
        value2 = acc.get_accumulator_value()

        assert value1 != value2

    def test_all_members_verifiable(self):
        """Test that all members can be verified."""
        acc = RSAAccumulator()

        elements = [2, 3, 5, 7, 11, 13, 17, 19]
        for elem in elements:
            acc.add_element(elem)

        # All members should be verifiable
        for elem in elements:
            witness = acc.get_witness(elem)
            assert acc.verify_member(elem, witness)


def test_comprehensive_set_membership():
    """Comprehensive test of set membership proofs."""
    # Test Merkle tree with realistic data
    tree = MerkleTree()

    # Simulate authorized email dataset
    authorized_emails = [
        f"user{i}@trusted-domain.com".encode()
        for i in range(1000)
    ]

    tree.add_leaves(authorized_emails)

    proof_system = SetMembershipProof(tree)

    # Test membership for various emails
    test_cases = [
        ("user0@trusted-domain.com", True),
        ("user500@trusted-domain.com", True),
        ("user999@trusted-domain.com", True),
        ("attacker@malicious.com", False),
        ("unknown@external.com", False),
    ]

    for email, should_be_member in test_cases:
        email_bytes = email.encode()

        if should_be_member:
            proof = proof_system.generate_proof(email_bytes)
            assert proof_system.verify(proof)
        else:
            with pytest.raises(ValueError):
                proof_system.generate_proof(email_bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
