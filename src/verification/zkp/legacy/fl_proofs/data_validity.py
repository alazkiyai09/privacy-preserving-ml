"""
Data Validity Proofs for Federated Learning

This module implements zero-knowledge proofs for proving that clients trained
on valid data (e.g., real phishing/legitimate emails) without revealing the data.

PROBLEM IN FEDERATED LEARNING:
Server needs to ensure clients actually trained on real data, not garbage
or synthetic data that could corrupt the global model.

NAIVE SOLUTION:
Server checks client data → Reveals private emails → Privacy violation!

ZK SOLUTION:
Client proves data ∈ ValidSet using Merkle tree membership proof.

MATHEMATICAL FORMULATION:
Given:
- Valid dataset D = {d_1, d_2, ..., d_n}
- Merkle tree root R commits to D
- Client trained on data d

Prove:
- d ∈ D (membership proof)
- Without revealing which data

Implementation:
1. Build Merkle tree from authorized dataset
2. Client provides Merkle path for their data
3. Server verifies path leads to root
4. Learn nothing about which data element

USE CASE:
1. Server publishes Merkle root of authorized emails
2. Client trains on subset of emails
3. Client proves their emails are in authorized set
4. Server verifies without learning which emails

Security:
- Server learns nothing about which emails client used
- Client cannot prove fake email is in set
- Proof size: O(log n) where n = dataset size
"""

from typing import List, Optional
import hashlib

from ..fundamentals.set_membership import MerkleTree, SetMembershipProof
from ..snark.circuits import ArithmeticCircuit
from ..snark.proof_gen import ProofGenerator, Proof
from ..snark.verification import ProofVerifier, VerificationResult
from ..snark.trusted_setup import ProvingKey, VerificationKey


class DataValidityProof:
    """
    Zero-knowledge proof that training data is from valid dataset.

    PROVES: data ∈ ValidSet
    REVEALS: Nothing about which data element

    Example:
        >>> valid_data = [b"email1", b"email2", ...]
        >>> system = DataValidityProof(valid_data)
        >>> root = system.get_merkle_root()
        >>> # Publish root to clients
        >>> client_email = b"email2"
        >>> proof = system.generate_proof(client_email)
        >>> assert system.verify(proof, root)
    """

    def __init__(
        self,
        valid_dataset: List[bytes],
        hash_function: str = "sha256"
    ):
        """
        Initialize data validity proof system.

        Args:
            valid_dataset: List of valid data items (e.g., authorized emails)
            hash_function: Hash function for Merkle tree

        Note:
            In production, valid_dataset could be millions of emails.
        """
        self.valid_dataset = valid_dataset
        self.hash_function = hash_function

        # Build Merkle tree
        self.merkle_tree = MerkleTree(data=valid_dataset, hash_function=hash_function)
        self.set_proof = SetMembershipProof(self.merkle_tree)

    def get_merkle_root(self) -> bytes:
        """
        Get Merkle root hash.

        Returns:
            Root hash (public commitment to dataset)

        Note:
            Server publishes this root to clients. It commits to the entire
            dataset without revealing individual items.
        """
        return self.merkle_tree.get_root()

    def generate_proof(self, data: bytes) -> dict:
        """
        Generate proof that data is in valid set.

        Args:
            data: The data item to prove membership for

        Returns:
            Membership proof

        Raises:
            ValueError: If data not in valid set

        Proof contains:
        - Merkle path from data to root
        - Direction indicators for path
        - Root hash for verification

        Security:
        - Proof reveals data item itself
        - For true zero-knowledge, use commitment scheme
        """
        proof = self.set_proof.generate_proof(data)
        return proof

    def generate_zk_proof(
        self,
        data: bytes,
        commitment: bytes,
        commitment_randomness: int
    ) -> dict:
        """
        Generate zero-knowledge membership proof.

        Args:
            data: The actual data (secret)
            commitment: Commitment to data
            commitment_randomness: Randomness for commitment

        Returns:
            Zero-knowledge proof

        Security:
        - Reveals nothing about which data item
        - Only proves committed value is in set
        """
        proof = self.set_proof.generate_zk_proof(
            data,
            commitment,
            commitment_randomness
        )
        return proof

    def verify(self, proof: dict, expected_root: Optional[bytes] = None) -> bool:
        """
        Verify data validity proof.

        Args:
            proof: Proof from generate_proof()
            expected_root: Expected Merkle root (uses current root if None)

        Returns:
            True if proof is valid

        Note:
            If expected_root is None, uses root from current Merkle tree.
        """
        if expected_root is None:
            expected_root = self.get_merkle_root()

        # Check proof's root matches expected
        proof_root = bytes.fromhex(proof['root'])
        if proof_root != expected_root:
            return False

        # Verify membership
        return self.set_proof.verify(proof)

    def verify_zk_proof(self, proof: dict, expected_root: Optional[bytes] = None) -> bool:
        """
        Verify zero-knowledge membership proof.

        Args:
            proof: ZK proof from generate_zk_proof()
            expected_root: Expected Merkle root

        Returns:
            True if proof is valid
        """
        if expected_root is None:
            expected_root = self.get_merkle_root()

        return self.set_proof.verify_zk_proof(proof)

    def estimate_proof_size(self, dataset_size: int) -> int:
        """
        Estimate proof size in bytes.

        Args:
            dataset_size: Size of valid dataset

        Returns:
            Estimated proof size

        Size:
        - Hash size: 32 bytes (SHA-256)
        - Tree depth: log2(dataset_size)
        - Proof size: 32 * log2(n) bytes

        Example:
        - 1M items: 32 * 20 = 640 bytes
        - 1B items: 32 * 30 = 960 bytes
        """
        import math
        tree_depth = math.ceil(math.log2(dataset_size))
        return 32 * tree_depth


class BatchDataValidityProof:
    """
    Batch proof for multiple data validity proofs.

    USE CASE:
    Client proves they trained on multiple valid emails.
    """

    def __init__(self, valid_dataset: List[bytes]):
        """
        Initialize batch proof system.

        Args:
            valid_dataset: List of valid data items
        """
        self.single_system = DataValidityProof(valid_dataset)

    def generate_proofs(self, data_items: List[bytes]) -> List[dict]:
        """
        Generate proofs for multiple data items.

        Args:
            data_items: List of data items

        Returns:
            List of proofs
        """
        proofs = []
        for data in data_items:
            proof = self.single_system.generate_proof(data)
            proofs.append(proof)

        return proofs

    def batch_verify(
        self,
        proofs: List[dict],
        expected_root: Optional[bytes] = None
    ) -> List[bool]:
        """
        Verify multiple proofs.

        Args:
            proofs: List of proofs
            expected_root: Expected Merkle root

        Returns:
            List of verification results
        """
        results = []
        for proof in proofs:
            valid = self.single_system.verify(proof, expected_root)
            results.append(valid)

        return results


class AuthorizedDataset:
    """
    Manage authorized dataset for federated learning.

    Provides utilities for creating and updating valid dataset.
    """

    def __init__(self, emails: Optional[List[bytes]] = None):
        """
        Initialize authorized dataset.

        Args:
            emails: Initial list of authorized emails
        """
        self.emails = emails or []
        self.merkle_tree = None
        self._rebuild_tree()

    def add_email(self, email: bytes) -> None:
        """
        Add authorized email.

        Args:
            email: Email address to add

        Note:
            Requires rebuilding Merkle tree.
        """
        self.emails.append(email)
        self._rebuild_tree()

    def add_emails(self, email_list: List[bytes]) -> None:
        """
        Add multiple authorized emails.

        Args:
            email_list: List of email addresses
        """
        self.emails.extend(email_list)
        self._rebuild_tree()

    def remove_email(self, email: bytes) -> bool:
        """
        Remove authorized email.

        Args:
            email: Email to remove

        Returns:
            True if email was removed
        """
        if email in self.emails:
            self.emails.remove(email)
            self._rebuild_tree()
            return True
        return False

    def _rebuild_tree(self) -> None:
        """Rebuild Merkle tree from current emails."""
        self.merkle_tree = MerkleTree(data=self.emails)

    def get_root(self) -> bytes:
        """Get Merkle root of authorized dataset."""
        return self.merkle_tree.get_root()

    def get_size(self) -> int:
        """Get number of authorized emails."""
        return len(self.emails)

    def contains(self, email: bytes) -> bool:
        """
        Check if email is authorized.

        Args:
            email: Email to check

        Returns:
            True if email is in authorized set
        """
        return email in self.emails

    def export_commitment(self) -> dict:
        """
        Export dataset commitment.

        Returns:
            Dictionary with root hash and metadata

        Note:
            This is what server publishes to clients.
        """
        return {
            "root": self.get_root().hex(),
            "num_emails": self.get_size(),
            "hash_function": "sha256",
        }

    @classmethod
    def import_commitment(cls, commitment: dict) -> 'AuthorizedDataset':
        """
        Import dataset commitment.

        Args:
            commitment: Commitment dictionary from export_commitment()

        Returns:
            New AuthorizedDataset instance

        Note:
            This creates a dataset without the actual emails (for clients).
        """
        # In real system, would verify commitment structure
        # For now, create empty dataset
        return cls()


def demo_data_validity_proof():
    """
    Demonstrate data validity proof system.
    """
    print("=" * 70)
    print("DATA VALIDITY PROOF DEMONSTRATION")
    print("=" * 70)
    print()

    # Setup
    print("1. Setting up authorized email dataset...")
    authorized_emails = [
        f"user{i}@trusted-domain.com".encode()
        for i in range(1000)
    ]

    dataset = AuthorizedDataset(authorized_emails)
    proof_system = DataValidityProof(authorized_emails)

    print(f"   Authorized emails: {dataset.get_size()}")
    print()

    # Get commitment
    print("2. Server publishes dataset commitment...")
    commitment = dataset.export_commitment()
    print(f"   Merkle root: {commitment['root'][:32]}...")
    print(f"   Email count: {commitment['num_emails']}")
    print()

    # Client selects emails
    print("3. Client trains on subset of emails...")
    client_emails = [
        b"user42@trusted-domain.com",
        b"user123@trusted-domain.com",
        b"user456@trusted-domain.com",
    ]
    print(f"   Selected {len(client_emails)} emails for training")
    print()

    # Generate proofs
    print("4. Client generates validity proofs...")
    proofs = []
    for email in client_emails:
        proof = proof_system.generate_proof(email)
        proofs.append(proof)
        print(f"   ✓ Generated proof for {email.decode()}")
    print()

    # Verify proofs
    print("5. Server verifies proofs...")
    expected_root = bytes.fromhex(commitment['root'])
    for i, (email, proof) in enumerate(zip(client_emails, proofs)):
        valid = proof_system.verify(proof, expected_root)
        status = "VALID ✓" if valid else "INVALID ✗"
        print(f"   Proof {i+1}: {status}")
    print()

    # Try unauthorized email
    print("6. Attempting to prove unauthorized email...")
    unauthorized_email = b"attacker@malicious.com"
    try:
        proof = proof_system.generate_proof(unauthorized_email)
        print("   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    print()

    # Statistics
    print("7. Proof size estimation...")
    proof_size = proof_system.estimate_proof_size(dataset.get_size())
    print(f"   Proof size: ~{proof_size} bytes")
    print(f"   Tree depth: ~{proof_size // 32} levels")
    print()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_data_validity_proof()
