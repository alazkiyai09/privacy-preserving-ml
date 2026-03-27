"""
Set Membership Proofs

Set membership proofs allow a prover to prove that a value belongs to a set
without revealing which element it is.

Applications in Federated Learning:
- Prove training data is from valid dataset
- Prove client is authorized (member of authorized clients)
- Prove gradient corresponds to known model architecture

Approaches:
1. Merkle Tree Proofs: Prove membership in Merkle tree
2. RSA Accumulators: Constant-size proofs
3. Bloom Filter: Probabilistic membership

This module implements Merkle tree-based set membership proofs.

Mathematical Concept:
- Build Merkle tree from set S = {s_1, s_2, ..., s_n}
- Prove value v ∈ S by providing Merkle path to root
- Root commitment is public and binding

Security Assumptions:
- Hash function is collision-resistant
- Merkle tree root is trusted
- Prover cannot create fake Merkle paths
"""

import hashlib
import json
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MerkleProof:
    """
    Merkle proof for set membership.

    Contains the path from leaf to root and direction indicators.
    """
    leaf_index: int
    path: List[bytes]  # Sibling hashes along the path
    directions: List[bool]  # True = left sibling, False = right sibling
    root: bytes  # Root hash for verification


class MerkleTree:
    """
    Merkle tree for efficient set membership proofs.

    PROPERTIES:
    - Root hash commits to entire set
    - Membership proof is O(log n) hashes
    - Can prove both membership and non-membership

    SECURITY ASSUMPTIONS:
    - Hash function is collision-resistant
    - Root hash is trusted and unforgeable
    """

    def __init__(self, data: List[bytes] = None, hash_function: str = "sha256"):
        """
        Initialize Merkle tree.

        Args:
            data: Initial data (list of byte strings)
            hash_function: Hash function to use (sha256, sha3_256, etc.)

        Example:
            >>> tree = MerkleTree()
            >>> tree.add_leaf(b"email1@domain.com")
            >>> tree.add_leaf(b"email2@domain.com")
            >>> root = tree.get_root()
            >>> proof = tree.get_proof(0)  # Proof for first element
            >>> assert tree.verify_proof(proof, b"email1@domain.com")
        """
        self.hash_function = hash_function
        self.leaves: List[bytes] = []
        self.levels: List[List[bytes]] = []
        self.root: Optional[bytes] = None

        if data:
            for item in data:
                self.add_leaf(item)

    def _hash(self, data: bytes) -> bytes:
        """
        Hash data using configured hash function.

        Args:
            data: Data to hash

        Returns:
            Hash digest
        """
        if self.hash_function == "sha256":
            return hashlib.sha256(data).digest()
        elif self.hash_function == "sha3_256":
            return hashlib.sha3_256(data).digest()
        else:
            raise ValueError(f"Unsupported hash function: {self.hash_function}")

    def add_leaf(self, data: bytes) -> int:
        """
        Add a leaf to the Merkle tree.

        Args:
            data: Leaf data (bytes)

        Returns:
            Index of the added leaf

        Note:
            Adding a leaf requires rebuilding the tree.
            For batch addition, use add_leaves() instead.
        """
        self.leaves.append(self._hash(data))
        self._build_tree()
        return len(self.leaves) - 1

    def add_leaves(self, data_list: List[bytes]) -> None:
        """
        Add multiple leaves efficiently.

        Args:
            data_list: List of leaf data

        Note:
            More efficient than multiple add_leaf() calls.
        """
        for data in data_list:
            self.leaves.append(self._hash(data))
        self._build_tree()

    def _build_tree(self) -> None:
        """
        Build the Merkle tree from leaves.

        Constructs binary tree bottom-up:
        - Level 0: Leaves
        - Level 1: Pairs of leaves hashed together
        - ...
        - Top level: Root
        """
        if not self.leaves:
            self.root = None
            self.levels = []
            return

        # Start with leaves
        current_level = self.leaves.copy()
        self.levels = [current_level]

        # Build tree level by level
        while len(current_level) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash left + right
                    combined = current_level[i] + current_level[i + 1]
                    next_level.append(self._hash(combined))
                else:
                    # Odd number of elements - promote to next level
                    next_level.append(current_level[i])

            self.levels.append(next_level)
            current_level = next_level

        # Root is the last level
        self.root = current_level[0] if current_level else None

    def get_root(self) -> Optional[bytes]:
        """
        Get the Merkle root hash.

        Returns:
            Root hash (or None if tree is empty)

        Note:
            The root commits to the entire dataset.
            If any leaf changes, the root changes (unforgeable).
        """
        return self.root

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """
        Generate Merkle proof for a leaf.

        Args:
            leaf_index: Index of the leaf

        Returns:
            MerkleProof object containing path to root

        Example:
            >>> tree = MerkleTree()
            >>> idx = tree.add_leaf(b"my_data")
            >>> proof = tree.get_proof(idx)
            >>> proof contains all information needed to verify membership
        """
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range")

        path = []
        directions = []

        # Start at leaf level
        current_idx = leaf_index

        # Walk up the tree
        for level in range(len(self.levels) - 1):
            current_level = self.levels[level]

            # Determine sibling index
            if current_idx % 2 == 0:
                # Current node is left child
                sibling_idx = current_idx + 1
                is_left = True
            else:
                # Current node is right child
                sibling_idx = current_idx - 1
                is_left = False

            # Add sibling to path if it exists
            if sibling_idx < len(current_level):
                path.append(current_level[sibling_idx])
                directions.append(is_left)
            else:
                # No sibling (odd number of nodes at this level)
                path.append(b"")
                directions.append(is_left)

            # Move to parent
            current_idx = current_idx // 2

        return MerkleProof(
            leaf_index=leaf_index,
            path=path,
            directions=directions,
            root=self.root
        )

    def verify_proof(self, proof: MerkleProof, data: bytes) -> bool:
        """
        Verify Merkle proof for a piece of data.

        Args:
            proof: Merkle proof from get_proof()
            data: The data to verify

        Returns:
            True if data is in the tree at the claimed position

        Verification Process:
        1. Hash the data to get leaf
        2. Walk up the tree using the proof path
        3. Check if computed root matches proof root
        """
        # Compute leaf hash
        current_hash = self._hash(data)

        # Walk up the tree using the proof
        for sibling, is_left in zip(proof.path, proof.directions):
            if not sibling:  # No sibling at this level
                continue

            if is_left:
                # Current node is left, combine with right sibling
                combined = current_hash + sibling
            else:
                # Current node is right, combine with left sibling
                combined = sibling + current_hash

            current_hash = self._hash(combined)

        # Check if we arrived at the root
        return current_hash == proof.root

    def to_json(self) -> str:
        """
        Serialize tree to JSON.

        Returns:
            JSON string representation

        Note:
            Useful for storing/publishing the tree structure.
        """
        data = {
            'root': self.root.hex() if self.root else None,
            'num_leaves': len(self.leaves),
            'hash_function': self.hash_function,
            'leaves': [leaf.hex() for leaf in self.leaves],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'MerkleTree':
        """
        Deserialize tree from JSON.

        Args:
            json_str: JSON string from to_json()

        Returns:
            MerkleTree object

        Note:
            Rebuilds the tree from stored data.
        """
        data = json.loads(json_str)
        tree = cls(hash_function=data['hash_function'])

        # Restore leaves
        for leaf_hex in data['leaves']:
            tree.leaves.append(bytes.fromhex(leaf_hex))

        # Rebuild tree
        tree._build_tree()

        return tree


class SetMembershipProof:
    """
    Zero-knowledge proof of set membership using Merkle trees.

    USE CASE IN FEDERATED LEARNING:
    Prove that training data is from the authorized dataset without
    revealing which specific emails were used.

    SECURITY ASSUMPTIONS:
    - Merkle tree root is trusted
    - Hash function is collision-resistant
    - Prover cannot find preimage for hash
    """

    def __init__(self, merkle_tree: MerkleTree):
        """
        Initialize set membership proof system.

        Args:
            merkle_tree: The Merkle tree representing the set

        Example:
            >>> tree = MerkleTree()
            >>> for email in authorized_emails:
            ...     tree.add_leaf(email.encode())
            >>> proof_system = SetMembershipProof(tree)
            >>> proof = proof_system.generate_proof(b"email@domain.com")
            >>> assert proof_system.verify(proof)
        """
        self.tree = merkle_tree

    def generate_proof(self, value: bytes) -> dict:
        """
        Generate proof that value is in the set.

        Args:
            value: The value to prove membership for

        Returns:
            Proof dictionary containing Merkle proof

        Note:
            The proof reveals the value itself. For true zero-knowledge,
            use a commitment scheme and prove the commitment opens to
            a value in the set.
        """
        # Find index of value in tree
        value_hash = self.tree._hash(value)

        try:
            leaf_index = self.tree.leaves.index(value_hash)
        except ValueError:
            raise ValueError(f"Value not in set: {value}")

        # Get Merkle proof
        merkle_proof = self.tree.get_proof(leaf_index)

        # Create proof object
        proof = {
            'value': value.hex() if isinstance(value, bytes) else value,
            'leaf_index': leaf_index,
            'merkle_path': [h.hex() for h in merkle_proof.path],
            'directions': merkle_proof.directions,
            'root': merkle_proof.root.hex(),
            'hash_function': self.tree.hash_function,
        }

        return proof

    def verify(self, proof: dict) -> bool:
        """
        Verify set membership proof.

        Args:
            proof: Proof dictionary from generate_proof()

        Returns:
            True if proof is valid

        Verification Process:
        1. Reconstruct MerkleProof object
        2. Verify Merkle proof
        3. Check root matches trusted root
        """
        try:
            # Reconstruct MerkleProof
            merkle_proof = MerkleProof(
                leaf_index=proof['leaf_index'],
                path=[bytes.fromhex(h) for h in proof['merkle_path']],
                directions=proof['directions'],
                root=bytes.fromhex(proof['root'])
            )

            # Get value
            value = bytes.fromhex(proof['value']) if isinstance(proof['value'], str) else proof['value']

            # Verify Merkle proof
            if not self.tree.verify_proof(merkle_proof, value):
                return False

            # Check root matches trusted root
            if merkle_proof.root != self.tree.root:
                return False

            return True

        except Exception:
            return False

    def generate_zk_proof(
        self,
        value: bytes,
        commitment: bytes,
        commitment_randomness: int
    ) -> dict:
        """
        Generate zero-knowledge proof of set membership.

        Instead of revealing the value, we prove that a commitment
        opens to a value in the set.

        Args:
            value: The actual value (secret)
            commitment: Commitment to the value
            commitment_randomness: Randomness used in commitment

        Returns:
            Proof dictionary

        Concept:
        1. Prove commitment opens to some value
        2. Prove that value is in the set (Merkle proof)
        3. Use sigma protocol to connect commitment to Merkle proof

        Note:
            This is a simplified version. Full implementation would use
            a sigma protocol to prove the commitment opens to the value
            corresponding to the Merkle proof without revealing the value.
        """
        # Get regular Merkle proof
        merkle_proof = self.generate_proof(value)

        # In full ZK proof, we'd also include:
        # - Proof that commitment opens to value (without revealing value)
        # - Sigma protocol connecting commitment to Merkle leaf
        # - Challenges and responses for zero-knowledge property

        proof = {
            'commitment': commitment.hex() if isinstance(commitment, bytes) else commitment,
            'merkle_proof': merkle_proof,
            # In full implementation, add ZK parts here
        }

        return proof

    def verify_zk_proof(self, proof: dict) -> bool:
        """
        Verify zero-knowledge set membership proof.

        Args:
            proof: Proof dictionary from generate_zk_proof()

        Returns:
            True if proof is valid

        Note:
            This simplified version just checks the Merkle proof.
            Full implementation would verify the ZK parts as well.
        """
        try:
            # Verify Merkle proof part
            if not self.verify(proof['merkle_proof']):
                return False

            # In full implementation, verify:
            # - Commitment is valid
            # - Sigma protocol proofs
            # - No information leakage

            return True

        except Exception:
            return False


class RSAAccumulator:
    """
    RSA Accumulator for constant-size set membership proofs.

    ADVANTAGE OVER MERKLE TREES:
    - Proof size is constant (O(1)) regardless of set size
    - Non-membership proofs also possible

    DISADVANTAGES:
    - Requires trusted setup
    - Computationally more expensive
    - Requires trapdoor for witness generation (or batch updates)

    Mathematical Form:
    Accumulator = g^(product of all elements) mod N
    Witness for element x: g^(product of all elements except x) mod N

    SECURITY ASSUMPTIONS:
    - RSA problem is hard (can't factor N)
    - Strong RSA assumption
    """

    def __init__(self, n: int = None, g: int = None):
        """
        Initialize RSA accumulator.

        Args:
            n: RSA modulus (product of two large primes)
            g: Generator (typically 2 or random element)

        SECURITY REQUIREMENT:
            - n must be product of two safe primes
            - Factorization of n must be destroyed (toxic waste)
            - In production, n should come from MPC ceremony

        Example:
            >>> acc = RSAAccumulator()
            >>> acc.add_element(42)
            >>> acc.add_element(123)
            >>> witness = acc.get_witness(42)
            >>> assert acc.verify_member(42, witness)
        """
        # WARNING: These are placeholder values!
        # Production: Use actual RSA-2048 or larger
        if n is None:
            # Simple RSA modulus for demonstration (NOT SECURE!)
            self.n = 2038074743  # Small product of primes
        else:
            self.n = n

        if g is None:
            self.g = 2
        else:
            self.g = g

        self.accumulated_product = 1
        self.elements = []

    def add_element(self, element: int) -> None:
        """
        Add an element to the accumulator.

        Args:
            element: The element to add (must be coprime to n)

        Note:
            In production, use prime fields or map elements to primes.
        """
        # In production, map element to prime first
        self.elements.append(element)
        self.accumulated_product *= element

    def get_accumulator_value(self) -> int:
        """
        Get the current accumulator value.

        Returns:
            Accumulator = g^(product of all elements) mod n
        """
        return pow(self.g, self.accumulated_product, self.n)

    def get_witness(self, element: int) -> int:
        """
        Get witness (proof of membership) for an element.

        Args:
            element: The element to get witness for

        Returns:
            Witness = g^(product of all elements except element) mod n

        Verification:
        accumulator == witness^element mod n
        """
        if element not in self.elements:
            raise ValueError(f"Element {element} not in accumulator")

        # Compute product of all elements except this one
        product_except = 1
        for e in self.elements:
            if e != element:
                product_except *= e

        return pow(self.g, product_except, self.n)

    def verify_member(self, element: int, witness: int) -> bool:
        """
        Verify that element is in the accumulator.

        Args:
            element: The element to verify
            witness: The witness for this element

        Returns:
            True if element is in accumulator

        Verification Check:
        accumulator == witness^element mod n
        """
        accumulator = self.get_accumulator_value()
        expected = pow(witness, element, self.n)
        return accumulator == expected
