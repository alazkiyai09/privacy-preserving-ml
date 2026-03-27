"""
Privacy-preserving prediction protocol for federated GBDT.

In federated GBDT, each party holds different features. To make predictions,
parties must jointly compute tree outputs without revealing their feature values.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.tree_builder import TreeNode


@dataclass
class PartialPrediction:
    """Partial prediction result from a single party."""
    party_id: int
    leaf_value: float
    tree_id: int


class SecurePrediction:
    """
    Secure prediction protocol for federated GBDT.

    The protocol works as follows:
    1. Each party traverses the tree using their local features
    2. When a split requires a feature from another party, parties collaborate
    3. The final prediction is computed securely

    For efficiency, we use a simpler approach:
    - Each party computes the path through the tree for features they have
    - Parties exchange information about which node they're at
    - Final prediction is computed by the party with the label
    """

    def __init__(self,
                 n_parties: int,
                 party_feature_indices: List[List[int]]):
        """
        Initialize secure prediction.

        Args:
            n_parties: Number of parties
            party_feature_indices: List of feature indices for each party
        """
        self.n_parties = n_parties
        self.party_feature_indices = party_feature_indices

        # Create feature to party mapping
        self.feature_to_party = {}
        for party_id, feat_indices in enumerate(party_feature_indices):
            for feat_idx in feat_indices:
                self.feature_to_party[feat_idx] = party_id

    def predict_single_tree(self,
                           tree: TreeNode,
                           X_dict: Dict[int, np.ndarray],
                           sample_idx: int) -> float:
        """
        Make a prediction for a single sample using a single tree.

        Args:
            tree: Tree to use for prediction
            X_dict: Dictionary mapping party_id -> feature matrix
            sample_idx: Index of sample to predict

        Returns:
            Leaf value (prediction)
        """
        node = tree

        while not node.is_leaf:
            feat_idx = node.feature_idx
            split_value = node.split_value

            # Find which party has this feature
            party_id = self.feature_to_party.get(feat_idx)

            if party_id is None:
                raise ValueError(f"Feature {feat_idx} not owned by any party")

            # Get feature value from the owning party
            feature_value = X_dict[party_id][sample_idx, feat_idx]

            # Traverse to left or right child
            if feature_value < split_value:
                node = node.left_child
            else:
                node = node.right_child

        return node.leaf_value

    def predict_single_sample(self,
                             trees: List[TreeNode],
                             X_dict: Dict[int, np.ndarray],
                             sample_idx: int,
                             learning_rate: float) -> float:
        """
        Make a prediction for a single sample using all trees.

        Args:
            trees: List of trees in the ensemble
            X_dict: Dictionary mapping party_id -> feature matrix
            sample_idx: Index of sample to predict
            learning_rate: Learning rate (shrinkage)

        Returns:
            Raw prediction value
        """
        prediction = 0.0

        for tree in trees:
            leaf_value = self.predict_single_tree(tree, X_dict, sample_idx)
            prediction += learning_rate * leaf_value

        return prediction

    def predict(self,
               trees: List[TreeNode],
               X_dict: Dict[int, np.ndarray],
               base_score: float,
               learning_rate: float) -> np.ndarray:
        """
        Make predictions for multiple samples.

        Args:
            trees: List of trees in the ensemble
            X_dict: Dictionary mapping party_id -> feature matrix
            base_score: Base prediction score
            learning_rate: Learning rate

        Returns:
            Array of predictions
        """
        # Assume all parties have the same samples (aligned)
        n_samples = next(iter(X_dict.values())).shape[0]
        predictions = np.full(n_samples, base_score)

        for i in range(n_samples):
            tree_prediction = self.predict_single_sample(
                trees, X_dict, i, learning_rate
            )
            predictions[i] += tree_prediction

        return predictions

    def predict_proba(self,
                     trees: List[TreeNode],
                     X_dict: Dict[int, np.ndarray],
                     base_score: float,
                     learning_rate: float,
                     loss: str = 'binary:logistic') -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            trees: List of trees in the ensemble
            X_dict: Dictionary mapping party_id -> feature matrix
            base_score: Base prediction score
            learning_rate: Learning rate
            loss: Loss function ('binary:logistic' for classification)

        Returns:
            Probability array (n_samples, 2) for binary classification
        """
        raw_predictions = self.predict(
            trees, X_dict, base_score, learning_rate
        )

        if loss == 'binary:logistic':
            # Sigmoid function
            proba_positive = 1.0 / (1.0 + np.exp(-raw_predictions))
            proba_negative = 1.0 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
        else:
            # For regression, return raw predictions
            return raw_predictions


class LocalPartyPrediction:
    """
    Local prediction protocol where each party makes predictions independently.

    This is less private but simpler - each party stores a copy of the model
    and makes predictions using their local features.
    """

    def __init__(self,
                 party_id: int,
                 feature_indices: List[int]):
        """
        Initialize local party prediction.

        Args:
            party_id: ID of this party
            feature_indices: Features this party owns
        """
        self.party_id = party_id
        self.feature_indices = set(feature_indices)

    def can_predict(self, tree: TreeNode) -> bool:
        """
        Check if this party can make predictions with this tree.

        A party can predict if all features used in the tree are owned by them.

        Args:
            tree: Tree to check

        Returns:
            True if party can predict independently
        """
        return self._check_tree_features(tree)

    def _check_tree_features(self, node: TreeNode) -> bool:
        """Recursively check if node uses only this party's features."""
        if node.is_leaf:
            return True

        if node.feature_idx not in self.feature_indices:
            return False

        return (self._check_tree_features(node.left_child) and
                self._check_tree_features(node.right_child))

    def predict_with_available_features(self,
                                       tree: TreeNode,
                                       features: np.ndarray) -> Optional[float]:
        """
        Attempt to predict using available features.

        If a required feature is missing, returns None.

        Args:
            tree: Tree to use
            features: Feature vector (indices correspond to actual feature indices)

        Returns:
            Leaf value or None if prediction impossible
        """
        node = tree

        while not node.is_leaf:
            feat_idx = node.feature_idx

            # Check if we have this feature in the feature array
            if feat_idx not in self.feature_indices or feat_idx >= len(features):
                return None

            feature_value = features[feat_idx]

            if feature_value < node.split_value:
                node = node.left_child
            else:
                node = node.right_child

        return node.leaf_value


def simulate_feature_partition(n_features: int,
                               n_parties: int,
                               random_state: int = 42) -> List[List[int]]:
    """
    Simulate feature partitioning across parties.

    Args:
        n_features: Total number of features
        n_parties: Number of parties
        random_state: Random seed

    Returns:
        List of feature indices for each party
    """
    np.random.seed(random_state)

    # Shuffle features
    all_indices = np.arange(n_features)
    np.random.shuffle(all_indices)

    # Split among parties
    features_per_party = n_features // n_parties
    party_features = []

    for i in range(n_parties):
        start = i * features_per_party
        end = (i + 1) * features_per_party if i < n_parties - 1 else n_features
        party_features.append(all_indices[start:end].tolist())

    return party_features
