"""
Decision tree builder for Gradient Boosted Decision Trees.

Implements decision tree construction using gradient-based split finding.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TreeNode:
    """A node in the decision tree."""
    node_id: int
    is_leaf: bool
    depth: int

    # Split node attributes
    feature_idx: Optional[int] = None
    split_value: Optional[float] = None
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None

    # Leaf node attributes
    leaf_value: Optional[float] = None
    sample_count: Optional[int] = None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(id={self.node_id}, value={self.leaf_value:.4f}, n={self.sample_count})"
        else:
            return f"Node(id={self.node_id}, feat={self.feature_idx}, thresh={self.split_value:.4f})"


class TreeBuilder:
    """
    Build a decision tree using gradient-based split finding.

    The tree is built to fit the negative gradients (residuals),
    which is the core idea of gradient boosting.
    """

    def __init__(self,
                 max_depth: int = 6,
                 min_child_weight: float = 1.0,
                 lambda_reg: float = 1.0,
                 gamma: float = 0.0,
                 min_samples_split: int = 2):
        """
        Initialize tree builder.

        Args:
            max_depth: Maximum depth of the tree
            min_child_weight: Minimum sum of Hessians in a child node
            lambda_reg: L2 regularization on leaf weights
            gamma: Minimum gain required to make a split
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None
        self.n_nodes = 0

    def build(self,
              features: np.ndarray,
              gradients: np.ndarray,
              hessians: np.ndarray,
              sample_indices: np.ndarray,
              depth: int = 0) -> TreeNode:
        """
        Recursively build the decision tree.

        Args:
            features: Full feature matrix (n_samples, n_features)
            gradients: Gradient values (n_samples,)
            hessians: Hessian values (n_samples,)
            sample_indices: Indices of samples in current node
            depth: Current depth

        Returns:
            TreeNode representing the root of the (sub)tree
        """
        node_id = self.n_nodes
        self.n_nodes += 1

        # Check stopping conditions
        should_split, split_info = self._should_split(
            features, gradients, hessians, sample_indices, depth
        )

        if not should_split:
            # Create leaf node
            leaf_value = self._compute_leaf_value(gradients, hessians, sample_indices)
            node = TreeNode(
                node_id=node_id,
                is_leaf=True,
                depth=depth,
                leaf_value=leaf_value,
                sample_count=len(sample_indices)
            )
            # Set root if this is the root node
            if depth == 0:
                self.root = node
            return node

        # Create internal node and split
        node = TreeNode(
            node_id=node_id,
            is_leaf=False,
            depth=depth,
            feature_idx=split_info['feature_idx'],
            split_value=split_info['split_value']
        )

        # Partition samples
        left_mask = features[sample_indices, split_info['feature_idx']] < split_info['split_value']
        right_mask = ~left_mask

        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[right_mask]

        # Recursively build children
        node.left_child = self.build(
            features, gradients, hessians, left_indices, depth + 1
        )
        node.right_child = self.build(
            features, gradients, hessians, right_indices, depth + 1
        )

        # Set root if this is the root node
        if depth == 0:
            self.root = node

        return node

    def _should_split(self,
                     features: np.ndarray,
                     gradients: np.ndarray,
                     hessians: np.ndarray,
                     sample_indices: np.ndarray,
                     depth: int) -> Tuple[bool, Optional[dict]]:
        """
        Determine if a node should be split.

        Args:
            features: Feature matrix
            gradients: Gradient values
            hessians: Hessian values
            sample_indices: Indices of samples in current node
            depth: Current depth

        Returns:
            Tuple of (should_split, split_info_dict)
        """
        # Check max depth
        if depth >= self.max_depth:
            return False, None

        # Check minimum samples
        if len(sample_indices) < self.min_samples_split:
            return False, None

        # Check minimum Hessian sum
        hessian_sum = np.sum(hessians[sample_indices])
        if hessian_sum < 2 * self.min_child_weight:
            return False, None

        # Find best split
        split_info = self._find_best_split(
            features, gradients, hessians, sample_indices
        )

        # For root node (depth=0), force at least one split if constraints are met
        # This ensures tree building even when gains are negative (balanced classes)
        if depth == 0:
            return True, split_info

        # Check minimum gain for non-root nodes
        if split_info['gain'] < self.gamma:
            return False, None

        return True, split_info

    def _find_best_split(self,
                        features: np.ndarray,
                        gradients: np.ndarray,
                        hessians: np.ndarray,
                        sample_indices: np.ndarray) -> dict:
        """
        Find the best split using exact (greedy) split finding.

        Args:
            features: Feature matrix
            gradients: Gradient values
            hessians: Hessian values
            sample_indices: Indices of samples in current node

        Returns:
            Dictionary with split information
        """
        n_features = features.shape[1]
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0

        # Total statistics
        G = np.sum(gradients[sample_indices])
        H = np.sum(hessians[sample_indices])

        for feat_idx in range(n_features):
            feature_values = features[sample_indices, feat_idx]
            grad_values = gradients[sample_indices]
            hess_values = hessians[sample_indices]

            # Sort by feature values
            sort_idx = np.argsort(feature_values)
            sorted_values = feature_values[sort_idx]
            sorted_grad = grad_values[sort_idx]
            sorted_hess = hess_values[sort_idx]

            # Scan through possible splits
            G_left = 0.0
            H_left = 0.0

            for i in range(len(sorted_values) - 1):
                G_left += sorted_grad[i]
                H_left += sorted_hess[i]

                # Skip duplicate values
                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                G_right = G - G_left
                H_right = H - H_left

                # Check minimum child weight
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                # Compute gain
                gain = self._compute_gain(G_left, H_left, G_right, H_right, G, H)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    # Split is between sorted_values[i] and sorted_values[i+1]
                    best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0

        return {
            'feature_idx': best_feature,
            'split_value': best_threshold,
            'gain': best_gain
        }

    def _compute_gain(self,
                     G_left: float, H_left: float,
                     G_right: float, H_right: float,
                     G: float, H: float) -> float:
        """
        Compute the gain of a split.

        Args:
            G_left, H_left: Gradient and Hessian sums for left child
            G_right, H_right: Gradient and Hessian sums for right child
            G, H: Total gradient and Hessian sums

        Returns:
            Split gain
        """
        def score(g, h):
            return -(g ** 2) / (h + self.lambda_reg)

        left_score = score(G_left, H_left)
        right_score = score(G_right, H_right)
        parent_score = score(G, H)

        return left_score + right_score - parent_score

    def _compute_leaf_value(self,
                           gradients: np.ndarray,
                           hessians: np.ndarray,
                           sample_indices: np.ndarray) -> float:
        """
        Compute the optimal leaf value.

        For a given loss function, the optimal leaf value is:
        w* = -sum(gradients) / (sum(hessians) + lambda)

        Args:
            gradients: Gradient values
            hessians: Hessian values
            sample_indices: Indices of samples in this node

        Returns:
            Optimal leaf weight
        """
        G = np.sum(gradients[sample_indices])
        H = np.sum(hessians[sample_indices])

        leaf_value = -G / (H + self.lambda_reg)

        return leaf_value

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions with the tree.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.root is None:
            raise ValueError("Tree not built. Call build() first.")

        predictions = np.zeros(features.shape[0])

        for i in range(features.shape[0]):
            predictions[i] = self._predict_single(self.root, features[i, :])

        return predictions

    def _predict_single(self, node: TreeNode, sample: np.ndarray) -> float:
        """
        Predict a single sample by traversing the tree.

        Args:
            node: Current tree node
            sample: Feature values for a single sample

        Returns:
            Leaf value
        """
        if node.is_leaf:
            return node.leaf_value

        if sample[node.feature_idx] < node.split_value:
            return self._predict_single(node.left_child, sample)
        else:
            return self._predict_single(node.right_child, sample)

    def get_n_leaves(self) -> int:
        """Get the number of leaf nodes in the tree."""
        if self.root is None:
            return 0
        return self._count_leaves(self.root)

    def _count_leaves(self, node: TreeNode) -> int:
        """Recursively count leaf nodes."""
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left_child) + self._count_leaves(node.right_child)

    def get_depth(self) -> int:
        """Get the depth of the tree."""
        if self.root is None:
            return 0
        return self._get_depth(self.root)

    def _get_depth(self, node: TreeNode) -> int:
        """Recursively compute tree depth."""
        if node.is_leaf:
            return node.depth
        return max(self._get_depth(node.left_child), self._get_depth(node.right_child))


def print_tree(node: TreeNode, depth: int = 0):
    """
    Print tree structure for debugging.

    Args:
        node: Tree node to print
        depth: Current depth for indentation
    """
    prefix = "  " * depth
    print(f"{prefix}{node}")

    if not node.is_leaf:
        print_tree(node.left_child, depth + 1)
        print_tree(node.right_child, depth + 1)
