"""
Base GBDT implementation without privacy features.

This implements standard gradient boosting for comparison with
privacy-preserving variants.
"""

import numpy as np
from typing import List, Optional, Callable
from .tree_builder import TreeBuilder, TreeNode
from .objective import Objective, LogisticLoss, get_objective


class GBDTBase:
    """
    Base Gradient Boosted Decision Tree implementation.

    This is a standard GBDT implementation without privacy features,
    used as a baseline for comparison with privacy-preserving versions.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 min_child_weight: float = 1.0,
                 lambda_reg: float = 1.0,
                 gamma: float = 0.0,
                 subsample: float = 1.0,
                 loss: str = 'binary:logistic',
                 random_state: Optional[int] = None):
        """
        Initialize GBDT model.

        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum depth of each tree
            learning_rate: Shrinkage factor for tree contributions
            min_child_weight: Minimum sum of Hessians in a child node
            lambda_reg: L2 regularization on leaf weights
            gamma: Minimum gain required to make a split
            subsample: Fraction of samples to use for each tree
            loss: Loss function ('binary:logistic' or 'reg:squarederror')
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.subsample = subsample
        self.loss_name = loss
        self.random_state = random_state

        # Initialize components
        self.objective: Objective = get_objective(loss)
        self.trees: List[TreeNode] = []
        self.base_score: float = 0.0
        self.n_classes: int = 1

        # Training history
        self.train_losses: List[float] = []

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[tuple] = None,
            verbose: bool = False) -> 'GBDTBase':
        """
        Fit the GBDT model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            eval_set: Optional tuple (X_val, y_val) for validation
            verbose: Whether to print progress

        Returns:
            self (fitted model)
        """
        # Initialize random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize base score (mean prediction)
        if self.loss_name == 'binary:logistic':
            # For logistic loss, use log-odds
            pos_ratio = np.mean(y)
            self.base_score = np.log(pos_ratio / (1 - pos_ratio + 1e-12))
        else:
            # For squared error, use mean
            self.base_score = np.mean(y)

        # Initial predictions
        predictions = np.full(n_samples, self.base_score)

        # Training loop
        for iteration in range(self.n_estimators):
            # Compute gradients and Hessians
            gradients = self.objective.compute_gradients(predictions, y)
            hessians = self.objective.compute_hessians(predictions, y)

            # Subsample if needed
            if self.subsample < 1.0:
                sample_mask = np.random.rand(n_samples) < self.subsample
                sample_indices = np.where(sample_mask)[0]
            else:
                sample_indices = np.arange(n_samples)

            # Build tree to fit gradients
            tree_builder = TreeBuilder(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma
            )

            tree = tree_builder.build(
                X, gradients, hessians, sample_indices, depth=0
            )
            self.trees.append(tree)

            # Update predictions
            tree_predictions = tree_builder.predict(X)
            predictions += self.learning_rate * tree_predictions

            # Compute and store loss
            loss = self.objective.loss(predictions, y)
            self.train_losses.append(loss)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_estimators}, Loss: {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make raw predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Raw predictions (log-odds for classification)
        """
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.base_score)

        for tree in self.trees:
            # Create a tree builder to make predictions
            tree_builder = TreeBuilder(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma
            )
            tree_builder.root = tree
            tree_predictions = tree_builder.predict(X)
            predictions += self.learning_rate * tree_predictions

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability array (n_samples, 2) for binary classification
        """
        raw_predictions = self.predict(X)

        if self.loss_name == 'binary:logistic':
            proba_positive = self.objective.sigmoid(raw_predictions)
            proba_negative = 1.0 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
        else:
            # For regression, return raw predictions as probabilities
            return np.column_stack([1 - raw_predictions, raw_predictions])

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Decision threshold for positive class

        Returns:
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy
        """
        predictions = self.predict_class(X)
        return np.mean(predictions == y)
