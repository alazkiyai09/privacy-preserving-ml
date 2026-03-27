"""
Federated server (coordinator) implementation.

The server coordinates training across multiple parties.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from core.objective import Objective, get_objective


@dataclass
class TrainingConfig:
    """Configuration for federated training."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: float = 1.0
    lambda_reg: float = 1.0
    gamma: float = 0.0
    loss: str = 'binary:logistic'


class FederatedServer:
    """
    Central coordinator for federated GBDT training.

    The server coordinates the training process, aggregates results,
    and manages the overall training flow.
    """

    def __init__(self,
                 n_clients: int,
                 config: TrainingConfig):
        """
        Initialize federated server.

        Args:
            n_clients: Number of participating clients
            config: Training configuration
        """
        self.n_clients = n_clients
        self.config = config

        # Initialize objective function
        self.objective: Objective = get_objective(config.loss)

        # Training state
        self.base_score: float = 0.0
        self.current_predictions: Optional[np.ndarray] = None
        self.trees: List = []
        self.train_losses: List[float] = []

        # Client management
        self.client_states: Dict[int, Dict] = {}

    def initialize_training(self,
                          n_samples: int,
                          labels: np.ndarray) -> None:
        """
        Initialize training state.

        Args:
            n_samples: Number of training samples
            labels: Training labels
        """
        # Initialize base score
        if self.config.loss == 'binary:logistic':
            pos_ratio = np.mean(labels)
            self.base_score = np.log(pos_ratio / (1 - pos_ratio + 1e-12))
        else:
            self.base_score = np.mean(labels)

        # Initialize predictions
        self.current_predictions = np.full(n_samples, self.base_score)

    def compute_gradients_hessians(self,
                                  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and Hessians.

        Args:
            labels: Training labels

        Returns:
            Tuple of (gradients, hessians)
        """
        gradients = self.objective.compute_gradients(
            self.current_predictions, labels
        )
        hessians = self.objective.compute_hessians(
            self.current_predictions, labels
        )

        return gradients, hessians

    def distribute_gradients(self,
                            gradients: np.ndarray,
                            hessians: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Distribute gradients to all clients.

        Args:
            gradients: Gradient values
            hessians: Hessian values

        Returns:
            Dictionary mapping client_id -> (gradients, hessians)
        """
        # In practice, would securely send to each client
        client_data = {}

        for client_id in range(self.n_clients):
            client_data[client_id] = (gradients.copy(), hessians.copy())

        return client_data

    def aggregate_client_histograms(self,
                                   client_histograms: Dict[int, List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Aggregate histograms from all clients.

        Args:
            client_histograms: Dictionary mapping client_id -> list of histograms

        Returns:
            List of aggregated histograms per feature
        """
        # In practice, would use secure aggregation
        # For now, simple sum aggregation
        all_histograms = []

        # Collect all client histograms
        for client_id in sorted(client_histograms.keys()):
            all_histograms.append(client_histograms[client_id])

        # Handle empty case
        if not all_histograms:
            return []

        # Aggregate (sum) corresponding histograms
        # Assumes all clients have been aligned on sample order
        n_features_per_client = len(all_histograms[0])

        aggregated = []
        for feat_idx in range(n_features_per_client):
            # Sum histograms for this feature across clients
            feat_histograms = [client_hist[feat_idx] for client_hist in all_histograms]
            summed = np.sum(feat_histograms, axis=0)
            aggregated.append(summed)

        return aggregated

    def select_best_split(self,
                         all_client_histograms: List[List[np.ndarray]],
                         client_features: List[List[int]]) -> Tuple[int, int, float, float]:
        """
        Select the best split across all clients and features.

        Args:
            all_client_histograms: Histograms from each client
            client_features: Feature indices for each client

        Returns:
            Tuple of (client_id, feature_idx, split_value, gain)
        """
        best_gain = -np.inf
        best_client = 0
        best_feature = 0
        best_split_value = 0.0

        # This is a simplified version - in practice would use secure split finding
        for client_id, client_histograms in enumerate(all_client_histograms):
            for local_feat_idx, histogram in enumerate(client_histograms):
                # Compute split gain (simplified)
                # In practice, would use find_best_split_histogram
                G_total = np.sum(histogram[:, 0])
                H_total = np.sum(histogram[:, 1])

                # Try each bin as potential split
                for bin_idx in range(len(histogram) - 1):
                    G_left = np.sum(histogram[:bin_idx+1, 0])
                    H_left = np.sum(histogram[:bin_idx+1, 1])
                    G_right = G_total - G_left
                    H_right = H_total - H_left

                    if H_left < self.config.min_child_weight or H_right < self.config.min_child_weight:
                        continue

                    # Compute gain
                    def score(g, h):
                        return -(g ** 2) / (h + self.config.lambda_reg)

                    left_score = score(G_left, H_left)
                    right_score = score(G_right, H_right)
                    parent_score = score(G_total, H_total)

                    gain = left_score + right_score - parent_score

                    if gain > best_gain:
                        best_gain = gain
                        best_client = client_id
                        best_feature = local_feat_idx
                        best_split_value = bin_idx  # Simplified

        return (best_client, best_feature, best_split_value, best_gain)

    def update_predictions(self,
                          tree_predictions: np.ndarray) -> None:
        """
        Update predictions with tree contribution.

        Args:
            tree_predictions: Predictions from the new tree
        """
        self.current_predictions += self.config.learning_rate * tree_predictions

    def compute_loss(self, labels: np.ndarray) -> float:
        """
        Compute training loss.

        Args:
            labels: Training labels

        Returns:
            Loss value
        """
        return self.objective.loss(self.current_predictions, labels)

    def get_training_progress(self) -> Dict:
        """Get training progress information."""
        return {
            'n_trees': len(self.trees),
            'losses': self.train_losses,
            'current_predictions': self.current_predictions
        }


class Coordinator:
    """
    High-level coordinator for federated GBDT training.

    Manages the overall training workflow.
    """

    def __init__(self,
                 n_clients: int,
                 config: TrainingConfig):
        """
        Initialize coordinator.

        Args:
            n_clients: Number of clients
            config: Training configuration
        """
        self.server = FederatedServer(n_clients, config)
        self.config = config

    def orchestrate_training_round(self,
                                  iteration: int,
                                  labels: np.ndarray,
                                  client_features: Dict[int, np.ndarray],
                                  sample_indices: np.ndarray) -> Dict:
        """
        Orchestrate one round of training.

        Args:
            iteration: Current iteration number
            labels: Training labels
            client_features: Dictionary mapping client_id -> feature matrix
            sample_indices: Indices of samples in current node

        Returns:
            Dictionary with round results
        """
        # Server computes gradients
        gradients, hessians = self.server.compute_gradients_hessians(labels)

        # Distribute to clients
        client_data = self.server.distribute_gradients(gradients, hessians)

        # Clients compute local histograms (would happen in parallel)
        client_histograms = {}
        for client_id, (grad, hess) in client_data.items():
            # In practice, client would compute this
            # For now, placeholder
            pass

        # Aggregate histograms
        aggregated = self.server.aggregate_client_histograms(client_histograms)

        # Select best split
        best_split = self.server.select_best_split(aggregated, [])

        return {
            'iteration': iteration,
            'split': best_split
        }
