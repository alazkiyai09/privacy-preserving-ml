"""
Core GBDT components.

This module contains the fundamental building blocks for gradient boosting:
- Objective functions (gradients, Hessians)
- Histogram builder for efficient split finding
- Tree builder for decision tree construction
- Base GBDT ensemble
"""

from .objective import (
    Objective,
    LogisticLoss,
    SquaredErrorLoss,
    get_objective,
    compute_gradients,
    compute_hessians
)

from .histogram import (
    HistogramBuilder,
    SplitInfo,
    find_best_split_histogram,
    find_best_split_all_features
)

from .tree_builder import (
    TreeNode,
    TreeBuilder,
    print_tree
)

from .gbdt_base import GBDTBase

__all__ = [
    # Objective
    'Objective',
    'LogisticLoss',
    'SquaredErrorLoss',
    'get_objective',
    'compute_gradients',
    'compute_hessians',

    # Histogram
    'HistogramBuilder',
    'SplitInfo',
    'find_best_split_histogram',
    'find_best_split_all_features',

    # Tree Builder
    'TreeNode',
    'TreeBuilder',
    'print_tree',

    # GBDT
    'GBDTBase',
]
