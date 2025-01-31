"""
Module containing hyperparameter choices for optimization and best
configurations.
"""

from typing import Dict, List, Union

# Dictionary containing hyperparameter ranges for optimization
HPARAMS_FOR_OPTIMIZATION: Dict[str, List[Union[int, float, str]]] = {
    'n_iterations': [2000, 5000],
    'learning_rate': [0.01, 0.05],
    'regularization': [0.001, 0.1],
    'l1_ratio': [0.2, 0.5, 0.7],
    'decay_rate': [0.006, 0.1, 0.5],
    'decay_type': ["time"],
}

# Dictionary containing the best hyperparameters found
BEST_HPARAMS: Dict[str, List[Union[int, float, str]]] = {
    'n_iterations': [5000],
    'learning_rate': [0.1],
    'regularization': [0.001],
    'l1_ratio': [0],
    'decay_rate': [0.5],
    'decay_type': ["time"],
}

# Default hyperparameters set to optimization parameters
HPARAMS: Dict[str, List[Union[int, float, str]]] = HPARAMS_FOR_OPTIMIZATION

# Uncomment to use best hyperparameters instead of running the optimization
# HPARAMS = BEST_HPARAMS
