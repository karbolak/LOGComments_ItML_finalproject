HPARAMS_FOR_OPTIMIZATION = {
        'n_iterations': [2000, 5000],
        'learning_rate': [0.01, 0.05],
        'regularization': [0.001, 0.1],
        'l1_ratio': [0.2, 0.5, 0.7],
        'decay_rate': [0.006, 0.1, 0.5],
        'decay_type': ["time"],
    }

BEST_HPARAMS = {
    'n_iterations': [5000],
    'learning_rate': [0.1],
    'regularization': [0.001],
    'l1_ratio': [0],
    'decay_rate': [0.5],
    'decay_type': ["time"],
}

HPARAMS = HPARAMS_FOR_OPTIMIZATION

# Uncomment if you want to use best hyperparameters instead of running the optimization
HPARAMS = BEST_HPARAMS