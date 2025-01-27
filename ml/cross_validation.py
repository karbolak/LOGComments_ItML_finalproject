import numpy as np
from itertools import product
from ml.metric import F1Score, LogLoss
from ml.logistic_regression import LogisticRegression

class CrossValidator:
    def __init__(self, model_class=LogisticRegression, n_splits=5, param_grid=None):
        self.n_splits = n_splits
        self.model_class = model_class
        self.param_grid = param_grid or {
            'n_iterations': [1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'regularization': [0.0, 0.1, 0.5],
            'decay_rate': [0.01, 0.1, 0.2],
            'decay_type': ["time", "exponential", "step", "none"],
            'threshold': [0.5, 0.55, 0.6],
        }
        
            # 'n_iterations': [1000],
            # 'learning_rate': [0.01, 0.05, 0.1],
            # 'regularization': [0.0, 0.1, 0.5],
            # 'decay_rate': [0.01, 0.05, 0.1],
            # 'decay_type': ["time", "exponential", "step", "none"],
    
    def _stratified_split(self, X, y):
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        pos_folds = np.array_split(pos_indices, self.n_splits)
        neg_folds = np.array_split(neg_indices, self.n_splits)
        splits = [(np.setdiff1d(np.arange(len(y)), np.concatenate([pos_folds[i], neg_folds[i]])),
                   np.concatenate([pos_folds[i], neg_folds[i]])) for i in range(self.n_splits)]
        return splits

    def cross_validate(self, X, y):
        best_model = None
        best_hparams = None
        best_score = -float('inf')
        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())

        metrics_summary = {'F1Score': [], 'loss': []}

        for hparams in param_combinations:
            print(f"CrossValidating for hyperparameters {hparams}.")
            hparam_dict = dict(zip(param_names, hparams))
            metrics = {'F1Score': [], 'loss': []}

            for train_idx, val_idx in self._stratified_split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = self.model_class(
                    learning_rate=hparam_dict['learning_rate'],
                    n_iterations=hparam_dict['n_iterations'],
                    regularization=hparam_dict['regularization'],
                    decay_type=hparam_dict['decay_type'],
                    decay_rate=hparam_dict['decay_rate'],
                    threshold=hparam_dict['threshold']
                )
                model.fit(X_train, y_train)

                predictions = model.predict(X_val)
                f1_metric = F1Score()  # Instantiate the metric class
                metrics['F1Score'].append(f1_metric(y_val, predictions))
                metrics['loss'].append(LogLoss()(y_val, model._sigmoid(np.dot(X_val, model.weights) + model.bias)))

            avg_F1Score = np.mean(metrics['F1Score'])
            if avg_F1Score > best_score:
                best_score = avg_F1Score
                best_model = model
                best_hparams = hparam_dict

            metrics_summary['F1Score'].extend(metrics['F1Score'])
            metrics_summary['loss'].extend(metrics['loss'])

        cv_summary = {
            metric: {'mean': np.mean(values), 'std': np.std(values)}
            for metric, values in metrics_summary.items()
        }

        return best_model, best_hparams, cv_summary

