import numpy as np
from ml.metric import F1Score, LogLoss
from ml.logistic_regression import LogisticRegression


class CrossValidator:
    def __init__(self, model_class=LogisticRegression, n_splits=5):
        self.n_splits = n_splits
        self.model_class = model_class

    def _stratified_split(self, X, y):
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        pos_folds = np.array_split(pos_indices, self.n_splits)
        neg_folds = np.array_split(neg_indices, self.n_splits)
        splits = [
            (np.setdiff1d(np.arange(len(y)), np.concatenate([pos_folds[i], neg_folds[i]])),
             np.concatenate([pos_folds[i], neg_folds[i]])) for i in range(self.n_splits)
        ]
        return splits

    def cross_validate(self, X, y, model_params):
        """Perform cross-validation for a single set of hyperparameters."""
        metrics = {'F1Score': [], 'loss': []}

        for train_idx, val_idx in self._stratified_split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.model_class(**model_params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val, y_val=y_val)

            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            if valid_indices:
                y_val_filtered = y_val[valid_indices]
                predictions_filtered = np.array(predictions)[valid_indices]

                f1_score = F1Score()(y_val_filtered, predictions_filtered)
                log_loss_score = LogLoss()(y_val_filtered, model._sigmoid(np.dot(X_val[valid_indices], model.weights) + model.bias))

                if f1_score is not None:
                    metrics['F1Score'].append(f1_score)
                if log_loss_score is not None:
                    metrics['loss'].append(log_loss_score)

        avg_f1score = np.mean(metrics['F1Score']) if metrics['F1Score'] else float('-inf')
        avg_log_loss = np.mean(metrics['loss']) if metrics['loss'] else float('inf')

        return avg_f1score, avg_log_loss
