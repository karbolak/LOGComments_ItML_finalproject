import numpy as np
from ml.metric import F1Score, LogLoss
from ml.logistic_regression import LogisticRegression
from typing import Dict, Any, List, Tuple


class CrossValidator:
    """Class for cross-validation.

    Attributes:
        n_splits (int): Number of splits.
        model_class (class): Model class to use for cross-validation.
    """
    def __init__(self, model_class=LogisticRegression, n_splits: int = 5
                 ) -> None:
        """
        Initializes the CrossValidator.

        Args:
            n_splits (int): Number of splits.
            model_class (class): Model class to use for cross-validation.

        Returns:
            None
        """
        self.n_splits = n_splits
        self.model_class = model_class

    def _stratified_split(self, X: np.ndarray, y: np.ndarray
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform stratified split of the dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            list: List of tuples containing train and validation indices."""
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        pos_folds = np.array_split(pos_indices, self.n_splits)
        neg_folds = np.array_split(neg_indices, self.n_splits)
        splits = [
            (np.setdiff1d(np.arange(len(y)), np.concatenate([pos_folds[i],
                                                            neg_folds[i]])),
             np.concatenate([pos_folds[i], neg_folds[i]])) for i in range(
                self.n_splits)
        ]
        return splits

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       model_params: Dict[str, Any]) -> Tuple[float, float]:
        """Perform cross-validation for a single set of hyperparameters.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            model_params (Dict[str, Any]): Model hyperparameters.

        Returns:
            Tuple[float, float]: Average F1 score and log loss across all
            folds.
        """
        metrics = {'F1Score': [], 'loss': []}

        for train_idx, val_idx in self._stratified_split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.model_class(**model_params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val, y_val=y_val)

            valid_indices = [i for i, p in enumerate(predictions
                                                     ) if p is not None]
            if valid_indices:
                y_val_filtered = y_val[valid_indices]
                predictions_filtered = np.array(predictions)[valid_indices]

                f1_score = F1Score()(y_val_filtered, predictions_filtered)
                log_loss_score = LogLoss()(y_val_filtered, model._sigmoid(
                    np.dot(X_val[valid_indices], model.weights) + model.bias))

                if f1_score is not None:
                    metrics['F1Score'].append(f1_score)
                if log_loss_score is not None:
                    metrics['loss'].append(log_loss_score)

        avg_f1score = np.mean(metrics['F1Score']
                              ) if metrics['F1Score'] else float('-inf')
        avg_log_loss = np.mean(metrics['loss']
                               ) if metrics['loss'] else float('inf')

        return avg_f1score, avg_log_loss
