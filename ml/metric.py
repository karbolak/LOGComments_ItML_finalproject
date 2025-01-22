from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric given ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated metric.
        """
        pass

class Accuracy(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

class Precision(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

class Recall(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

class F1Score(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = Precision()(y_true, y_pred)
        recall = Recall()(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

class ConfusionMatrix(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        true_negatives = np.sum((y_pred == 0) & (y_true == 0))
        false_negatives = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

class ROCAUC(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sorted_indices = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_indices]
        cum_true_positives = np.cumsum(y_true_sorted)
        cum_false_positives = np.cumsum(1 - y_true_sorted)

        true_positive_rates = cum_true_positives / cum_true_positives[-1]
        false_positive_rates = cum_false_positives / cum_false_positives[-1]

        return np.trapz(true_positive_rates, false_positive_rates)

class LogLoss(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
