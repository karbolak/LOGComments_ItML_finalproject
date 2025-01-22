from abc import ABC, abstractmethod
import numpy as np

LOG_CLASSIFICATION_METRICS = [
    "logistic_accuracy",
    "logistic_precision",
    "logistic_recall"
]


def get_metric(name: str):
    """Factory function to get a metric by name."""
    if name == "logistic_accuracy":
        return LogisticAccuracy()
    elif name == "logistic_precision":
        return LogisticPrecision()
    elif name == "logistic_recall":
        return LogisticRecall()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


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


class LogisticAccuracy(Metric):
    """Accuracy metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return float(np.mean(y_true == y_pred))

    def __str__(self):
        return "LogisticAccuracy"


class LogisticPrecision(Metric):
    """Precision metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (true_positives / predicted_positives
                if predicted_positives != 0 else 0.0)

    def __str__(self):
        return "LogisticPrecision"


class LogisticRecall(Metric):
    """Recall metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (true_positives / actual_positives
                if actual_positives != 0 else 0.0)

    def __str__(self):
        return "LogisticRecall"