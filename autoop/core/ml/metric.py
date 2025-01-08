from abc import ABC, abstractmethod
import numpy as np

REGRESSION_METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
]

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
]

LOG_CLASSIFICATION_METRICS = [
    "logistic_accuracy",
    "logistic_precision",
    "logistic_recall"
]


def get_metric(name: str):
    """Factory function to get a metric by name."""
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "logistic_accuracy":
        return LogisticAccuracy()
    elif name == "logistic_precision":
        return LogisticPrecision()
    elif name == "logistic_recall":
        return LogisticRecall()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
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


# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def __str__(self):
        return "MeanSquaredError"


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def __str__(self):
        return "MeanAbsoluteError"


class RSquared(Metric):
    """R-squared (R²) metric for regression, measuring goodness of fit."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    def __str__(self):
        return "RSquared"


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


# OLD Classification Metrics
class Accuracy(Metric):
    """Accuracy metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def __str__(self):
        return "LogisticAccuracy"


class Precision(Metric):
    """Precision metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (true_positives / predicted_positives if
                predicted_positives != 0 else 0.0)

    def __str__(self):
        return "Precision"


class Recall(Metric):
    """Recall metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (true_positives / actual_positives
                if actual_positives != 0 else 0.0)

    def __str__(self):
        return "Recall"
