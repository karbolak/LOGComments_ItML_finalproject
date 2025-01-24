import numpy as np
from ml.model import Model


class LogisticRegression(Model):
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0):
        super().__init__(model_type="classification")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if y.ndim > 1 and y.shape[1] == 2:
            y = y[:, 1]

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            # Compute gradients with L2 regularization
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + self.regularization * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return np.where(predictions >= 0.5, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
