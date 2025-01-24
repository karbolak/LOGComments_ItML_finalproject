import numpy as np
from ml.model import Model


class LogisticRegression(Model):
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0,
                 epochs=1, decay_type="none", decay_rate=0.01):
        """
        Logistic Regression model with learning rate decay.
        
        Args:
            learning_rate (float): Initial learning rate.
            n_iterations (int): Number of iterations for gradient descent per epoch.
            regularization (float): L2 regularization strength.
            epochs (int): Number of epochs for training.
            decay_type (str): Type of learning rate decay ('time', 'exponential', 'step').
            decay_rate (float): Rate of learning rate decay.
        """
        super().__init__(model_type="classification")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.epochs = epochs
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None
        
    def _update_learning_rate(self, t):
        """Update the learning rate using the specified decay type."""
        if self.decay_type == "time":
            return self.learning_rate / (1 + self.decay_rate * t)
        elif self.decay_type == "exponential":
            return self.learning_rate * np.exp(-self.decay_rate * t)
        elif self.decay_type == "step":
            return self.learning_rate * (0.5 ** (t // 100))
        elif self.decay_type == "none":
            return self.learning_rate 

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None):
        """
        Fit the logistic regression model to the training data and optionally track validation loss.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            validation_data (tuple): Optional (X_val, y_val) for tracking validation loss.
        """
        if y.ndim > 1 and y.shape[1] == 2:
            y = y[:, 1]  # Convert one-hot to single-class labels if necessary

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.train_losses = []  # Store training loss for each epoch
        self.val_losses = []    # Store validation loss for each epoch (if validation data is provided)

        for epoch in range(self.epochs):
            for iteration in range(self.n_iterations):
                t = epoch * self.n_iterations + iteration  # Total iteration count
                current_lr = self._update_learning_rate(t)

                model = np.dot(X, self.weights) + self.bias
                predictions = self._sigmoid(model)

                # Compute gradients with L2 regularization
                dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + self.regularization * self.weights
                db = (1 / n_samples) * np.sum(predictions - y)

                # Update weights and bias using the current learning rate
                self.weights -= current_lr * dw
                self.bias -= current_lr * db

            # Compute and log training loss
            train_loss = self._compute_loss(y, self._sigmoid(np.dot(X, self.weights) + self.bias))
            self.train_losses.append(train_loss)

            # Compute and log validation loss (if provided)
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self._compute_loss(y_val, self._sigmoid(np.dot(X_val, self.weights) + self.bias))
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

        self.trained = True


    def _compute_loss(self, y, predictions):
        """Calculate loss using log-loss function."""
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """
        Predict class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            threshold (float): Classification threshold.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        model = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(model)
        return np.where(probabilities >= threshold, 1, 0)
