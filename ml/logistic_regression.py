import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0,
                 l1_ratio=0.5, epochs=1, decay_type="none", decay_rate=0.01):
        """
        Logistic Regression model with Elastic Net regularization and learning rate decay.

        Args:
            learning_rate (float): Initial learning rate.
            n_iterations (int): Number of iterations per epoch.
            regularization (float): Regularization strength (L1 + L2).
            l1_ratio (float): Proportion of L1 penalty in Elastic Net (0: pure L2, 1: pure L1).
            epochs (int): Number of training epochs.
            decay_type (str): Type of learning rate decay ("none", "time", "exponential").
            decay_rate (float): Rate of learning rate decay.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.l1_ratio = l1_ratio
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
        else:
            return self.learning_rate 

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None, patience=5):
        """
        Fit the logistic regression model with Elastic Net regularization and optional early stopping.
        """
        if y.ndim > 1 and y.shape[1] == 2:
            y = y[:, 1]  # Convert one-hot to single-class labels if necessary

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.train_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            for iteration in range(self.n_iterations):
                t = epoch * self.n_iterations + iteration
                current_lr = self._update_learning_rate(t)
                
                linear_output = np.dot(X, self.weights) + self.bias
                predictions = self._sigmoid(linear_output)
                
                # Compute Elastic Net penalties
                l1_penalty = self.l1_ratio * np.sign(self.weights)
                l2_penalty = (1 - self.l1_ratio) * self.weights
                
                # Gradient calculation with Elastic Net regularization
                dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + self.regularization * (l1_penalty + l2_penalty)
                db = (1 / n_samples) * np.sum(predictions - y)
                
                # Update weights and bias
                self.weights -= current_lr * dw
                self.bias -= current_lr * db

            # Compute training loss
            train_loss = self._compute_loss(y, self._sigmoid(np.dot(X, self.weights) + self.bias))
            self.train_losses.append(train_loss)
            
            # Compute validation loss (if validation data is provided)
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_loss = self._compute_loss(y_val, val_predictions)
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")

                min_delta = 1e-3
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}")
                        return  
            else:
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

    def _compute_loss(self, y, predictions):
        """Calculate log-loss with Elastic Net regularization."""
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)
        log_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Elastic Net penalty
        l1_penalty = self.l1_ratio * np.sum(np.abs(self.weights))
        l2_penalty = (1 - self.l1_ratio) * np.sum(self.weights ** 2)
        
        return log_loss + self.regularization * (l1_penalty + l2_penalty)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def find_optimal_threshold(y_true, probabilities):
        """Computes the optimal classification threshold without sklearn."""
        sorted_indices = np.argsort(-probabilities)
        y_sorted = y_true[sorted_indices]
        cum_true_positives = np.cumsum(y_sorted)
        cum_false_positives = np.cumsum(1 - y_sorted)
        
        true_positive_rates = cum_true_positives / np.max(cum_true_positives)
        false_positive_rates = cum_false_positives / np.max(cum_false_positives)
        
        optimal_idx = np.argmax(true_positive_rates - false_positive_rates)
        threshold = probabilities[sorted_indices][optimal_idx]
        return threshold

    def predict(self, X: np.ndarray, y_val: np.ndarray = None, threshold=None):
        """Predict class labels using an optimal threshold if available."""
        probabilities = self._sigmoid(np.dot(X, self.weights) + self.bias)
        
        if threshold is None and y_val is not None and len(y_val) == len(probabilities):
            threshold = self.find_optimal_threshold(y_val, probabilities)
        
        if threshold is None:
            threshold = 0.5  # Default threshold
        
        return np.where(probabilities >= threshold, 1, 0)
