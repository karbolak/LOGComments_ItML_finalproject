import numpy as np
import matplotlib.pyplot as plt
from ml.metric import LogLoss

class LearningCurve:
    def __init__(self, model_class, learning_rate, n_iterations, regularization, epochs):
        self.model_class = model_class
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.epochs = epochs

    def generate(self, X, y):
        """Generates training and validation loss across epochs."""
        train_losses = []
        val_losses = []

        # Split data into training and validation
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * 0.8)  # 80% train, 20% validation
        X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]

        # Initialize the model
        model = self.model_class(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularization=self.regularization,
            epochs=1,  # One epoch per iteration for tracking
        )

        for epoch in range(1, self.epochs + 1):
            model.fit(X_train, y_train)  # Train for one epoch

            # Calculate training and validation loss
            train_loss = LogLoss()(y_train, model._sigmoid(np.dot(X_train, model.weights) + model.bias))
            val_loss = LogLoss()(y_val, model._sigmoid(np.dot(X_val, model.weights) + model.bias))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Optionally log progress
            print(f"Epoch {epoch}/{self.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        return list(range(1, self.epochs + 1)), train_losses, val_losses


def plot_learning_curves(epochs, train_losses, val_losses):
    """Plots loss vs epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    plt.title('Learning Curves (Loss vs Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/learning_curves_epoch_loss.png')
    plt.close()
