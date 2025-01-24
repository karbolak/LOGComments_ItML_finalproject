import numpy as np
import matplotlib.pyplot as plt
from ml.metric import LogLoss

class LearningCurve:
    def __init__(self, model_class, learning_rate, n_iterations):
        self.model_class = model_class
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def generate(self, X, y, train_sizes=None):
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_losses, val_losses = [], []

        for size in train_sizes:
            n_train_samples = int(len(X) * size)
            if n_train_samples == 0:
                continue  # Skip if no training samples are selected

            indices = np.random.permutation(len(X))
            X_train = X[indices[:n_train_samples]]
            y_train = y[indices[:n_train_samples]]
            X_val = X[indices[n_train_samples:]]
            y_val = y[indices[n_train_samples:]]

            # Skip if validation set is empty
            if len(X_val) == 0 or len(y_val) == 0:
                continue

            model = self.model_class(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations
            )
            model.fit(X_train, y_train)

            train_loss = LogLoss()(y_train, model._sigmoid(np.dot(X_train, model.weights) + model.bias))
            val_loss = LogLoss()(y_val, model._sigmoid(np.dot(X_val, model.weights) + model.bias))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if not train_losses or not val_losses:
            raise ValueError("No valid train or validation losses computed. Check your data or train sizes.")

        return train_sizes[:len(train_losses)], train_losses, val_losses


def plot_learning_curves(train_sizes, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(train_sizes, val_losses, label='Validation Loss', color='red', marker='o')
    plt.title('Learning Curves (Loss Comparison)')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/learning_curves_loss.png')
    plt.close()
