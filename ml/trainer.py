import numpy as np
from ml.learning_curve import plot_learning_curves
from ml.data_processing import stratified_train_validation_split
from ml.logistic_regression import LogisticRegression
from ml.metric import (Accuracy, Precision, Recall, F1Score, ConfusionMatrix,
                       ROCAUC, LogLoss)


def train_final_model(X_train, y_train, best_hparams) -> LogisticRegression:
    """Train the final model with the best hyperparameters.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        best_hparams (dict): Best hyperparameters.

        Returns:
            LogisticRegression: Trained model.
        """

    X_train, X_val, y_train, y_val = stratified_train_validation_split(
        X_train, y_train, val_size=0.2)

    final_model = LogisticRegression(
        learning_rate=best_hparams['learning_rate'],
        n_iterations=best_hparams['n_iterations'],
        regularization=best_hparams['regularization'],
        l1_ratio=best_hparams['l1_ratio'],
        epochs=50,
        decay_type=best_hparams['decay_type'],
        decay_rate=best_hparams['decay_rate']
    )

    final_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    print("Final model training complete.")

    plot_learning_curves(train_losses=final_model.train_losses,
                         val_losses=final_model.val_losses)
    print("Learning curves plotted.")
    return final_model


def evaluate_model(final_model, X_test: np.ndarray, y_test: np.ndarray
                   ) -> np.ndarray:
    """Evaluate the model on the test dataset.

    Args:
        final_model (LogisticRegression): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.

    Returns:
        np.ndarray: Predictions on the test dataset."""

    print("\nEvaluating the model on the test dataset...")
    
    predictions = final_model.predict(X_test, y_val=y_test)

    metrics_calculators = {
        'Accuracy': Accuracy(),
        'Precision': Precision(),
        'Recall': Recall(),
        'F1 Score': F1Score(),
        'ROC-AUC': ROCAUC(),
        'Log Loss': LogLoss(),
        'Confusion Matrix': ConfusionMatrix(),
    }

    print("\nModel Performance Metrics:")
    for name, calculator in metrics_calculators.items():
        if name == 'Log Loss':
            score = calculator(y_test, final_model._sigmoid(np.dot(X_test,
                               final_model.weights) + final_model.bias))
        else:
            score = calculator(y_test, predictions)

        if name == 'Confusion Matrix':
            print(f"{name}: \n{score}")
        else:
            print(f"{name}: {score:.4f}")

    return predictions
