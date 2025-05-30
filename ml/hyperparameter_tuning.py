from itertools import product
from ml.cross_validation import CrossValidator
from ml.logistic_regression import LogisticRegression
from typing import Dict, List, Any
import numpy as np


class HyperparameterTuner:
    """
    A class to perform hyperparameter tuning using cross-validation.

    Attributes:
        cross_validator (CrossValidator): The cross-validation instance.
        hparam_grid (Dict[str, List[Any]]): Dictionary containing
        hyperparameter values to tune.
    """
    def __init__(self, cross_validator: CrossValidator, hparam_grid: Dict[
                str, List[Any]]) -> None:
        """
        Initializes the HyperparameterTuner.

        Args:
            cross_validator (CrossValidator): The cross-validation instance.
            hparam_grid (Dict[str, List[Any]]): Dictionary containing
            hyperparameter values to tune.
        """
        self.cross_validator = cross_validator
        self.hparam_grid = hparam_grid

    def tune(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """
        Performs hyperparameter tuning by cross-validating each hyperparameter
        combination.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            List[Dict[str, Any]]: Top 5 hyperparameter configurations balancing
            F1-score and log loss.
        """
        hparam_combinations = list(product(*self.hparam_grid.values()))
        hparam_names = list(self.hparam_grid.keys())

        results = []

        for iteration, hparams in enumerate(hparam_combinations, 1):
            print(f"Tuning hyperparameters {hparams}. Process {iteration}/{len(
                hparam_combinations)}.")
            hparam_dict = dict(zip(hparam_names, hparams))

            avg_f1score, avg_log_loss = self.cross_validator.cross_validate(
                X, y, hparam_dict)

            results.append({
                'hparams': hparam_dict,
                'f1score': avg_f1score,
                'log_loss': avg_log_loss
            })

        # Filter configurations with log_loss ≤ 1.0
        valid_results = [r for r in results if r['log_loss'] <= 1.0]

        # Sort by highest F1Score, then lowest Log Loss
        top_5_balanced = sorted(valid_results, key=lambda x: (-x['f1score'],
                                x['log_loss']))[:5]

        print("\nTop 5 Hyperparameter Configurations Balancing F1Score & Log"
              " Loss (≤ 1.0):")
        for rank, res in enumerate(top_5_balanced, 1):
            if res['hparams'] is not None:
                print(f"{rank}. F1Score: {res['f1score']:.4f} | Log Loss: "
                      f"{res['log_loss']:.4f} | HParams: {res['hparams']}")
            else:
                print(f"{rank}. No valid configuration found.")

        return top_5_balanced


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         hparam_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning and returns the best hyperparameters.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        hparam_grid (Dict[str, List[Any]]): Hyperparameter search space.

    Returns:
        Dict[str, Any]: The best hyperparameters found.
    """
    cross_validator = CrossValidator(model_class=LogisticRegression)
    tuner = HyperparameterTuner(cross_validator=cross_validator,
                                hparam_grid=hparam_grid)

    print("\nPerforming Hyperparameter Tuning...")
    top_5_results = tuner.tune(X_train, y_train)

    best_hparams = top_5_results[0]['hparams']
    print("\nBest Hyperparameters:")
    for param, value in best_hparams.items():
        print(f"{param}: {value}")

    return best_hparams
