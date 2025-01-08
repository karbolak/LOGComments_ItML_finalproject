from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Optional


class Model(ABC):
    """Base class for all machine learning models."""

    def __init__(self, model_type: Literal["regression", "classification"],
                 parameters: Optional[dict] = None):
        self.type = model_type
        self.parameters = parameters if parameters is not None else {}
        self.trained = False  # Flag to indicate if the model has been trained

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Target values.
        """
        self.trained = True  # Set to True when model is trained

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model on given data.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.trained:
            raise ValueError("Model must be trained before making "
                             "predictions.")

    def to_artifact(self, name: str) -> Artifact:
        """Convert the model to an Artifact for storage.

        Args:
            name (str): Name of the artifact.

        Returns:
            Artifact: An artifact representing the model's state.
        """
        # Serialize the model's parameters and type
        data = {
            "type": self.type,
            "parameters": self.parameters,
            "trained": self.trained
        }
        return Artifact(name=name, data=deepcopy(data), version="1.0.0",
                        type="model:base")

    def load_artifact(self, artifact: Artifact) -> None:
        """Load model parameters from an artifact.

        Args:
            artifact (Artifact): Artifact to load data from.
        """
        if artifact.type.startswith("model:"):
            data = artifact.data
            self.type = data.get("type", self.type)
            self.parameters = data.get("parameters", {})
            self.trained = data.get("trained", False)
        else:
            raise ValueError("Invalid artifact type. Expected a model "
                             "artifact.")

    def __str__(self) -> str:
        return (f"Model(type={self.type}, trained={self.trained} parameters={
                self.parameters})")
