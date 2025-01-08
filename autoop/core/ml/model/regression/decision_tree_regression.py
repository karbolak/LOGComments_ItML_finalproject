import numpy as np
from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegression(Model):
    def __init__(self, max_depth=None):
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before making "
                             "predictions.")
        return self.model.predict(X)
