from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def preprocess_features(
    features: List[Feature],
    dataset: Dataset) -> List[
        Tuple[str, np.ndarray, dict]]:
    """Preprocess features.
    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.
    Returns:
        List[str, Tuple[np.ndarray, dict]]: List of preprocessed features.
        Each ndarray of shape (N, ...)
    """
    results = []
    raw = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(
                raw[feature.name].values.reshape(-1, 1)).toarray()
            aritfact = {
                "type": "OneHotEncoder", "encoder": encoder.get_params()}
            results.append((feature.name, data, aritfact))
        if feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1))
            artifact = {
                "type": "StandardScaler", "scaler": scaler.get_params()}
            results.append((feature.name, data, artifact))
    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results


def check_multicollinearity(data, threshold=5.0) -> bool:
    """
    Checks for multicollinearity in the given DataFrame using Variance
    Inflation Factor (VIF).

    Parameters:
    - data: pd.DataFrame - The dataset with features to check for
      multicollinearity.
    - threshold: float - VIF threshold above which a feature is considered
      highly collinear.

    Returns:
    - True if there are any features with VIF exceeding the threshold.
    """

    # Ensure the data contains only numeric columns
    numerical_data = data.select_dtypes(include=[float, int])

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = numerical_data.columns
    vif_data["VIF"] = [variance_inflation_factor(numerical_data.values, i) for
                       i in range(len(numerical_data.columns))]

    # Identify features with VIF greater than the threshold
    collinear_features = vif_data[vif_data["VIF"] > threshold]["feature"
                                                               ].tolist()
    return True if collinear_features else False
